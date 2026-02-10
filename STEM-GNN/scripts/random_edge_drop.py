#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import warnings
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb
import yaml
from torch_geometric.utils import to_undirected, coalesce

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from task.node import ft_node, eval_node
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.others import seed_everything, load_params, freeze_params, ensure_finetune_lr, get_pretrain_run_id
from utils.preprocess import pre_node

warnings.filterwarnings("ignore")

dataset2task = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
}


def _num_nodes(data) -> int:
    n = getattr(data, "num_nodes", None)
    if n is not None:
        return int(n)
    if hasattr(data, "node_text_feat") and isinstance(getattr(data, "node_text_feat"), torch.Tensor):
        return int(getattr(data, "node_text_feat").size(0))
    if hasattr(data, "x") and isinstance(getattr(data, "x"), torch.Tensor):
        return int(getattr(data, "x").size(0))
    raise ValueError("Cannot infer number of nodes from data.")


def _to_bool_mask(idx_or_mask, n: int, device: torch.device) -> torch.Tensor:
    if idx_or_mask is None:
        return torch.zeros(n, dtype=torch.bool, device=device)

    if isinstance(idx_or_mask, torch.Tensor) and idx_or_mask.dtype == torch.bool:
        mask = idx_or_mask
        if mask.dim() > 1:
            raise ValueError(f"Expected 1-D bool mask; got shape {tuple(mask.shape)}")
        return mask.to(device)

    if isinstance(idx_or_mask, (list, tuple, np.ndarray)):
        idx_or_mask = torch.as_tensor(idx_or_mask, dtype=torch.long, device=device)

    if isinstance(idx_or_mask, torch.Tensor):
        idx = idx_or_mask.to(dtype=torch.long, device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        if idx.numel() > 0:
            mask[idx] = True
        return mask

    raise TypeError(f"Unsupported split type: {type(idx_or_mask)}")


def _extract_masks_train_valid_test(split: dict, data, device: torch.device):
    n = _num_nodes(data)

    tr = split.get("train", None)
    va = split.get("valid", split.get("val", None))
    te = split.get("test", None)

    if tr is None and hasattr(data, "train_mask"):
        tr = getattr(data, "train_mask")
    if va is None and hasattr(data, "val_mask"):
        va = getattr(data, "val_mask")
    if te is None and hasattr(data, "test_mask"):
        te = getattr(data, "test_mask")

    train_mask = _to_bool_mask(tr, n, device)
    valid_mask = _to_bool_mask(va, n, device)
    test_mask = _to_bool_mask(te, n, device)
    return train_mask, valid_mask, test_mask


def _edge_candidate_mask(edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    if node_mask is None:
        return torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    if node_mask.dtype != torch.bool:
        node_mask = node_mask.to(dtype=torch.bool)
    node_mask = node_mask.to(device=edge_index.device)
    src, dst = edge_index
    return node_mask[src] | node_mask[dst]


def _apply_random_edge_drops(
    data,
    split_masks: Dict[str, torch.Tensor],
    *,
    drop_prob: float = 0.2,
    seed: int = 1,
    perturb: str = "test",
    drop_mode: str = "per_undirected",
    debug: bool = False,
):
    drop_prob = float(drop_prob)
    if drop_prob <= 0.0:
        return data.clone()
    drop_prob = min(drop_prob, 1.0)

    out = data.clone()
    edge_index = out.edge_index
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("data.edge_index must be a torch.Tensor")

    num_nodes = _num_nodes(out)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    out.edge_index = edge_index

    num_edges = edge_index.size(1)
    if num_edges == 0:
        return out

    device = edge_index.device

    train_mask = split_masks.get("train")
    valid_mask = split_masks.get("valid", split_masks.get("val"))
    test_mask = split_masks.get("test")

    if perturb == "test":
        node_scope = test_mask
    elif perturb in ("valtest", "val_test"):
        node_scope = valid_mask | test_mask if valid_mask is not None else test_mask
    elif perturb == "all":
        node_scope = torch.ones(num_nodes, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unsupported perturb option: {perturb}")

    node_scope = _to_bool_mask(node_scope, num_nodes, device)
    candidate_mask = _edge_candidate_mask(edge_index, node_scope)
    candidate_total = int(candidate_mask.sum().item())
    if candidate_total == 0:
        if debug:
            print("[edge-drop] No candidate edges to perturb; returning original graph.")
        return out

    generator = torch.Generator(device=device) if edge_index.is_cuda else torch.Generator()
    generator.manual_seed(int(seed))

    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"per_undirected", "per_edge"}:
        raise ValueError(f"Unsupported drop_mode: {drop_mode}")

    drop_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)

    src, dst = edge_index
    undirected_src = torch.minimum(src, dst)
    undirected_dst = torch.maximum(src, dst)
    undirected_key = undirected_src.to(torch.long) * num_nodes + undirected_dst.to(torch.long)

    cand_idx = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    dropped_units = 0
    total_units = 0

    if drop_mode == "per_edge":
        if cand_idx.numel() > 0:
            try:
                edge_rand = torch.rand(cand_idx.numel(), device=edge_index.device, generator=generator)
            except TypeError:
                edge_rand = torch.rand(cand_idx.numel(), device=edge_index.device)
            drop_flags = edge_rand < drop_prob
            if drop_flags.all():
                keep_edge = torch.argmax(edge_rand)
                drop_flags[keep_edge] = False
            drop_mask[cand_idx] = drop_flags
            dropped_units = int(drop_flags.sum().item())
            total_units = cand_idx.numel()
    else:  # per_undirected
        if cand_idx.numel() > 0:
            cand_keys = undirected_key[cand_idx]
            unique_keys, inverse = torch.unique(cand_keys, return_inverse=True)
            total_units = unique_keys.size(0)
            try:
                pair_rand = torch.rand(total_units, device=edge_index.device, generator=generator)
            except TypeError:
                pair_rand = torch.rand(total_units, device=edge_index.device)
            drop_pairs = pair_rand < drop_prob
            if drop_pairs.all():
                keep_pair = torch.argmax(pair_rand)
                drop_pairs[keep_pair] = False
            drop_mask[cand_idx] = drop_pairs[inverse]
            dropped_units = int(drop_pairs.sum().item())

    keep_mask = ~drop_mask

    dropped_edges = int(drop_mask.sum().item())
    out.edge_index = edge_index[:, keep_mask].contiguous()

    if hasattr(out, "xe") and isinstance(out.xe, torch.Tensor) and out.xe.size(0) == num_edges:
        out.xe = out.xe[keep_mask]

    if hasattr(out, "edge_attr") and isinstance(out.edge_attr, torch.Tensor) and out.edge_attr.size(0) == num_edges:
        out.edge_attr = out.edge_attr[keep_mask]

    for key in out.keys:
        if key in {"edge_index", "xe", "edge_attr", "edge_text_feat"}:
            continue
        value = getattr(out, key)
        if isinstance(value, torch.Tensor) and value.size(0) == num_edges:
            setattr(out, key, value[keep_mask])

    if debug:
        actual = drop_mask[candidate_mask].float().mean().item()
        unit_ratio = (dropped_units / total_units) if total_units else 0.0
        unit_label = "pair" if drop_mode == "per_undirected" else "edge"
        print(
            f"[edge-drop] p={drop_prob:.4f} -> dropped {dropped_edges}/{candidate_total} candidate edges "
            f"(edge_ratio={actual:.4f}, {unit_label}_ratio={unit_ratio:.4f}) seed={seed}"
        )

    return out


def build_parser():
    import argparse

    parser = argparse.ArgumentParser("Finetune (node only) with random edge-drop evaluation")

    # General Parameters
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument("--setting", type=str, default="standard")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    # Few-shot Parameters
    parser.add_argument("--n_task", type=int, default=20)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--n_query", type=int, default=3)

    # Pre-train Parameters
    parser.add_argument("--pretrain_dataset", "--pt_data", type=str, default="all")
    parser.add_argument("--pretrain_task", "--pt_task", type=str, default="all")
    parser.add_argument("--pretrain_model_epoch", "--pt_epochs", type=int, default=25)
    parser.add_argument("--pretrain_seed", "--pt_seed", type=int, default=42)
    parser.add_argument("--pretrain_path", type=str, default="")

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", "--act", type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # VQ Parameters
    parser.add_argument("--code_dim", type=int, default=768)
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--codebook_head", type=int, default=4)
    parser.add_argument("--codebook_decay", type=float, default=0.8)
    parser.add_argument("--commit_weight", type=float, default=0.25)
    parser.add_argument("--ortho_reg_weight", type=float, default=1)
    parser.add_argument("--ortho_reg_max_codes", type=int, default=32)

    # MoE Parameters
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--moe_layers", type=str, default="none", choices=["none", "all", "last"])
    parser.add_argument("--moe_experts", "--K", type=int, default=3)
    parser.add_argument("--moe_tau", "--tau", type=float, default=1.0)
    parser.add_argument("--lamda_env", type=float, default=0.0)

    # Fine-Tune Parameters
    parser.add_argument("--finetune_dataset", "--dataset", "--data", type=str, default="cora")
    parser.add_argument(
        "--freeze_vq",
        type=int,
        default=1,
        choices=[0, 1],
        help="Freeze VQ parameters during finetuning (1: freeze [default], 0: update).",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--finetune_epochs", "--epochs", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--finetune_lr", "--lr", type=float, default=1e-3)

    # Model Parameters
    parser.add_argument("--separate_decoder_for_each_head", type=bool, default=True)
    parser.add_argument("--decoder_jac_coeff", type=float, default=0.0)

    # Edge-drop Noise Parameters
    parser.add_argument("--edge_drop_prob", type=float, default=0.2, help="Probability of dropping a candidate edge during evaluation.")
    parser.add_argument("--edge_drop_seed", type=int, default=1, help="Base random seed for edge drop sampling.")
    parser.add_argument("--perturb", choices=["test", "valtest", "all"], default="test", help="Which node partitions induce edge drops.")
    parser.add_argument(
        "--edge_drop_mode",
        choices=["per_undirected", "per_edge"],
        default="per_undirected",
        help="per_undirected deletes undirected edges (pair-wise) with rate p; per_edge drops directed edges independently.",
    )
    parser.add_argument("--include_val_drop", action="store_true", help="Log validation scores when edges are dropped on validation nodes.")
    parser.add_argument("--debug_edge_drop", action="store_true", help="Print debugging info for the first noisy graph.")
    parser.add_argument("--save_tsv", action="store_true", help="Save edge-drop evaluation summary as TSV.")
    parser.add_argument("--tsv_name", type=str, default="")

    return parser


_MOE_PARAM_KEYS = ("moe", "moe_layers", "moe_experts", "moe_tau", "lamda_env")


def get_args_edge_drop(default_params=None):
    parser = build_parser()
    if default_params:
        parser.set_defaults(**default_params)
    args = parser.parse_args()
    args_dict = vars(args)

    moe_defaults = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if dest in _MOE_PARAM_KEYS:
            moe_defaults[dest] = action.default
    args_dict["_moe_parser_defaults"] = moe_defaults
    return args_dict


def maybe_infer_moe_settings(
    path: str,
    target: Dict[str, object],
    *,
    defaults: Optional[Dict[str, object]] = None,
    protected_keys: Iterable[str] = (),
):
    if not path:
        return
    defaults = defaults or {}
    protected = set(protected_keys or ())
    missing = object()
    name = osp.basename(path.rstrip("/"))

    def maybe_set(key: str, value):
        if key in protected:
            return

        current = target.get(key, missing)
        default_val = defaults.get(key, missing)

        should_set = current is missing
        if not should_set and default_val is not missing and current == default_val:
            should_set = True
        if not should_set and isinstance(current, str) and current.lower() == "auto":
            should_set = True

        if not should_set:
            return

        if current is missing:
            current_repr = "unset"
        else:
            current_repr = repr(current)

        if current != value:
            print(f"[info] Inferred {key}={value} from checkpoint name (previous: {current_repr}).")
        target[key] = value

    import re

    idx = name.find("moe_")
    if idx >= 0:
        suffix = name[idx + 4 :]
        parts = suffix.split("_")
        if parts and parts[0].isdigit():
            maybe_set("moe", parts[0] != "0")

    if not target.get("moe", False):
        return

    layers_match = re.search(r"layers_([A-Za-z]+)", name)
    if layers_match:
        maybe_set("moe_layers", layers_match.group(1).lower())
    experts_match = re.search(r"_K_(\d+)", name)
    if experts_match:
        maybe_set("moe_experts", int(experts_match.group(1)))
    tau_match = re.search(r"_tau_([0-9.]+)", name)
    if tau_match:
        tau_val = float(tau_match.group(1))
        maybe_set("moe_tau", tau_val)
        maybe_set("tau", tau_val)
    lam_match = re.search(r"_lam_([0-9.]+)", name)
    if lam_match:
        maybe_set("lamda_env", float(lam_match.group(1)))


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    moe_defaults = params.pop("_moe_parser_defaults", {}) or {}
    if not isinstance(moe_defaults, dict):
        try:
            moe_defaults = dict(moe_defaults)
        except TypeError:
            moe_defaults = {}

    if params.get("task") != "node":
        raise NotImplementedError("random_edge_drop.py supports node classification only.")

    preprocess = pre_node
    finetune = ft_node
    evaluate = eval_node

    data_name = params["finetune_dataset"]
    task = params["task"]
    setting = params["setting"]

    dataset, splits, labels, num_classes, _num_tasks = get_finetune_graph(params["data_path"], data_name)
    params["num_classes"] = num_classes

    dataset = preprocess(dataset)
    data = dataset[0]
    data.y = labels

    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
        moe=params.get("moe", False),
        num_experts=params.get("moe_experts", params.get("K", 3)),
        tau=params.get("moe_tau", params.get("tau", 1.0)),
        moe_layers=params.get("moe_layers", "none"),
    )

    vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    pretrain_run_id = get_pretrain_run_id(params)
    pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if pretrain_path.lower() in {"default", "auto"}:
        pretrain_path = ""
    if pretrain_path and not osp.isabs(pretrain_path):
        pretrain_path = osp.join(params["pt_model_path"], pretrain_path)

    if pretrain_path or params["pretrain_dataset"] != "na":
        pretrain_task = params["pretrain_task"]

        if pretrain_path:
            path = pretrain_path
        elif pretrain_task == "all":
            path = osp.join(params["pt_model_path"], pretrain_run_id)
        else:
            raise ValueError("Invalid Pretrain Task")

        encoder_path = osp.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt')
        vq_path = osp.join(path, f'vq_{params["pretrain_model_epoch"]}.pt')

        if not osp.exists(encoder_path):
            raise FileNotFoundError("Cannot find encoder checkpoint. Set --pretrain_path to a valid folder.")
        if not osp.exists(vq_path):
            raise FileNotFoundError("Cannot find vector-quantizer checkpoint. Set --pretrain_path to a valid folder.")

        encoder = load_params(encoder, encoder_path)
        vq = load_params(vq, vq_path)

        print("Loaded pretrained encoder and VQ.")
        protected_moe_keys = {
            key
            for key, default in moe_defaults.items()
            if key in params and params[key] != default
        }
        maybe_infer_moe_settings(path, params, defaults=moe_defaults, protected_keys=protected_moe_keys)

    if params.get("freeze_vq", 1):
        freeze_params(vq)
        print("Freeze VQ parameters during finetuning")

    train_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()
    moe_print_interval = max(1, params["finetune_epochs"] // 20)

    best_state_dicts: List[Dict[str, torch.Tensor]] = []
    eval_splits: List[dict] = []

    if setting != "standard":
        raise ValueError("Edge-drop finetuning only supports the standard setting.")

    for idx, split in enumerate(splits):
        seed_everything(idx)

        task_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        print(
            f"[Finetune] MoE enabled: {getattr(task_model.encoder, 'moe', False)} | "
            f"MoE layers: {getattr(task_model.encoder, 'moe_layer_flags', [])}"
        )

        opt_params = task_model.parameters()
        task_opt = AdamW(opt_params, lr=params["finetune_lr"])
        stopper = EarlyStopping(patience=params["early_stop"])

        if params["batch_size"] != 0:
            train_loader, subgraph_loader = get_loader(data, split, labels, params)

        best_val = -float("inf")
        best_state = None

        for epoch in range(params["finetune_epochs"]):
            loss = finetune(
                model=task_model,
                dataset=data,
                loader=train_loader,
                optimizer=task_opt,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[30] * params["num_layers"],
            )
            moe_stats = task_model.get_moe_usage(reset=True)

            result = evaluate(
                model=task_model,
                dataset=data,
                loader=subgraph_loader,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[-1] * params["num_layers"],
            )
            moe_log = {}
            for stat in moe_stats:
                layer = stat["layer"]
                for expert_idx, value in enumerate(stat["avg_prob"]):
                    moe_log[f"train/moe_layer{layer}_avg_prob_{expert_idx}"] = value
                for expert_idx, value in enumerate(stat["top1_frac"]):
                    moe_log[f"train/moe_layer{layer}_top1_frac_{expert_idx}"] = value

            if result["val"] >= best_val:
                best_val = result["val"]
                best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}

            is_stop = stopper(result)
            should_print_moe = (epoch % moe_print_interval == 0) or is_stop
            logger.log(idx, epoch, loss, result)
            if should_print_moe:
                for stat in moe_stats:
                    avg_str = ", ".join(f"{v:.3f}" for v in stat["avg_prob"])
                    top1_str = ", ".join(f"{v:.3f}" for v in stat["top1_frac"])
                    print(f"[MoE] Layer {stat['layer']} avg_prob=[{avg_str}] top1_frac=[{top1_str}]")
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            log_payload = {
                "train/lin_loss": loss["act_loss"],
                "train/jac_loss": loss["jac_loss"],
                "train/env_loss": loss["env_loss"],
                "train/loss": loss["loss"],
                "train/train_value": result["train"],
                "train/val_value": result["val"],
                "train/test_value": result["test"],
            }
            log_payload.update(moe_log)
            wandb.log(log_payload)

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}

        best_state_dicts.append(best_state)
        eval_splits.append(split)

        single_best = logger.get_single_best(idx)
        wandb.log(
            {
                "best/train": single_best["train"],
                "best/val": single_best["val"],
                "best/test": single_best["test"],
            }
        )

    best = logger.get_best()

    wandb.log(
        {
            "final/train": "{:.2f} ± {:.2f}".format(best["train"]["mean"], best["train"]["std"]),
            "final/val": "{:.2f} ± {:.2f}".format(best["val"]["mean"], best["val"]["std"]),
            "final/test": "{:.2f} ± {:.2f}".format(best["test"]["mean"], best["test"]["std"]),
            "final/train_mean": best["train"]["mean"],
            "final/val_mean": best["val"]["mean"],
            "final/test_mean": best["test"]["mean"],
            "final/train_std": best["train"]["std"],
            "final/val_std": best["val"]["std"],
            "final/test_std": best["test"]["std"],
        }
    )
    wandb.log({"meta/run": logger.get_run_raw(), "meta/best": logger.get_best_raw()})

    if task != "node":
        raise NotImplementedError("random_edge_drop.py supports node classification only.")

    rows: List[dict] = []
    log_val_drop = params.get("include_val_drop", False) or params.get("perturb") != "test"
    drop_prob = float(params.get("edge_drop_prob", 0.2))
    drop_trials = 1

    for split_idx, (split, state_dict) in enumerate(zip(eval_splits, best_state_dicts)):
        eval_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        eval_model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})

        mask_device = device
        train_mask, val_mask, test_mask = _extract_masks_train_valid_test(split, data, mask_device)
        mask_dict = {
            "train": train_mask,
            "valid": val_mask,
            "val": val_mask,
            "test": test_mask,
        }

        trial_metrics = []
        base_seed = int(params.get("edge_drop_seed", 1)) + split_idx * 1000
        for trial in range(drop_trials):
            trial_seed = base_seed + trial
            noisy_data = _apply_random_edge_drops(
                data,
                mask_dict,
                drop_prob=drop_prob,
                seed=trial_seed,
                perturb=params.get("perturb", "test"),
                drop_mode=params.get("edge_drop_mode", "per_undirected"),
                debug=params.get("debug_edge_drop", False) and split_idx == 0 and trial == 0,
            )
            noisy_data = noisy_data.to(device)
            metrics = eval_node(
                model=eval_model,
                dataset=noisy_data,
                loader=None,
                split=mask_dict,
                labels=labels,
                params=params,
                num_neighbors=[-1] * params["num_layers"],
            )
            trial_metrics.append(metrics)

        train_vals = np.array([m["train"] for m in trial_metrics], dtype=float)
        val_vals = np.array([m["val"] for m in trial_metrics], dtype=float)
        test_vals = np.array([m["test"] for m in trial_metrics], dtype=float)

        record = {
            "edge_drop_train_mean": float(train_vals.mean()),
            "edge_drop_train_std": float(train_vals.std()),
            "edge_drop_test_mean": float(test_vals.mean()),
            "edge_drop_test_std": float(test_vals.std()),
            "edge_drop_test_trials": test_vals.tolist(),
        }
        if log_val_drop:
            record["edge_drop_val_mean"] = float(val_vals.mean())
            record["edge_drop_val_std"] = float(val_vals.std())
            record["edge_drop_val_trials"] = val_vals.tolist()
        rows.append(record)

        log_payload = {
            f"edge_drop/test_split{split_idx}_mean": record["edge_drop_test_mean"],
            f"edge_drop/test_split{split_idx}_std": record["edge_drop_test_std"],
            f"edge_drop/train_split{split_idx}_mean": record["edge_drop_train_mean"],
            f"edge_drop/train_split{split_idx}_std": record["edge_drop_train_std"],
        }
        if log_val_drop:
            log_payload[f"edge_drop/val_split{split_idx}_mean"] = record["edge_drop_val_mean"]
            log_payload[f"edge_drop/val_split{split_idx}_std"] = record["edge_drop_val_std"]
        wandb.log(log_payload)

        msg = (
            f"[split {split_idx}] EdgeDrop p={drop_prob} -> test {record['edge_drop_test_mean']:.2f} ± {record['edge_drop_test_std']:.2f}"
        )
        if log_val_drop:
            msg += f" | val {record['edge_drop_val_mean']:.2f} ± {record['edge_drop_val_std']:.2f}"
        print(msg)

    if rows:
        all_test = np.concatenate([row["edge_drop_test_trials"] for row in rows])
        wandb.log(
            {
                "edge_drop/test_overall_mean": float(all_test.mean()),
                "edge_drop/test_overall_std": float(all_test.std()),
            }
        )
        summary = (
            f"EdgeDrop p={drop_prob} ({params.get('perturb', 'test')}) test {all_test.mean():.2f} ± {all_test.std():.2f}"
        )
        if log_val_drop:
            all_val = np.concatenate([row["edge_drop_val_trials"] for row in rows])
            wandb.log(
                {
                    "edge_drop/val_overall_mean": float(all_val.mean()),
                    "edge_drop/val_overall_std": float(all_val.std()),
                }
            )
            summary += f" | val {all_val.mean():.2f} ± {all_val.std():.2f}"
        print("\n=== EDGE DROP SUMMARY ===")
        print(summary)

        if params.get("save_tsv", False):
            tsv_name = params.get("tsv_name", "")
            if not tsv_name:
                tsv_name = (
                    f"finetune_{params['finetune_dataset']}_edge_drop_p{drop_prob}_"
                    f"{params.get('perturb', 'test')}_seed{params.get('edge_drop_seed', 1)}.tsv"
                )
            base_path = params.get("pretrain_path") or params.get("pt_model_path")
            tsv_path = osp.join(base_path, tsv_name)
            os.makedirs(osp.dirname(tsv_path), exist_ok=True)
            import csv

            with open(tsv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                header = ["split", "train_mean", "train_std", "test_mean", "test_std"]
                if log_val_drop:
                    header.extend(["val_mean", "val_std"])
                writer.writerow(header)
                for idx, row in enumerate(rows):
                    line = [
                        idx,
                        row["edge_drop_train_mean"],
                        row["edge_drop_train_std"],
                        row["edge_drop_test_mean"],
                        row["edge_drop_test_std"],
                    ]
                    if log_val_drop:
                        line.extend([row["edge_drop_val_mean"], row["edge_drop_val_std"]])
                    writer.writerow(line)
            print(f"[saved] {tsv_path}")

    wandb.finish()


def main():
    params = get_args_edge_drop()

    if params["use_params"]:
        dataset = params["finetune_dataset"]
        if dataset not in dataset2task:
            raise ValueError(f"Unsupported dataset: {dataset}")
        task = dataset2task[dataset]
        with open(osp.join(osp.dirname(__file__), "..", "config", "finetune.yaml"), "r") as f:
            default_params = yaml.safe_load(f)
        params = get_args_edge_drop(default_params=default_params[task][dataset])

    ensure_finetune_lr(params)
    params["data_path"] = osp.join(osp.dirname(__file__), "..", "data")
    params["pt_model_path"] = osp.join(osp.dirname(__file__), "..", "ckpts", "pretrain_model")

    dataset = params["finetune_dataset"]
    if dataset not in dataset2task:
        raise ValueError(f"Unsupported dataset: {dataset}")
    task = dataset2task[dataset]
    params["task"] = task

    wandb.init(
        project="STEM-GNN-Finetune",
        name="{} - EdgeDrop".format(str.upper(params["finetune_dataset"])),
        config=params,
        mode="disabled" if params["debug"] else "online",
        tags=[params["setting"], "edge-drop"],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    print("Params loaded.")

    run(params)


if __name__ == "__main__":
    main()
