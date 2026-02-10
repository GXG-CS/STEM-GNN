#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import warnings
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb
import yaml

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


# -------------------- noise helpers --------------------
def _num_nodes(data) -> int:
    n = getattr(data, "num_nodes", None)
    if n is not None:
        return int(n)
    if hasattr(data, "node_text_feat"):
        return int(data.node_text_feat.size(0))
    if hasattr(data, "x"):
        return int(data.x.size(0))
    raise ValueError("Cannot infer number of nodes from data.")


def _select_feature_field(data, feature_field: str = "node_text_feat"):
    def _coerce_2d(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        return t.unsqueeze(-1) if t.dim() == 1 else t

    if hasattr(data, feature_field) and isinstance(getattr(data, feature_field), torch.Tensor):
        t = getattr(data, feature_field)
        return _coerce_2d(t), feature_field
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        t = data.x
        return _coerce_2d(t), "x"
    raise AssertionError(f"Neither '{feature_field}' nor 'x' exists as a Tensor on data.")


def _infer_feature_field(params, data, cli_field: str):
    if cli_field:
        return cli_field

    in_dim = int(params.get("input_dim", -1))

    dim_txt = -1
    if hasattr(data, "node_text_feat") and isinstance(getattr(data, "node_text_feat"), torch.Tensor):
        t = getattr(data, "node_text_feat")
        if t.dim() >= 2:
            dim_txt = int(t.size(-1))

    dim_x = -1
    if hasattr(data, "x") and isinstance(getattr(data, "x"), torch.Tensor):
        t = data.x
        if t.dim() >= 2:
            dim_x = int(t.size(-1))

    if in_dim == dim_txt:
        return "node_text_feat"
    if in_dim == dim_x:
        return "x"
    return "node_text_feat" if hasattr(data, "node_text_feat") else "x"


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


def _apply_missing_features(
    data,
    split_masks: Dict[str, torch.Tensor],
    *,
    missing_prob: float = 0.4,
    seed: int = 1,
    perturb: str = "valtest",
    relative_noise_alpha: float = 0.0,
    relative_noise_seed: Optional[int] = None,
    feature_field: str = "node_text_feat",
    debug: bool = False,
):
    x, field_name = _select_feature_field(data, feature_field)
    device, dtype = x.device, x.dtype
    n, feature_dim = x.size(0), x.size(1)

    train_mask = split_masks["train"]
    val_mask = split_masks["valid"]
    test_mask = split_masks["test"]

    if perturb == "test":
        noise_mask = test_mask
    elif perturb == "all":
        noise_mask = torch.ones(n, dtype=torch.bool, device=device)
    else:
        noise_mask = val_mask | test_mask

    missing_prob = float(missing_prob)
    if not (0.0 <= missing_prob <= 1.0):
        raise ValueError(f"Missing probability must be within [0, 1]; got {missing_prob}")

    node_mask = noise_mask.unsqueeze(1)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        drop_rand = torch.rand((n, feature_dim), device=device, generator=generator)
    except TypeError:
        torch.manual_seed(int(seed))
        drop_rand = torch.rand((n, feature_dim), device=device)

    drop_mask = (drop_rand < missing_prob) & node_mask
    x_new = x.clone()
    x_new = x_new.masked_fill(drop_mask, 0.0)

    base_norm = x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Optional relative-L2 noise on retained dimensions
    rel_alpha = float(relative_noise_alpha)
    if rel_alpha > 0.0:
        keep_mask = (~drop_mask) & node_mask
        noise_seed = seed if relative_noise_seed is None else relative_noise_seed
        try:
            generator2 = torch.Generator(device=device)
            generator2.manual_seed(int(noise_seed))
            z = torch.randn((n, feature_dim), device=device, dtype=dtype, generator=generator2)
        except TypeError:
            torch.manual_seed(int(noise_seed))
            z = torch.randn((n, feature_dim), device=device, dtype=dtype)

        z = z * keep_mask.to(dtype=dtype)
        z_norm = z.norm(dim=1, keepdim=True)
        z_norm = z_norm.clamp_min(1e-12)
        dir_z = torch.where(z_norm > 0, z / z_norm, torch.zeros_like(z))

        delta = rel_alpha * base_norm * dir_z
        delta = delta * keep_mask.to(dtype=dtype)
        x_new = x_new + delta

    out = data.clone()
    setattr(out, field_name, x_new)

    if debug:
        with torch.no_grad():
            if noise_mask.any():
                drop_ratio = drop_mask[noise_mask].float().mean().item()
            else:
                drop_ratio = 0.0
            msg = f"[missing-debug] field={field_name} p={missing_prob} perturb={perturb} drop_ratio={drop_ratio:.4f}"
            if rel_alpha > 0.0 and noise_mask.any():
                delta_vec = (x_new - x)[noise_mask]
                rel_delta = (delta_vec.norm(dim=1, keepdim=True) / base_norm[noise_mask]).mean().item()
                msg += f" relative_noise={rel_alpha:.4f} mean_rel_delta={rel_delta:.4f}"
            print(msg)
    return out


# -------------------- parser helpers --------------------
def build_parser():
    import argparse

    parser = argparse.ArgumentParser("Finetune (node only) with optional Gaussian noise evaluation")

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

    # Missing-feature Parameters
    parser.add_argument(
        "--missing_prob",
        type=float,
        default=0.4,
        help="Probability of masking each feature dimension on selected nodes.",
    )
    parser.add_argument("--missing_seed", type=int, default=1, help="Random seed for feature masking.")
    parser.add_argument("--perturb", choices=["test", "valtest", "all"], default="test")
    parser.add_argument("--feature_field", default="")
    parser.add_argument(
        "--relative_noise_alpha",
        type=float,
        default=0.0,
        help="Optional relative-L2 noise strength applied after masking (0 disables).",
    )
    parser.add_argument(
        "--relative_noise_seed",
        type=int,
        default=None,
        help="Seed for relative-L2 noise (defaults to masking seed when omitted).",
    )
    parser.add_argument("--include_val_missing", action="store_true")
    parser.add_argument("--debug_missing", action="store_true")
    parser.add_argument("--save_tsv", action="store_true")
    parser.add_argument("--tsv_name", type=str, default="")

    return parser


_MOE_PARAM_KEYS = ("moe", "moe_layers", "moe_experts", "moe_tau", "lamda_env")


def get_args_with_noise(default_params=None):
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

    idx = name.find("moe_")
    if idx >= 0:
        suffix = name[idx + 4 :]
        parts = suffix.split("_")
        if parts and parts[0].isdigit():
            maybe_set("moe", parts[0] != "0")

    if not target.get("moe", False):
        return

    import re

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


# -------------------- main routine --------------------
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
        raise NotImplementedError("missing_feature.py supports node classification only.")

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

    if task != "node":
        raise NotImplementedError("missing_feature.py currently supports node classification only.")
    if setting != "standard":
        raise NotImplementedError("finetune_with_noise.py currently supports the standard setting only.")

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
        raise ValueError("Missing-feature finetuning only supports the standard setting.")

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

    # ---------------- missing-feature evaluation (node + standard only) ----------------
    if task != "node":
        print("[info] Noise evaluation skipped (only supported for node-standard setting).")
        wandb.finish()
        return

    feature_field = _infer_feature_field(params, data, params.get("feature_field", ""))
    rows: List[dict] = []
    log_val_missing = params.get("include_val_missing", False) or params.get("perturb") != "test"

    for split_idx, (split, state_dict) in enumerate(zip(eval_splits, best_state_dicts)):
        eval_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        eval_model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})

        feat_tensor, _ = _select_feature_field(data, feature_field)
        mask_device = feat_tensor.device
        train_mask, val_mask, test_mask = _extract_masks_train_valid_test(split, data, mask_device)
        mask_dict = {
            "train": train_mask,
            "valid": val_mask,
            "val": val_mask,
            "test": test_mask,
        }

        noisy_data = _apply_missing_features(
            data,
            mask_dict,
            missing_prob=params["missing_prob"],
            seed=params["missing_seed"],
            perturb=params["perturb"],
            relative_noise_alpha=params.get("relative_noise_alpha", 0.0),
            relative_noise_seed=params.get("relative_noise_seed", None),
            feature_field=feature_field,
            debug=params.get("debug_missing", False) and split_idx == 0,
        )

        noisy_metrics = eval_node(
            model=eval_model,
            dataset=noisy_data,
            loader=None,
            split=mask_dict,
            labels=labels,
            params=params,
            num_neighbors=[-1] * params["num_layers"],
        )

        record = {"missing_test": float(noisy_metrics["test"])}
        if log_val_missing:
            record["missing_val"] = float(noisy_metrics["val"])
        rows.append(record)

        log_payload = {f"missing/test_split{split_idx}": noisy_metrics["test"]}
        if log_val_missing:
            log_payload[f"missing/val_split{split_idx}"] = noisy_metrics["val"]
        wandb.log(log_payload)

        msg = (
            f"[split {split_idx}] Missing p={params['missing_prob']} "
            f"(alpha={params.get('relative_noise_alpha', 0.0)}) => test {noisy_metrics['test']:.2f}"
        )
        if log_val_missing:
            msg += f" | val {noisy_metrics['val']:.2f}"
        print(msg)

    if rows:
        missing_test = np.array([row["missing_test"] for row in rows], dtype=float)
        print("\n=== MISSING FEATURE SUMMARY ===")
        summary = (
            f"Missing p={params['missing_prob']} alpha={params.get('relative_noise_alpha', 0.0)} "
            f"({params['perturb']}) test {missing_test.mean():.2f} ± {missing_test.std():.2f}"
        )
        wandb.log(
            {
                "missing/missing_test_mean": missing_test.mean(),
                "missing/missing_test_std": missing_test.std(),
            }
        )
        if log_val_missing:
            missing_val = np.array([row["missing_val"] for row in rows], dtype=float)
            summary += f" | val {missing_val.mean():.2f} ± {missing_val.std():.2f}"
            wandb.log(
                {
                    "missing/missing_val_mean": missing_val.mean(),
                    "missing/missing_val_std": missing_val.std(),
                }
            )
        print(summary)

        if params.get("save_tsv", False):
            tsv_name = params["tsv_name"]
            if not tsv_name:
                tsv_name = (
                    f"finetune_{params['finetune_dataset']}_missing_p{params['missing_prob']}_"
                    f"alpha{params.get('relative_noise_alpha', 0.0)}_{params['perturb']}_"
                    f"seed{params['missing_seed']}.tsv"
                )
            tsv_path = osp.join(params["pretrain_path"] or params["pt_model_path"], tsv_name)
            os.makedirs(osp.dirname(tsv_path), exist_ok=True)
            import csv

            with open(tsv_path, "w", newline="") as f:
                header = ["split", "missing_test"]
                if log_val_missing:
                    header.insert(1, "missing_val")
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(header)
                for idx, row in enumerate(rows):
                    data_row = [idx]
                    if log_val_missing:
                        data_row.append(row["missing_val"])
                    data_row.append(row["missing_test"])
                    writer.writerow(data_row)
            print(f"[saved] {tsv_path}")

    wandb.finish()


def main():
    params = get_args_with_noise()

    if params["use_params"]:
        dataset = params["finetune_dataset"]
        if dataset not in dataset2task:
            raise ValueError(f"Unsupported dataset: {dataset}")
        task = dataset2task[dataset]
        with open(osp.join(osp.dirname(__file__), "..", "config", "finetune.yaml"), "r") as f:
            default_params = yaml.safe_load(f)
        params = get_args_with_noise(default_params=default_params[task][dataset])

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
        name="{} - Noise Eval".format(str.upper(params["finetune_dataset"])),
        config=params,
        mode="disabled" if params["debug"] else "online",
        tags=[params["setting"], "noise-eval"],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    print("Params loaded.")

    run(params)


if __name__ == "__main__":
    main()
