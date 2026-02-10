import argparse
import math
import os
import os.path as osp
import re
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.utils import remove_self_loops, to_undirected

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.ft_model import TaskModel
from model.vq import VectorQuantize
from task.node import ft_node
from utils.args import get_args_finetune
from utils.eval import evaluate
from utils.others import ensure_finetune_lr, freeze_params, load_params, seed_everything, get_pretrain_run_id
from utils.preprocess import pre_node

import warnings
import wandb


BUCKET_NAMES: List[str] = ["ID", "OOD1", "OOD2", "OOD3"]
PRIMARY_RATIOS: Tuple[float, float, float] = (0.5, 0.25, 0.25)
SECONDARY_RATIOS: Tuple[float, float, float] = (0.6, 0.2, 0.2)
DEFAULT_MISSING_PROBS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
DATASET2TASK = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
}


def _label_valid_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.is_floating_point():
        return torch.isfinite(labels) & (labels >= 0)
    return labels >= 0


def _coerce_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            return 1
        if lowered in {"false", "no", "n"}:
            return 0
        try:
            return int(float(lowered))
        except ValueError:
            return default
    return default


def _safe_wandb_log(payload: Dict, state: Dict[str, bool]) -> None:
    if not state.get("active", True):
        return
    try:
        wandb.log(payload)
    except Exception as exc:
        state["active"] = False
        print(f"[wandb] logging failed: {exc}")


def _nanmean_std(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(values)
    counts = mask.sum(dim=0)
    if hasattr(torch, "nanmean"):
        mean = torch.nanmean(values, dim=0)
    else:
        summed = torch.where(mask, values, torch.zeros_like(values)).sum(dim=0)
        mean = summed / counts.clamp(min=1)

    if hasattr(torch, "nanstd"):
        std = torch.nanstd(values, dim=0, unbiased=False)
    else:
        mean_broadcast = mean.unsqueeze(0)
        diff = torch.where(mask, values - mean_broadcast, torch.zeros_like(values))
        var = (diff ** 2).sum(dim=0) / counts.clamp(min=1)
        std = torch.sqrt(var)

    mean = torch.where(counts > 0, mean, torch.full_like(mean, float("nan")))
    std = torch.where(counts > 0, std, torch.full_like(std, float("nan")))
    return mean, std


def _nanmin(values: List[float]) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float32)
    finite = torch.isfinite(tensor)
    if not finite.any():
        return float("nan")
    return float(tensor[finite].min().item())


def _load_params_cpu(model: nn.Module, path: str) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _parse_missing_probs(value, default: Tuple[float, ...]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        if parts:
            return [float(v) for v in parts]
    return list(default)


def _extract_tri_args(argv: List[str]) -> Tuple[Dict[str, object], List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--missing_probs", type=str, default=None)
    parser.add_argument("--missing_seed", type=int, default=None)
    parser.add_argument("--perturb_seed_mode", type=str, default=None)
    args, rest = parser.parse_known_args(argv)
    return vars(args), rest


# -------------------- missing-feature helpers --------------------
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


def _apply_missing_features(
    data,
    split_masks: Dict[str, torch.Tensor],
    *,
    missing_prob: float = 0.4,
    seed: int = 1,
    perturb: str = "test",
    relative_noise_alpha: float = 0.0,
    relative_noise_seed: int = None,
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


def compute_alignment_buckets(
    data,
    feature_field: str,
) -> Tuple[torch.Tensor, Tuple[float, float, float, float, float], Dict[str, torch.Tensor], int, int]:
    x, _ = _select_feature_field(data, feature_field)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError("Feature-structure alignment expects node features with shape [N, F].")
    if not x.is_floating_point():
        x = x.to(torch.float32)

    finite_mask = torch.isfinite(x).all(dim=1)
    norm = x.norm(p=2, dim=1)
    valid_feat = finite_mask & (norm > 0)
    invalid_feat_count = int((~valid_feat).sum().item())
    x_norm = x / norm.clamp(min=1e-12).unsqueeze(1)

    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index

    valid_edge = valid_feat[row] & valid_feat[col]
    row_valid = row[valid_edge]
    col_valid = col[valid_edge]

    sim = (x_norm[row_valid] * x_norm[col_valid]).sum(dim=1)

    denom = torch.zeros(data.num_nodes, device=edge_index.device, dtype=torch.float32)
    numer = torch.zeros_like(denom)
    denom.index_add_(0, row_valid, torch.ones_like(sim))
    numer.index_add_(0, row_valid, sim)

    alignment = torch.zeros_like(denom)
    nonzero = denom > 0
    alignment[nonzero] = numer[nonzero] / denom[nonzero]
    no_feature_neighbors = int((~nonzero).sum().item())

    valid_nodes = (nonzero & valid_feat).nonzero(as_tuple=False).view(-1)
    if valid_nodes.numel() == 0:
        raise ValueError("All nodes have denom==0; feature-structure alignment is undefined.")
    sorted_idx = valid_nodes[torch.argsort(alignment[valid_nodes])]
    n = sorted_idx.numel()
    if n < 10:
        raise ValueError("valid_nodes too small for 4-way split")

    c3 = max(1, int(math.floor(n * 0.10)))
    c2 = max(c3 + 1, int(math.floor(n * 0.20)))
    c1 = max(c2 + 1, int(math.floor(n * 0.30)))
    id_lo = max(c1, int(math.floor(n * 0.30)))
    id_hi = max(id_lo + 1, int(math.floor(n * 0.80)))
    if id_hi > n:
        id_hi = n
    if id_lo >= id_hi:
        raise ValueError("valid_nodes too small for ID split (30%-80%).")

    ood3_idx = sorted_idx[:c3]
    ood2_idx = sorted_idx[c3:c2]
    ood1_idx = sorted_idx[c2:c1]
    id_idx = sorted_idx[id_lo:id_hi]

    b3 = alignment[ood3_idx[-1]].item()
    b2 = alignment[ood2_idx[-1]].item()
    b1 = alignment[ood1_idx[-1]].item()
    id_low = alignment[id_idx[0]].item()
    id_high = alignment[id_idx[-1]].item()

    bucket_indices = {
        "ID": id_idx,
        "OOD1": ood1_idx,
        "OOD2": ood2_idx,
        "OOD3": ood3_idx,
    }

    return (
        alignment,
        (float(b3), float(b2), float(b1), float(id_low), float(id_high)),
        bucket_indices,
        no_feature_neighbors,
        invalid_feat_count,
    )


def determine_split_counts(class_size: int) -> Tuple[int, int]:
    if class_size < 3:
        raise RuntimeError(f"class size {class_size} too small for 3-way split")

    for ratios in (PRIMARY_RATIOS, SECONDARY_RATIOS):
        train = max(1, math.floor(class_size * ratios[0]))
        val = max(1, math.floor(class_size * ratios[1]))
        if train + val >= class_size:
            overflow = train + val - (class_size - 1)
            if overflow > 0:
                reducible_val = max(0, val - 1)
                reduction = min(overflow, reducible_val)
                val -= reduction
                overflow -= reduction
            if overflow > 0:
                reducible_train = max(0, train - 1)
                reduction = min(overflow, reducible_train)
                train -= reduction
                overflow -= reduction
        test = class_size - train - val
        if train >= 1 and val >= 1 and test >= 1:
            return train, val

    train = max(1, class_size - 2)
    val = 1
    return train, val


def stratified_split(id_indices: torch.Tensor, labels: torch.Tensor, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_parts, val_parts, test_parts = [], [], []
    id_labels = labels[id_indices]
    classes = torch.unique(id_labels)

    for cls in classes.tolist():
        cls_idx = id_indices[id_labels == cls]
        cls_size = cls_idx.numel()
        train_count, val_count = determine_split_counts(cls_size)

        perm = torch.randperm(cls_size, generator=generator)
        cls_idx = cls_idx[perm]

        train_parts.append(cls_idx[:train_count])
        val_parts.append(cls_idx[train_count:train_count + val_count])
        test_parts.append(cls_idx[train_count + val_count:])

    train_idx = torch.sort(torch.cat(train_parts))[0]
    val_idx = torch.sort(torch.cat(val_parts))[0]
    test_idx = torch.sort(torch.cat(test_parts))[0]
    return train_idx, val_idx, test_idx


def indices_to_mask(indices: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[indices.to(device=device)] = True
    return mask


def compute_predictions(model, data, labels, params, feature_field: str):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        x, _ = _select_feature_field(data, feature_field)
        edge_index = data.edge_index
        edge_attr = data.edge_text_feat[data.xe]
        z = model.encode(x, edge_index, edge_attr)
        logits = model.get_lin_logits(z)
        if logits.ndim == 3:
            logits = logits.mean(1)
        elif logits.ndim != 2:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
        pred = logits.softmax(dim=-1)

    if was_training:
        model.train()
    return pred


def compute_accuracy(pred, labels, mask, params):
    if mask.sum().item() == 0:
        return float("nan")
    value = evaluate(pred, labels, mask, params)
    return value / 100.0


def run(params):
    data_dir = params['data_path']
    dataset_name = params['finetune_dataset']

    if dataset_name not in DATASET2TASK:
        raise ValueError(f"Unsupported dataset: {dataset_name} (node datasets only).")

    params['task'] = 'node'

    dataset, _, labels, num_classes, _ = get_finetune_graph(data_dir, dataset_name)
    params["num_classes"] = num_classes

    dataset = pre_node(dataset)
    data = dataset[0]
    labels = labels.long()
    data.y = labels
    missing = [name for name in ("node_text_feat", "edge_text_feat", "xe") if not hasattr(data, name)]
    if missing:
        raise AttributeError(
            f"Dataset missing required fields for node task: {', '.join(missing)}"
        )
    if data.xe is None:
        raise AttributeError("Dataset field xe is None; cannot index edge_text_feat.")
    if data.xe.dtype not in (torch.int64, torch.long):
        raise TypeError(f"xe must be int64 for indexing (got {data.xe.dtype}).")
    if data.xe.numel() != data.edge_index.size(1):
        raise ValueError(
            f"xe length {data.xe.numel()} does not match num_edges {data.edge_index.size(1)}"
        )
    if data.xe.numel() > 0:
        max_xe = int(data.xe.max().item())
        if data.edge_text_feat.size(0) <= max_xe:
            raise ValueError(
                f"edge_text_feat has size {data.edge_text_feat.size(0)} but xe max is {max_xe}"
            )

    feature_field = _infer_feature_field(params, data, params.get("feature_field", ""))
    alignment, boundaries, bucket_indices, no_feature_neighbors, invalid_feat_count = (
        compute_alignment_buckets(data, feature_field)
    )
    feat_tensor, _ = _select_feature_field(data, feature_field)
    print(
        f"dataset {dataset_name}: nodes {data.num_nodes} | edges {data.edge_index.size(1)} | "
        f"classes {num_classes} | feats {feat_tensor.shape[1]}"
    )
    print("Feature-structure alignment boundaries:")
    print(
        f"  OOD3<= {boundaries[0]:.4f}, OOD2<= {boundaries[1]:.4f}, "
        f"OOD1<= {boundaries[2]:.4f}, ID range [{boundaries[3]:.4f}, {boundaries[4]:.4f}]"
    )
    print("Note: feature-structure alignment is computed from node features and graph structure only.")
    print(f"Nodes with no valid-feature neighbors (excluded): {no_feature_neighbors}")
    print(f"Nodes with invalid features (excluded): {invalid_feat_count}")
    for name in BUCKET_NAMES:
        print(f"Bucket {name}: {bucket_indices[name].numel()} nodes")

    activation_str = params["activation"].lower() if isinstance(params["activation"], str) else "relu"
    activation_map = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "leaky": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
    }
    if activation_str not in activation_map:
        raise ValueError(f"Unsupported activation: {activation_str}")
    activation = activation_map[activation_str]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available; this script requires GPU. "
            "Check your conda activation and CUDA setup."
        )
    if params["gpu"] >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested gpu index {params['gpu']} but only {torch.cuda.device_count()} device(s) available."
        )
    device = torch.device(f"cuda:{params['gpu']}")
    params["activation"] = activation
    print(f"Using device: {device}")

    base_encoder = Encoder(
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

    base_vq = VectorQuantize(
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
        pretrain_path = osp.join(params['pt_model_path'], pretrain_path)

    if pretrain_path or params.get("pretrain_dataset", "na") != 'na':
        pretrain_task = params.get('pretrain_task', 'all')
        if pretrain_path:
            path = pretrain_path
        elif pretrain_task == 'all':
            path = osp.join(params['pt_model_path'], pretrain_run_id)
        else:
            raise ValueError("Invalid pretrain task configuration.")

        encoder_path = osp.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt')
        vq_path = osp.join(path, f'vq_{params["pretrain_model_epoch"]}.pt')
        if not osp.exists(encoder_path):
            raise FileNotFoundError("Cannot find encoder checkpoint. Set --pretrain_path to a valid folder.")
        if not osp.exists(vq_path):
            raise FileNotFoundError("Cannot find vector-quantizer checkpoint. Set --pretrain_path to a valid folder.")

        base_encoder = _load_params_cpu(base_encoder, encoder_path)
        base_vq = _load_params_cpu(base_vq, vq_path)
        print("Loaded pretrained encoder and VQ.")

    if params.get("freeze_vq", 1):
        freeze_params(base_vq)
        print("Freeze VQ parameters during fine-tuning")

    if params["batch_size"] != 0:
        raise ValueError("This script only supports full-batch training (batch_size must be 0).")

    data = data.to(device)
    labels = labels.to(device)

    num_runs = params.get("repeat", 1)
    display_step = params.get("display_step", 10)
    base_seed = 0

    run_metrics = []
    run_eval_metrics = []
    run_perturb_means = []
    missing_seed = int(params.get("missing_seed", 1))
    perturb_seed_mode = str(params.get("perturb_seed_mode", "per_run")).strip().lower()
    if perturb_seed_mode not in {"per_run", "fixed"}:
        raise ValueError(f"perturb_seed_mode must be 'per_run' or 'fixed' (got {perturb_seed_mode})")
    missing_probs = _parse_missing_probs(params.get("missing_probs", None), DEFAULT_MISSING_PROBS)
    missing_probs = tuple(round(float(p), 2) for p in missing_probs)
    if not missing_probs:
        raise ValueError("missing_probs must contain at least one value.")
    if any(p < 0.0 or p > 1.0 for p in missing_probs):
        raise ValueError(f"missing_probs must be within [0, 1] (got {missing_probs})")
    print(f"Missing sweep probs: {missing_probs}")
    print(f"Perturb seed mode: {perturb_seed_mode}")

    wandb_state = {"active": True}

    for run in range(num_runs):
        current_seed = base_seed + run
        seed_everything(current_seed)

        run_bucket_indices = {name: idx.clone() for name, idx in bucket_indices.items()}

        task_model = TaskModel(
            encoder=deepcopy(base_encoder),
            vq=deepcopy(base_vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        print(f"[Run {run + 1:02d}] MoE enabled: {getattr(task_model.encoder, 'moe', False)} | "
              f"MoE layers: {getattr(task_model.encoder, 'moe_layer_flags', [])}")

        optimizer = torch.optim.AdamW(task_model.parameters(), lr=params["finetune_lr"])

        id_indices = run_bucket_indices["ID"]
        labels_cpu = labels.cpu()
        labeled_mask = labels_cpu >= 0
        unlabeled_in_id = (~labeled_mask[id_indices]).sum().item()
        if unlabeled_in_id:
            print(f"[Run {run + 1:02d}] Excluding {unlabeled_in_id} unlabeled ID nodes.")
        id_indices = id_indices[labeled_mask[id_indices]]
        id_labels = labels_cpu[id_indices]
        classes, counts = torch.unique(id_labels, return_counts=True)
        small_classes = classes[counts < 3]
        if small_classes.numel() > 0:
            small_mask = torch.isin(id_labels, small_classes)
            dropped = id_indices[small_mask]
            id_indices = id_indices[~small_mask]
            print(
                f"[Run {run + 1:02d}] Dropping {dropped.numel()} ID nodes from "
                f"{small_classes.numel()} class(es) with <3 samples."
            )
        if id_indices.numel() < 3:
            raise ValueError("ID bucket too small after filtering (<3 nodes).")
        try:
            train_idx, val_idx, id_test_idx = stratified_split(id_indices, labels.cpu(), current_seed)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to stratify ID bucket for dataset {dataset_name}. "
                f"Consider adjusting bucket ratios or ensuring each class has at least 3 nodes within the ID bucket."
            ) from exc

        train_mask = indices_to_mask(train_idx, data.num_nodes, device)
        val_mask = indices_to_mask(val_idx, data.num_nodes, device)
        id_test_mask = indices_to_mask(id_test_idx, data.num_nodes, device)

        ood_masks = []
        labeled_mask_device = labels >= 0
        for name in BUCKET_NAMES[1:]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            mask[run_bucket_indices[name].to(device=device)] = True
            mask &= labeled_mask_device
            ood_masks.append(mask)

        print(
            f"[Run {run + 1:02d}] ID splits -> Train: {train_idx.numel()}, "
            f"Val: {val_idx.numel()}, ID-Test: {id_test_idx.numel()}"
        )
        for name, mask in zip(BUCKET_NAMES[1:], ood_masks):
            count = mask.sum().item()
            print(f"          Bucket {name}: {count} nodes")
            if count == 0:
                print(f"          [Warning] Bucket {name} is empty for this dataset/run.")

        split = {"train": train_mask, "valid": val_mask, "test": id_test_mask}

        best_val = float("-inf")
        best_state = None
        best_epoch = -1
        patience_counter = 0

        for epoch in range(params["finetune_epochs"]):
            loss_dict = ft_node(
                model=task_model,
                dataset=data,
                loader=None,
                optimizer=optimizer,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[30] * params["num_layers"],
            )

            pred = compute_predictions(task_model, data, labels, params, feature_field)
            train_acc = compute_accuracy(pred, labels, train_mask, params)
            val_acc = compute_accuracy(pred, labels, val_mask, params)
            id_test_acc = compute_accuracy(pred, labels, id_test_mask, params)
            ood_accs = [compute_accuracy(pred, labels, mask, params) for mask in ood_masks]

            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % display_step == 0:
                bucket_msg = ", ".join(
                    f"{name}: {100.0 * score:.2f}%"
                    for name, score in zip(BUCKET_NAMES, [id_test_acc] + ood_accs)
                )
                total_loss = loss_dict.get('loss', float('nan'))
                print(
                    f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}, "
                    f"Train(ID): {100.0 * train_acc:.2f}%, "
                    f"Valid(ID): {100.0 * val_acc:.2f}%, "
                    f"{bucket_msg}"
                )

            _safe_wandb_log(
                {
                    "epoch": epoch,
                    "train/lin_loss": loss_dict.get("act_loss", float("nan")),
                    "train/jac_loss": loss_dict.get("jac_loss", float("nan")),
                    "train/env_loss": loss_dict.get("env_loss", float("nan")),
                    "train/loss": loss_dict.get("loss", float("nan")),
                    "train/train_value": 100.0 * train_acc,
                    "train/val_value": 100.0 * val_acc,
                    "train/test_value": 100.0 * id_test_acc,
                    "ood/OOD1": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
                    "ood/OOD2": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
                    "ood/OOD3": 100.0 * (ood_accs[2] if len(ood_accs) > 2 else float("nan")),
                },
                wandb_state,
            )

            if params["early_stop"] > 0 and patience_counter >= params["early_stop"]:
                print(
                    f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, "
                    f"best valid(ID): {100.0 * best_val:.2f}%)."
                )
                break

        if best_state is None:
            raise RuntimeError("Training finished without recording validation improvements.")

        task_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        pred = compute_predictions(task_model, data, labels, params, feature_field)
        id_test_acc = compute_accuracy(pred, labels, id_test_mask, params)
        ood_accs = [compute_accuracy(pred, labels, mask, params) for mask in ood_masks]
        ood_min_acc = _nanmin(ood_accs)

        mask_dict = {"train": train_mask, "valid": val_mask, "test": id_test_mask}
        perturb_accs = []
        for idx, prob in enumerate(missing_probs):
            prob_seed = missing_seed + idx
            if perturb_seed_mode == "per_run":
                prob_seed += 1000 * run
            noisy_data = _apply_missing_features(
                data,
                mask_dict,
                missing_prob=prob,
                seed=prob_seed,
                perturb="test",
                feature_field=feature_field,
                relative_noise_alpha=float(params.get("relative_noise_alpha", 0.0)),
                relative_noise_seed=params.get("relative_noise_seed", None),
                debug=bool(params.get("debug_missing", False)) and run == 0 and idx == 0,
            )
            noisy_pred = compute_predictions(task_model, noisy_data, labels, params, feature_field)
            perturb_accs.append(compute_accuracy(noisy_pred, labels, id_test_mask, params))
        assert len(perturb_accs) == len(missing_probs), "perturb_accs length mismatch"

        nonzero_indices = [i for i, prob in enumerate(missing_probs) if prob > 0.0]
        if nonzero_indices:
            perturb_mean = float(torch.tensor([perturb_accs[i] for i in nonzero_indices]).mean().item())
        else:
            perturb_mean = float(torch.tensor(perturb_accs).mean().item())

        print(f"Run {run + 1:02d} (seed={current_seed}), best epoch {best_epoch}:")
        for name, score in zip(BUCKET_NAMES, [id_test_acc] + ood_accs):
            print(f"  {name}: {100.0 * score:.2f}%")
        print(f"  Fit: {100.0 * id_test_acc:.2f}%")
        print(f"  OOD (min clean): {100.0 * ood_min_acc:.2f}%")
        for prob, acc in zip(missing_probs, perturb_accs):
            print(f"  Perturb p={prob:.2f}: {100.0 * acc:.2f}%")
        print(f"  Perturb mean: {100.0 * perturb_mean:.2f}%")

        best_payload = {
            "best/epoch": best_epoch,
            "best/train": 100.0 * compute_accuracy(pred, labels, train_mask, params),
            "best/val": 100.0 * best_val,
            "best/test": 100.0 * id_test_acc,
            "best/OOD1": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
            "best/OOD2": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
            "best/OOD3": 100.0 * (ood_accs[2] if len(ood_accs) > 2 else float("nan")),
            "best/Fit": 100.0 * id_test_acc,
            "best/OOD": 100.0 * ood_min_acc,
            "best/Perturb_mean": 100.0 * perturb_mean,
        }
        for prob, acc in zip(missing_probs, perturb_accs):
            key = f"best/Perturb_p{prob:.2f}"
            best_payload[key] = 100.0 * acc
        _safe_wandb_log(best_payload, wandb_state)

        run_metrics.append([id_test_acc] + ood_accs)
        run_eval_metrics.append([id_test_acc, ood_min_acc] + perturb_accs)
        run_perturb_means.append(perturb_mean)

    metrics_tensor = torch.tensor(run_metrics, dtype=torch.float32)
    mean_scores, std_scores = _nanmean_std(metrics_tensor)
    mean_scores = mean_scores * 100.0
    std_scores = std_scores * 100.0

    print(f"\nSummary over {num_runs} run(s) on {dataset_name}:")
    for name, mean, std in zip(BUCKET_NAMES, mean_scores, std_scores):
        print(f"  {name}: {mean:.2f}% ± {std:.2f}%")

    _safe_wandb_log(
        {
            "final/ID": f"{mean_scores[0].item():.2f} ± {std_scores[0].item():.2f}",
            "final/OOD1": f"{mean_scores[1].item():.2f} ± {std_scores[1].item():.2f}",
            "final/OOD2": f"{mean_scores[2].item():.2f} ± {std_scores[2].item():.2f}",
            "final/OOD3": f"{mean_scores[3].item():.2f} ± {std_scores[3].item():.2f}",
            "final/ID_mean": mean_scores[0].item(),
            "final/OOD1_mean": mean_scores[1].item(),
            "final/OOD2_mean": mean_scores[2].item(),
            "final/OOD3_mean": mean_scores[3].item(),
            "final/ID_std": std_scores[0].item(),
            "final/OOD1_std": std_scores[1].item(),
            "final/OOD2_std": std_scores[2].item(),
            "final/OOD3_std": std_scores[3].item(),
        },
        wandb_state,
    )

    eval_tensor = torch.tensor(run_eval_metrics, dtype=torch.float32)
    eval_mean, eval_std = _nanmean_std(eval_tensor)
    eval_mean = eval_mean * 100.0
    eval_std = eval_std * 100.0

    print("\nFit/OOD/Perturb summary:")
    print(f"  Fit: {eval_mean[0]:.2f}% ± {eval_std[0]:.2f}%")
    print(f"  OOD: {eval_mean[1]:.2f}% ± {eval_std[1]:.2f}%")
    for idx, prob in enumerate(missing_probs):
        pos = 2 + idx
        print(f"  Perturb p={prob:.2f}: {eval_mean[pos]:.2f}% ± {eval_std[pos]:.2f}%")
    if run_perturb_means:
        perturb_mean_tensor = torch.tensor(run_perturb_means, dtype=torch.float32)
        mean_pm, std_pm = _nanmean_std(perturb_mean_tensor)
        mean_pm = float(mean_pm.item())
        std_pm = float(std_pm.item())
        print(f"  Perturb mean: {100.0 * mean_pm:.2f}% ± {100.0 * std_pm:.2f}%")

    final_payload = {
        "final/Fit": f"{eval_mean[0].item():.2f} ± {eval_std[0].item():.2f}",
        "final/OOD": f"{eval_mean[1].item():.2f} ± {eval_std[1].item():.2f}",
        "final/Fit_mean": eval_mean[0].item(),
        "final/OOD_mean": eval_mean[1].item(),
        "final/Fit_std": eval_std[0].item(),
        "final/OOD_std": eval_std[1].item(),
    }
    for idx, prob in enumerate(missing_probs):
        pos = 2 + idx
        key = f"final/Perturb_p{prob:.2f}"
        final_payload[key] = f"{eval_mean[pos].item():.2f} ± {eval_std[pos].item():.2f}"
        final_payload[f"final/Perturb_p{prob:.2f}_mean"] = eval_mean[pos].item()
        final_payload[f"final/Perturb_p{prob:.2f}_std"] = eval_std[pos].item()
    if run_perturb_means:
        final_payload["final/Perturb_mean"] = f"{100.0 * mean_pm:.2f} ± {100.0 * std_pm:.2f}"
        final_payload["final/Perturb_mean_mean"] = mean_pm
        final_payload["final/Perturb_mean_std"] = std_pm
    _safe_wandb_log(final_payload, wandb_state)


def main():
    tri_args, remaining = _extract_tri_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    params = get_args_finetune()
    base_dir = osp.dirname(__file__)

    if params["use_params"]:
        config_path = osp.join(base_dir, '..', 'config', 'finetune.yaml')
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        import yaml
        with open(config_path, 'r') as f:
            default_params = yaml.safe_load(f)
        dataset = params["finetune_dataset"]
        if dataset not in DATASET2TASK:
            raise ValueError(f"Unsupported dataset: {dataset} (node datasets only).")
        params = get_args_finetune(default_params=default_params["node"][dataset])

    for key, value in tri_args.items():
        if value is not None:
            params[key] = value

    ensure_finetune_lr(params)
    params['data_path'] = osp.join(base_dir, '..', 'data')
    params['pt_model_path'] = osp.join(base_dir, '..', 'ckpts', 'pretrain_model')

    params.setdefault("display_step", 10)

    pretrain_run_id = get_pretrain_run_id(params)
    default_pretrain_path = osp.join(base_dir, "..", "ckpts", "pretrain_model", pretrain_run_id)
    DEFAULT_PRETRAIN_SEED = 42
    DEFAULT_PRETRAIN_EPOCH = 25

    explicit_pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if not explicit_pretrain_path or explicit_pretrain_path.lower() in {"default", "auto"}:
        params["pretrain_path"] = default_pretrain_path
    else:
        params["pretrain_path"] = explicit_pretrain_path

    if "pretrain_model_epoch" not in params or params["pretrain_model_epoch"] in (None, ""):
        params["pretrain_model_epoch"] = DEFAULT_PRETRAIN_EPOCH
    if "pretrain_seed" not in params or params["pretrain_seed"] in (None, ""):
        params["pretrain_seed"] = DEFAULT_PRETRAIN_SEED

    def _maybe_infer_moe_settings(path: str, target: Dict):
        if not path:
            return
        name = osp.basename(path.rstrip("/"))
        inferred = False
        moe_flag = re.search(r"moe_(\d+)", name)
        if moe_flag:
            target["moe"] = moe_flag.group(1) != "0"
            inferred = True
        if target.get("moe", False):
            layers_match = re.search(r"layers_([A-Za-z]+)", name)
            if layers_match:
                target["moe_layers"] = layers_match.group(1).lower()
                inferred = True
            experts_match = re.search(r"_K_(\d+)", name)
            if experts_match:
                target["moe_experts"] = int(experts_match.group(1))
                inferred = True
            tau_match = re.search(r"_tau_([0-9.]+)", name)
            if tau_match:
                tau_val = float(tau_match.group(1))
                target["moe_tau"] = tau_val
                target["tau"] = tau_val
                inferred = True
            lam_match = re.search(r"_lam_([0-9.]+)", name)
            if lam_match:
                target["lamda_env"] = float(lam_match.group(1))
                inferred = True
        if not inferred:
            print(f"[Pretrain] Could not infer MoE settings from '{name}', using existing params.")

    _maybe_infer_moe_settings(params.get("pretrain_path", ""), params)

    warnings.filterwarnings("ignore")
    run_name = f"{str.upper(params['finetune_dataset'])} - Tri Objective"
    wandb.init(
        project="STEM-GNN-Finetune",
        name=run_name,
        config=params,
        mode="disabled" if params.get("debug", False) else "online",
        tags=[params.get('setting', 'standard'), 'tri-objective'],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    params["freeze_vq"] = _coerce_int(params.get("freeze_vq", 1), 1)
    params["use_vq"] = _coerce_int(params.get("use_vq", 1), 1)
    params["early_stop"] = _coerce_int(params.get("early_stop", 0), 0)
    params["batch_size"] = _coerce_int(params.get("batch_size", 0), 0)
    params["repeat"] = _coerce_int(params.get("repeat", 1), 1)
    params["finetune_epochs"] = _coerce_int(params.get("finetune_epochs", 1), 1)
    if params["finetune_epochs"] < 1:
        raise ValueError(f"finetune_epochs must be >= 1 (got {params['finetune_epochs']})")
    print("Params loaded.")

    try:
        run(params)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
