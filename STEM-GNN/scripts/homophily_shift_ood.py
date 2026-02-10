import math
import os
import os.path as osp
import re
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


BUCKET_NAMES: List[str] = ["ID", "OOD-low", "OOD-high"]
PRIMARY_RATIOS: Tuple[float, float, float] = (0.5, 0.25, 0.25)
SECONDARY_RATIOS: Tuple[float, float, float] = (0.6, 0.2, 0.2)
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


def compute_homophily_buckets(
    data,
) -> Tuple[torch.Tensor, Tuple[float, float], Dict[str, torch.Tensor], int, int]:
    x = data.node_text_feat
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError("Feature homophily expects node features with shape [N, F].")
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

    homophily = torch.zeros_like(denom)
    nonzero = denom > 0
    homophily[nonzero] = numer[nonzero] / denom[nonzero]
    no_feature_neighbors = int((~nonzero).sum().item())

    valid_nodes = (nonzero & valid_feat).nonzero(as_tuple=False).view(-1)
    if valid_nodes.numel() == 0:
        raise ValueError("All nodes have denom==0; feature homophily is undefined.")
    sorted_idx = valid_nodes[torch.argsort(homophily[valid_nodes])]
    n = sorted_idx.numel()
    if n < 3:
        raise ValueError("valid_nodes too small for 3-way split")

    low_count = max(1, int(math.floor(n * 0.15)))
    high_count = max(1, int(math.floor(n * 0.15)))
    if low_count + high_count >= n:
        overflow = low_count + high_count - (n - 1)
        if overflow > 0:
            reducible_high = max(0, high_count - 1)
            reduction = min(overflow, reducible_high)
            high_count -= reduction
            overflow -= reduction
        if overflow > 0:
            reducible_low = max(0, low_count - 1)
            reduction = min(overflow, reducible_low)
            low_count -= reduction
            overflow -= reduction

    id_count = n - low_count - high_count
    ood_low_idx = sorted_idx[:low_count]
    id_idx = sorted_idx[low_count:low_count + id_count]
    ood_high_idx = sorted_idx[low_count + id_count:]

    low_boundary = homophily[ood_low_idx[-1]].item()
    high_boundary = homophily[ood_high_idx[0]].item()

    bucket_indices = {
        "ID": id_idx,
        "OOD-low": ood_low_idx,
        "OOD-high": ood_high_idx,
    }

    return (
        homophily,
        (float(low_boundary), float(high_boundary)),
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


def compute_predictions(model, data, labels, params, train_mask):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        x = data.node_text_feat
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

    homophily, boundaries, bucket_indices, no_feature_neighbors, invalid_feat_count = (
        compute_homophily_buckets(data)
    )
    print(
        f"dataset {dataset_name}: nodes {data.num_nodes} | edges {data.edge_index.size(1)} | "
        f"classes {num_classes} | feats {data.node_text_feat.shape[1]}"
    )
    print(f"Feature homophily boundaries (15/85): {boundaries[0]:.4f}, {boundaries[1]:.4f}")
    print("Note: feature-homophily buckets are computed from node features and graph structure only.")
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
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
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

        base_encoder = load_params(base_encoder, encoder_path)
        base_vq = load_params(base_vq, vq_path)
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

            pred = compute_predictions(task_model, data, labels, params, train_mask)
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
                    "ood/OOD-low": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
                    "ood/OOD-high": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
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
        pred = compute_predictions(task_model, data, labels, params, train_mask)
        id_test_acc = compute_accuracy(pred, labels, id_test_mask, params)
        ood_accs = [compute_accuracy(pred, labels, mask, params) for mask in ood_masks]

        print(f"Run {run + 1:02d} (seed={current_seed}), best epoch {best_epoch}:")
        for name, score in zip(BUCKET_NAMES, [id_test_acc] + ood_accs):
            print(f"  {name}: {100.0 * score:.2f}%")

        _safe_wandb_log(
            {
                "best/epoch": best_epoch,
                "best/train": 100.0 * compute_accuracy(pred, labels, train_mask, params),
                "best/val": 100.0 * best_val,
                "best/test": 100.0 * id_test_acc,
                "best/OOD-low": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
                "best/OOD-high": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
            },
            wandb_state,
        )

        run_metrics.append([id_test_acc] + ood_accs)

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
            "final/OOD-low": f"{mean_scores[1].item():.2f} ± {std_scores[1].item():.2f}",
            "final/OOD-high": f"{mean_scores[2].item():.2f} ± {std_scores[2].item():.2f}",
            "final/ID_mean": mean_scores[0].item(),
            "final/OOD-low_mean": mean_scores[1].item(),
            "final/OOD-high_mean": mean_scores[2].item(),
            "final/ID_std": std_scores[0].item(),
            "final/OOD-low_std": std_scores[1].item(),
            "final/OOD-high_std": std_scores[2].item(),
        },
        wandb_state,
    )


def main():
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
    run_name = f"{str.upper(params['finetune_dataset'])} - Homophily OOD"
    wandb.init(
        project="STEM-GNN-Finetune",
        name=run_name,
        config=params,
        mode="disabled" if params.get("debug", False) else "online",
        tags=[params.get('setting', 'standard'), 'homophily-ood'],
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
