import datetime
from lib2to3.pytree import BasePattern
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path

import torch
from torch_geometric.utils import (degree, remove_self_loops, add_self_loops, to_undirected, k_hop_subgraph, coalesce,
                                   to_edge_index, to_torch_coo_tensor, is_undirected, to_dense_adj)

from model.encoder import Encoder
from model.vq import VectorQuantize

EPS = 1e-6


def ensure_finetune_lr(params, canonical_key: str = "finetune_lr", alias_key: str = "lr"):
    """
    Keep the canonical finetune LR and its alias (`--lr`) in sync.
    Preference order:
      1. Explicit alias value if provided.
      2. Existing canonical value.
    The resolved value is written back to both keys so downstream code can
    reliably read either one (e.g., when wandb injects `lr` separately).
    """
    if not isinstance(params, dict):
        return params

    alias_is_set = alias_key in params and params[alias_key] is not None
    canonical_is_set = canonical_key in params and params[canonical_key] is not None

    if alias_is_set:
        resolved = params[alias_key]
    elif canonical_is_set:
        resolved = params[canonical_key]
    else:
        return params

    params[canonical_key] = resolved
    params[alias_key] = resolved
    return params


def get_pretrain_run_id(params, default: str = "default") -> str:
    if not isinstance(params, dict):
        return default
    run_id = params.get("pretrain_run_id") or params.get("pt_run_id") or params.get("run_id")
    if run_id is None:
        return default
    run_id = str(run_id).strip()
    return run_id if run_id else default


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def get_mask(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def flip_edges(data, p=0.2):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip_edges = int(num_edges * p / 2)
    else:
        num_flip_edges = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]

    flipped_edges = torch.randint(0, num_nodes, size=(num_flip_edges, 2))

    for n1, n2 in flipped_edges:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    edge_index = adj.to_sparse().coalesce().indices()
    data.edge_index = edge_index
    data.edge_attr = None
    return data


def get_device(params, optimized_params=None):
    if optimized_params is None or len(optimized_params) == 0:
        device = torch.device(f"cuda:{params['device']}")
    else:
        device = torch.device(f"cuda")
    return device


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def get_device_from_model(model):
    return next(model.parameters()).device


def active_code(encoder, vq, data):
    z = encoder(data.x, data.edge_index, data.edge_attr)
    _, indices, _, _ = vq(z)
    codebook_size = vq.codebook_size
    codebook_head = vq.heads
    return indices.unique(), indices.unique().numel() / (codebook_size * codebook_head)


def load_params(model, path):
    if torch.cuda.is_available():
        state = torch.load(path)
    else:
        state = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(model, Encoder):
        model.load_state_dict(state)
    elif isinstance(model, VectorQuantize):
        z = torch.randn(100, model.dim)
        model(z)
        model.load_state_dict(state)
    return model


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize(embedding, label=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label, cmap='tab10')
    plt.show()


def mask2idx(mask):
    return torch.where(mask == True)[0]


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask
