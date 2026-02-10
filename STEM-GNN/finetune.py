#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import yaml
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.args import get_args_finetune
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.others import seed_everything, load_params, freeze_params, ensure_finetune_lr, get_pretrain_run_id

from task.node import ft_node, eval_node
from task.link import ft_link, eval_link
from task.graph import ft_graph, eval_graph

import warnings
import wandb

warnings.filterwarnings("ignore")

dataset2task = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    "chemhiv": "graph",
    "chempcba": "graph",
}


def get_preprocess(params):
    if params['task'] == 'node':
        return pre_node
    elif params['task'] == 'link':
        return pre_link
    elif params['task'] == 'graph':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')


def get_ft(params):
    task = params['task']

    if task == "node":
        return ft_node
    elif task == "link":
        return ft_link
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Invalid Task")


def get_eval(params):
    task = params['task']

    if task == "node":
        return eval_node
    elif task == "link":
        return eval_link
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Invalid Task")


def run(params):
    if params["setting"] != "standard":
        raise ValueError("Only the standard setting is supported.")

    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    preprocess = get_preprocess(params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

    data_name = params["finetune_dataset"]
    task = params["task"]

    dataset, splits, labels, num_classes, num_tasks = get_finetune_graph(params['data_path'], data_name)
    num_classes = num_tasks if task == "graph" else num_classes
    params["num_classes"] = num_classes

    dataset = preprocess(dataset)
    data = dataset[0]
    data.y = labels

    single_seed = params.get("finetune_seed")

    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

    if single_seed is not None:
        if single_seed < 0 or single_seed >= len(splits):
            raise ValueError(f"Requested finetune_seed {single_seed} is out of range for {len(splits)} available splits.")
        splits = [splits[single_seed]]
        params["repeat"] = 1

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
        use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    # Load Pretrained Model
    pretrain_run_id = get_pretrain_run_id(params)
    pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if pretrain_path.lower() in {"default", "auto"}:
        pretrain_path = ""
    if pretrain_path and not osp.isabs(pretrain_path):
        pretrain_path = osp.join(params['pt_model_path'], pretrain_path)

    if pretrain_path or params["pretrain_dataset"] != 'na':
        pretrain_task = params['pretrain_task']

        if pretrain_path:
            path = pretrain_path
        elif pretrain_task == 'all':
            path = osp.join(params['pt_model_path'], pretrain_run_id)
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

    if params.get("freeze_vq", 1):
        freeze_params(vq)
        print("Freeze VQ parameters during finetuning")

    train_loader = None
    val_loader = None
    test_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()

    moe_print_interval = max(1, params["finetune_epochs"] // 20)

    for idx, split in enumerate(splits):
        run_seed = single_seed if single_seed is not None else idx
        seed_everything(run_seed)

        task_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        print(f"[Finetune] MoE enabled: {getattr(task_model.encoder, 'moe', False)} | MoE layers: {getattr(task_model.encoder, 'moe_layer_flags', [])}")

        opt_params = task_model.parameters()
        task_opt = AdamW(opt_params, lr=params["finetune_lr"])
        stopper = EarlyStopping(patience=params["early_stop"])

        if params["batch_size"] != 0 and task in ["node", "link"]:
            train_loader, subgraph_loader = get_loader(data, split, labels, params)
        elif params["batch_size"] != 0 and task == "graph":
            train_loader, val_loader, test_loader = get_loader(dataset, split, labels, params)

        for epoch in range(params["finetune_epochs"]):
            loss = finetune(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
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
                dataset=data if task in ["node", "link"] else dataset,
                loader=subgraph_loader if task in ["node", "link"] else [train_loader, val_loader, test_loader],
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
                "train/lin_loss": loss['act_loss'],
                "train/env_loss": loss['env_loss'],
                "train/jac_loss": loss['jac_loss'],
                "train/loss": loss['loss'],
                "train/train_value": result['train'],
                "train/val_value": result['val'],
                "train/test_value": result['test'],
            }
            log_payload.update(moe_log)
            wandb.log(log_payload)

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })

    best = logger.get_best()

    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    params = get_args_finetune()

    if params["use_params"]:
        dataset = params["finetune_dataset"]
        task = dataset2task[dataset]
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'finetune.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
        params = get_args_finetune(default_params=default_params[task][dataset])

    ensure_finetune_lr(params)
    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pt_model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    dataset = params["finetune_dataset"]
    task = dataset2task[dataset]
    params['task'] = task

    wandb.init(
        project="STEM-GNN-Finetune",
        name="{} - Finetune".format(str.upper(params["finetune_dataset"])),
        config=params,
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        tags=[params['setting']],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    print("Params loaded.")

    run(params)
