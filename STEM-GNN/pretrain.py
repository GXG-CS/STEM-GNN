#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import yaml
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader

from dataset.process_datasets import get_pt_data, get_train_node_idx, WEIGHT
from model.encoder import Encoder, InnerProductDecoder
from model.pt_model import PretrainModel
from model.vq import VectorQuantize
from utils.args import get_args_pretrain
from utils.others import seed_everything, get_scheduler, get_device_from_model, check_path, get_pretrain_run_id

import wandb


def pretrain(model, loader, optimizer, params, scheduler=None, no_codebook=False):
    model.train()
    device = get_device_from_model(model)

    for data in loader:
        bs = data.batch_size
        data_x_is_idx = data.x.size(0) != data.node_text_feat.size(0)

        if data_x_is_idx:
            x = data.node_text_feat[data.x].to(device)
        else:
            x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_text_feat[data.xe].to(device)
        graph = [x, edge_index, edge_attr]

        aug_x, _ = mask_feature(x, p=params['feat_p'])
        aug_edge_index, aug_edge_attr = dropout_adj(
            edge_index, edge_attr, p=params['edge_p'], force_undirected=True, num_nodes=x.size(0)
        )
        aug_graph = [aug_x, aug_edge_index, aug_edge_attr]

        z, quantize, indices, losses = model(
            aug_graph, graph, params['topo_recon_ratio'], bs=bs, no_codebook=no_codebook
        )

        feat_recon_loss = params['feat_lambda'] * losses['feat_recon_loss']
        topo_recon_loss = params['topo_lambda'] * losses['topo_recon_loss']
        topo_sem_recon_loss = params['topo_sem_lambda'] * losses['topo_sem_recon_loss']
        sem_recon_loss = params['sem_lambda'] * losses['sem_recon_loss']
        commit_loss = losses['commit_loss']
        env_reg_loss = losses['env_reg_loss']
        env_loss = params.get('lamda_env', 0.0) * env_reg_loss
        loss = feat_recon_loss + topo_recon_loss + topo_sem_recon_loss + sem_recon_loss + commit_loss + env_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        model.ema_update_sem_encoder(decay=params['sem_encoder_decay'])

        losses = {
            'losses/feat_recon_loss': feat_recon_loss.item(),
            'losses/topo_recon_loss': topo_recon_loss.item(),
            'losses/topo_sem_recon_loss': topo_sem_recon_loss.item(),
            'losses/sem_recon_loss': sem_recon_loss.item(),
            'losses/commit_loss': commit_loss.item(),
            'losses/env_reg_loss': env_reg_loss.item(),
            'losses/env_loss': env_loss.item(),
            'losses/loss': loss.item(),
        }

        wandb.log(losses)


def run(params):
    seed_everything(params["seed"])
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    # Data
    pretrain_data = get_pt_data(params['data_path'], params["pretrain_dataset"])

    # Model
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
        kmeans_init=False,
        ema_update=False,
    )

    feat_recon_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_recon_decoder = InnerProductDecoder(hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"])
    topo_sem_recon_decoder = nn.Linear(params["hidden_dim"] * 2, params["hidden_dim"])

    pretrain_model = PretrainModel(
        encoder=encoder, vq=vq,
        feat_recon_decoder=feat_recon_decoder,
        topo_recon_decoder=topo_recon_decoder,
        topo_sem_recon_decoder=topo_sem_recon_decoder,
    ).to(device)

    # Optimizer

    optimizer = AdamW(pretrain_model.parameters(), lr=params["pretrain_lr"],
                      weight_decay=params["pretrain_weight_decay"])
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["pretrain_epochs"])

    run_id = get_pretrain_run_id(params)
    save_root = osp.join(params["model_path"], run_id)
    check_path(save_root)

    for i in range(1, params["pretrain_epochs"] + 1):
        # Loader
        # Define the loader inside the loop to enable weighted sampling.
        batch_size = params["pretrain_batch_size"]
        if batch_size != 0:
            weights = list(WEIGHT[params["pretrain_dataset"]].values())
            train_nodes = get_train_node_idx(pretrain_data, weights)
            print("Number of training nodes is {}".format(len(train_nodes)))

            loader = NeighborLoader(pretrain_data, input_nodes=train_nodes,
                                    num_neighbors=[10] * params["num_layers"],
                                    batch_size=batch_size, shuffle=True)
            print("Number of mini-batches is {} at epoch {}.".format(len(loader), i))

        # Pretrain
        pretrain(model=pretrain_model, loader=loader, optimizer=optimizer, params=params, scheduler=scheduler)

        # Save the model
        try:
            pretrain_model.save_encoder(osp.join(save_root, f"encoder_{i}.pt"))
            pretrain_model.save_vq(osp.join(save_root, f"vq_{i}.pt"))
            print("Save the model at epoch {}".format(i))
        except:
            print("Failed to save the model at epoch {}".format(i))

    wandb.finish()


if __name__ == "__main__":
    params = get_args_pretrain()

    default_params = None
    if params['use_params']:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'pretrain.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
        params = get_args_pretrain(default_params=default_params)

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    run_id = get_pretrain_run_id(params)
    wandb.init(
        project="STEM-GNN-Pretrain",
        name=f"Pretrain ({run_id})",
        mode="disabled" if params["debug"] else "online",
        config=params,
    )

    run(params)
