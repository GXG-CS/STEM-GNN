import torch

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model


def _edge_embeddings(z, edge_index):
    return (z[edge_index[0]] + z[edge_index[1]]) / 2


def ft_link(model, dataset, loader, optimizer, split, labels, params, scheduler=None, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.train()

    device = get_device_from_model(model)
    lamda_env = params.get("lamda_env", 0.0)
    lambda_act = 1.0

    if loader is None:
        x = dataset.node_text_feat[dataset.x].to(device)
        edge_index = dataset.edge_index.to(device)
        edge_attr = dataset.edge_text_feat[dataset.xe].to(device)
        y = labels.to(device)

        z = model.encode(x, edge_index, edge_attr)
        env_loss = lamda_env * model.get_env_reg()

        train_mask = split["train"].to(device)
        edge_index_train = edge_index[:, train_mask]
        y_train = y[train_mask]
        edge_z_train = _edge_embeddings(z, edge_index_train)

        act_loss = model.compute_activation_loss(edge_z_train, y_train) * lambda_act
        jac_loss = model.decoder_jacobian_penalty()
        loss = act_loss + jac_loss + env_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        return {
            "act_loss": act_loss.item(),
            "jac_loss": jac_loss.item(),
            "env_loss": env_loss.item(),
            "loss": loss.item(),
        }

    total_act_loss = 0.0
    total_jac_loss = 0.0
    total_env_loss = 0.0
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]
        edge_label_index = batch.edge_label_index
        y = batch.edge_label

        z = model.encode(x, edge_index, edge_attr)
        env_reg = model.get_env_reg()
        edge_z = _edge_embeddings(z, edge_label_index)

        act_loss = model.compute_activation_loss(edge_z, y) * lambda_act
        jac_loss = model.decoder_jacobian_penalty()
        env_loss = lamda_env * env_reg
        loss = act_loss + jac_loss + env_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_act_loss += act_loss.item()
        total_jac_loss += jac_loss.item()
        total_env_loss += env_loss.item()
        total_loss += loss.item()

    num_batches = len(loader)
    return {
        "act_loss": total_act_loss / num_batches,
        "jac_loss": total_jac_loss / num_batches,
        "env_loss": total_env_loss / num_batches,
        "loss": total_loss / num_batches,
    }


def eval_link(model, dataset, loader, split, labels, params, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.eval()
    device = get_device_from_model(model)

    with torch.no_grad():
        if loader is None:
            x = dataset.node_text_feat[dataset.x].to(device)
            edge_index = dataset.edge_index.to(device)
            edge_attr = dataset.edge_text_feat[dataset.xe].to(device)
            y = labels.to(device)

            z = model.encode(x, edge_index, edge_attr)
            edge_z = _edge_embeddings(z, edge_index)
            pred = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)
        else:
            preds, gts = [], []
            for batch in loader:
                batch = batch.to(device)
                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]
                edge_label_index = batch.edge_label_index
                y = batch.edge_label

                z = model.encode(x, edge_index, edge_attr)
                edge_z = _edge_embeddings(z, edge_label_index)
                pred = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)

                preds.append(pred.detach())
                gts.append(y)

            pred = torch.cat(preds, dim=0)
            y = torch.cat(gts, dim=0)

        train_mask = split["train"].to(pred.device)
        val_mask = split["valid"].to(pred.device)
        test_mask = split["test"].to(pred.device)

        train_value = evaluate(pred, y, train_mask, params)
        val_value = evaluate(pred, y, val_mask, params)
        test_value = evaluate(pred, y, test_mask, params)

    return {
        "train": train_value,
        "val": val_value,
        "test": test_value,
        "metric": task2metric[params["task"]],
    }
