import torch

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model


def ft_graph(model, dataset, loader, optimizer, split, labels, params, scheduler=None, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.train()

    device = get_device_from_model(model)
    lamda_env = params.get("lamda_env", 0.0)
    lambda_act = 1.0

    total_act_loss = 0.0
    total_jac_loss = 0.0
    total_env_loss = 0.0
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat
        y = batch.y.to(torch.float64)

        z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
        env_reg = model.get_env_reg()

        act_loss = model.compute_activation_loss(z, y, task="multi") * lambda_act
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


def _predict_graphs(model, loader, device):
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat
        y = batch.y.to(torch.float64)

        z = model.encode_graph(x, edge_index, edge_attr, batch.batch, pool="mean")
        pred = model.get_lin_logits(z).mean(1)

        preds.append(pred.detach())
        labels.append(y)
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def _evaluate_loader(model, loader, device, params):
    if loader is None or len(loader) == 0:
        return float("nan")
    pred, y = _predict_graphs(model, loader, device)
    return evaluate(pred, y, None, params)


def eval_graph(model, dataset, loader, split, labels, params, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.eval()
    device = get_device_from_model(model)

    train_loader, val_loader, test_loader = loader

    with torch.no_grad():
        train_value = _evaluate_loader(model, train_loader, device, params)
        val_value = _evaluate_loader(model, val_loader, device, params)
        test_value = _evaluate_loader(model, test_loader, device, params)

    return {
        "train": train_value,
        "val": val_value,
        "test": test_value,
        "metric": task2metric[params["task"]],
    }
