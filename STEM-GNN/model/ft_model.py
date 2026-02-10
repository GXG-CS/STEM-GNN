import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


def compute_multitask_loss(pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


class TaskModel(nn.Module):
    """Linear decoder built on top of the encoder + VQ backbone."""

    def __init__(self, encoder, vq, num_classes, params):
        super().__init__()

        self.encoder = encoder
        self.vq = vq

        num_heads, _, code_dim = vq.codebook.shape
        self.num_classes = num_classes
        self.num_heads = vq._codebook.num_codebooks if vq is not None else 1

        self.separate_decoder_for_each_head = params["separate_decoder_for_each_head"]
        self.decoder_jac_coeff = params.get("decoder_jac_coeff", 0.0)
        self.use_vq = params.get("use_vq", 1)

        if self.separate_decoder_for_each_head:
            self.decoder = nn.Linear(code_dim * num_heads, num_classes * num_heads)
        else:
            self.decoder = nn.Linear(code_dim, num_classes)

    def decoder_jacobian_penalty(self):
        if self.decoder_jac_coeff <= 0:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        weight = self._get_linear_weight(self.decoder)
        return self.decoder_jac_coeff * weight.pow(2).sum()

    @staticmethod
    def _get_linear_weight(module: nn.Module):
        if isinstance(module, nn.Linear):
            return module.weight
        raise TypeError(f"Unsupported decoder module: {type(module).__name__}")

    def encode(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr)

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None, pool="mean"):
        z = self.encoder(x, edge_index, edge_attr)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def get_env_reg(self, reset=True):
        if hasattr(self.encoder, "get_env_reg"):
            return self.encoder.get_env_reg(reset=reset)
        device = next(self.parameters()).device
        return torch.zeros(1, device=device)

    def get_moe_usage(self, reset=True):
        if hasattr(self.encoder, "get_moe_usage"):
            return self.encoder.get_moe_usage(reset=reset)
        return []

    def compute_activation_loss(self, z, y, task="single"):
        logits = self.get_lin_logits(z).mean(1)
        if task == "single":
            return F.cross_entropy(logits, y)
        if task == "multi":
            return compute_multitask_loss(logits, y)
        raise ValueError('task must be either "single" or "multi"')

    def get_lin_logits(self, z):
        if self.use_vq:
            quantize, _, _, codes = self.vq(z)
            if self.separate_decoder_for_each_head:
                pred = self.decoder(codes).reshape(-1, self.num_heads, self.num_classes)
            else:
                pred = self.decoder(quantize).reshape(-1, 1, self.num_classes)
            return pred
        if self.separate_decoder_for_each_head:
            codes = self.vq.project_in(z)
            pred = self.decoder(codes).reshape(-1, self.num_heads, self.num_classes)
        else:
            pred = self.decoder(z).reshape(-1, 1, self.num_classes)
        return pred

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        return self.get_lin_logits(z)
