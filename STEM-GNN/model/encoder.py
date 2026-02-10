from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, SAGEConv, GATConv, GCNConv, GINConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch_scatter import scatter_mean


class MySAGEConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Union[Tensor, None] = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, xe=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, xe) -> Tensor:
        if xe is not None:
            x_j = x_j + xe
        return F.relu(x_j)

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class MixtureSageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_experts: int, residual: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.residual = residual and (in_dim == out_dim)
        self.weights = nn.Parameter(torch.empty(num_experts, in_dim * 2, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        row, col = edge_index
        agg = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        combined = torch.cat([agg, x], dim=-1)  # [N, 2 * in_dim]
        outputs = torch.einsum('nd,kdo->nko', combined, self.weights)  # [N, K, out_dim]
        if self.residual:
            outputs = outputs + x.unsqueeze(1)
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation, num_layers, backbone='sage',
                 normalize='none', dropout=0.0,
                 moe=False, num_experts=3, tau=1.0,
                 moe_layers='all'):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize
        self.moe = moe and num_experts > 1
        self.num_experts = num_experts
        self.tau = tau
        self.moe_layers = moe_layers

        self.activation = activation()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.env_encoders = nn.ModuleList()
        self._last_env_reg: Optional[Tensor] = None
        self._moe_usage: Optional[list] = None
        self._router_cache: Optional[list] = None
        self._cache_router = False

        self.moe_layer_flags = self._build_moe_layer_flags()

        dims = [input_dim] + [hidden_dim] * num_layers

        env_layer_idx = 0
        for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if self.moe_layer_flags[layer_idx] and self.backbone == 'sage':
                moe_layer = MixtureSageLayer(in_dim, out_dim, self.num_experts, residual=True)
                self.layers.append(moe_layer)
                self.env_encoders.append(nn.Linear(in_dim, self.num_experts))
                env_layer_idx += 1
            else:
                self.layers.append(self._build_conv(in_dim, out_dim))
            self.norms.append(nn.BatchNorm1d(out_dim))

        self.reset_parameters()

    def _build_moe_layer_flags(self):
        if not self.moe:
            return [False] * self.num_layers
        if self.moe_layers == 'all':
            return [True] * self.num_layers
        if self.moe_layers == 'last':
            flags = [False] * self.num_layers
            if self.num_layers > 0:
                flags[-1] = True
            return flags
        if self.moe_layers == 'none':
            return [False] * self.num_layers
        raise ValueError(f'Unsupported moe_layers setting: {self.moe_layers}')

    def _build_conv(self, in_dim, out_dim):
        if self.backbone == 'sage':
            return MySAGEConv(in_dim, out_dim, aggr='mean', normalize=False, root_weight=True)
        if self.backbone == 'gat':
            return GATConv(in_dim, out_dim, heads=1)
        if self.backbone == 'gcn':
            return GCNConv(in_dim, out_dim)
        if self.backbone == 'gin':
            return GINConv(nn.Linear(in_dim, out_dim))
        raise ValueError(f'Unsupported backbone: {self.backbone}')

    def _reg_loss(self, weights, logits):
        log_pi = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return torch.mean(torch.sum(weights * log_pi, dim=-1))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for encoder in self.env_encoders:
            if hasattr(encoder, 'reset_parameters'):
                encoder.reset_parameters()
        self._last_env_reg = None
        self._reset_moe_usage()

    def _reset_moe_usage(self):
        self._moe_usage = None

    def enable_router_cache(self, flag: bool = True):
        self._cache_router = flag
        self._router_cache = [] if flag else None

    def get_router_cache(self, reset: bool = True):
        out = self._router_cache or []
        if reset:
            self._router_cache = [] if self._cache_router else None
        return out

    def _ensure_moe_usage(self, device, dtype):
        if not self.moe:
            return
        if self._moe_usage is not None:
            return
        self._moe_usage = []
        for flag in self.moe_layer_flags:
            if flag:
                self._moe_usage.append({
                    'sum_prob': torch.zeros(self.num_experts, device=device, dtype=dtype),
                    'sum_top1': torch.zeros(self.num_experts, device=device, dtype=dtype),
                    'count': 0,
                })

    def _update_moe_usage(self, env_idx, weights):
        if not self.moe:
            return
        weights = weights.detach()
        self._ensure_moe_usage(weights.device, weights.dtype)
        if self._moe_usage is None:
            return
        stats = self._moe_usage[env_idx]
        stats['sum_prob'] += weights.sum(dim=0)
        top1 = F.one_hot(weights.argmax(dim=-1), num_classes=self.num_experts).type_as(weights)
        stats['sum_top1'] += top1.sum(dim=0)
        stats['count'] += weights.size(0)

    def get_moe_usage(self, reset=True):
        if not self.moe or self._moe_usage is None:
            return []
        usage = []
        env_idx = 0
        for layer_idx, flag in enumerate(self.moe_layer_flags):
            if not flag:
                continue
            stats = self._moe_usage[env_idx]
            denom = max(stats['count'], 1)
            avg_prob = (stats['sum_prob'] / denom).detach().cpu().tolist()
            top1_frac = (stats['sum_top1'] / denom).detach().cpu().tolist()
            usage.append({
                "layer": layer_idx,
                "avg_prob": avg_prob,
                "top1_frac": top1_frac,
            })
            env_idx += 1
        if reset:
            self._reset_moe_usage()
        return usage

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return z

    def encode(self, x, edge_index, edge_attr=None):
        z = x
        env_idx = 0
        env_reg_total: Optional[Tensor] = None
        env_layers = 0
        self._last_env_reg = None

        for i in range(self.num_layers):
            layer = self.layers[i]
            if isinstance(layer, MixtureSageLayer):
                logits = self.env_encoders[env_idx](z)
                if self.training:
                    weights = F.gumbel_softmax(logits, tau=self.tau, dim=-1)
                    reg = self._reg_loss(weights, logits)
                    env_reg_total = reg if env_reg_total is None else env_reg_total + reg
                    env_layers += 1
                else:
                    weights = F.softmax(logits, dim=-1)
                if self._cache_router:
                    if self._router_cache is None:
                        self._router_cache = []
                    self._router_cache.append(weights.detach())
                self._update_moe_usage(env_idx, weights)

                expert_outputs = layer(z, edge_index, edge_attr)
                z = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
                env_idx += 1
            else:
                z = layer(z, edge_index, edge_attr)

            if self.normalize != 'none':
                z = self.norms[i](z)
            if i < self.num_layers - 1:
                z = self.activation(z)
                z = self.dropout(z)

        if env_reg_total is not None and self.training and env_layers > 0:
            self._last_env_reg = env_reg_total / env_layers
        else:
            self._last_env_reg = z.new_zeros(1)
        return z

    def get_env_reg(self, reset=True):
        if self._last_env_reg is None:
            device = next(self.parameters()).device
            reg = torch.zeros(1, device=device)
        else:
            reg = self._last_env_reg
        if reset:
            self._last_env_reg = None
        return reg


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def __init__(self, hidden_dim=None, output_dim=None):
        super().__init__()
        self.proj_z = False
        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
