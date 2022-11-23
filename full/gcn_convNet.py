import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_geometric.typing import OptTensor, PairTensor
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from math import log

import math
import random
import os
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


set_seed(1)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.MetaNet = Linear(in_channels * 2, out_channels * 2, bias=bias)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j}
        \frac{1}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`i` to target
    node :obj:`j` (default: :obj:`1`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, res: bool = False,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.res = res
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.weight = Linear(in_channels, out_channels)  # Parameter(torch.Tensor(in_channels, out_channels))
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.MetaNet = Linear(in_channels * 2, out_channels * 2, bias=bias)
        self.lin3 = Linear(in_channels, out_channels)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        self.weight.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        x_r = self.weight(x)
        # x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x_r, edge_weight=edge_weight,
                             size=None)

        # if self.bias is not None:
        #     out += self.bias

        # out+=self.lin3(x)

        # if self.res:
        #     out+=x

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class OurConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = False,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super(OurConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.weight = Linear(in_channels, out_channels)  # Parameter(torch.Tensor(in_channels, out_channels))
        self.MetaNet = Linear(in_channels * 2, out_channels * 2, bias=bias)
        self.lin3 = Linear(in_channels, out_channels, bias=bias)
        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        # if bias:
        #     self.bias = Parameter(torch.FloatTensor(out_channels * 2))
        #     #self.bias = Parameter(torch.FloatTensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.reset_parameters()
        self.MetaNet.reset_parameters()
        self.lin3.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        out = self.propagate(edge_index, x=x,
                             size=None)
        # if self.bias is not None:
        #     out += self.bias
        # out = self.lin_l(out)

        out += self.lin_r(x)
        #
        # if isinstance(x, Tensor):
        #     x: OptPairTensor = (x, x)
        #
        # out = self.propagate(edge_index, x=x, size=None)
        # out = self.lin_l(out)
        #
        # x_r = x[1]
        # if x_r is not None:
        #     out += self.lin_r(x_r)

        return out

    def message(self, x_i, x_j: Tensor) -> Tensor:
        # z = torch.cat([x_i, x_j], dim=-1)
        # embedding = self.MetaNet(z)
        # Sigmoid = torch.nn.Sigmoid()
        # embedding = Sigmoid(embedding)
        # # #x_j= torch.matmul(x_j, self.weight)
        # # x_j = self.weight(x_j)
        # gammas, betas = torch.split(embedding, x_j.size(1), dim=-1)

        z = torch.cat([x_i, x_j], dim=-1)
        embedding = self.MetaNet(z)
        Sigmoid = torch.nn.Sigmoid()
        embedding = Sigmoid(embedding)
        # #x_j= torch.matmul(x_j, self.weight)
        x_j = self.weight(x_j)
        gammas, betas = torch.split(embedding, x_j.size(1), dim=-1)
        # gammas=F.gumbel_softmax(gammas, tau=1, hard=False)

        return gammas * x_j  # performance 0.779
        # return gammas * x_j + betas  # performance 0.760

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        # self.edge_weight = 1
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # edge_weight = self.edge_weight
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:

        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class OurGCN2ConvNewData(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, eachLayer: bool = False, MetaNet1=False, MetaNet2=False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super(OurGCN2ConvNewData, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha  # [layer-1]
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))
        # self.alpha = torch.nn.Parameter(torch.Tensor([0.1]))
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))
        # self.MetaNet = Linear(in_channels * 2, out_channels * 2, bias=bias)
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # if self.MetaNet1 is not None:
        #
        #     embedding = self.MetaNet1(x)
        #     eT = embedding.permute(1, 0)
        #     embedding = self.MetaNet2(eT)
        #     Sigmoid = torch.nn.Sigmoid()
        #     embedding = Sigmoid(embedding)
        #     self.alpha = embedding[0]
        #     self.beta = embedding[1]
        # print('startMetaNet')
        # edge_weight = adj._values()
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # print(self.alpha,self.beta)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class OurGCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, eachLayer: bool = False, MetaNet1=False, MetaNet2=False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        # kwargs.setdefault('flow', 'target_to_source')
        super(OurGCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha  # [layer-1]
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))
        # self.alpha = torch.nn.Parameter(torch.Tensor([0.1]))
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))
        # self.MetaNet = Linear(in_channels * 2, out_channels * 2, bias=bias)
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # if self.MetaNet1 is not None:
        #
        #     embedding = self.MetaNet1(x)
        #     eT = embedding.permute(1, 0)
        #     embedding = self.MetaNet2(eT)
        #     Sigmoid = torch.nn.Sigmoid()
        #     embedding = Sigmoid(embedding)
        #     self.alpha = embedding[0]
        #     self.beta = embedding[1]
        # print('startMetaNet')

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # print(self.alpha,self.beta)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class OurGammaGCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, eachLayer: bool = False, MetaNet1=False, MetaNet2=False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(OurGammaGCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha  # [layer-1]
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))
        # self.alpha = torch.nn.Parameter(torch.Tensor([0.1]))
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.layerNumber = layer
        self.firstGamma = 1
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))
        # self.MetaNet = Linear(channels*2, channels*2, bias=True)
        self.MetaNet = Linear(1, 2, bias=True)
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # if self.MetaNet1 is not None:
        #
        #     embedding = self.MetaNet1(x)
        #     eT = embedding.permute(1, 0)
        #     embedding = self.MetaNet2(eT)
        #     Sigmoid = torch.nn.Sigmoid()
        #     embedding = Sigmoid(embedding)
        #     self.alpha = embedding[0]
        #     self.beta = embedding[1]
        # print('startMetaNet')

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # print(self.alpha,self.beta)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_i, x_j: Tensor, edge_weight: Tensor) -> Tensor:

        # z = torch.cat([x_i, x_j], dim=-1)

        z = x_i * x_j
        z = torch.matmul(z, torch.ones(x_i.shape[1], 1).cuda())

        embedding = self.MetaNet(z)
        Sigmoid = torch.nn.Sigmoid()
        embedding = Sigmoid(embedding)
        # # #x_j= torch.matmul(x_j, self.weight)
        # gammas, betas = torch.split(embedding, x_j.size(1), dim=-1)
        gammas = embedding[:, 0:1]

        # firstGamma
        if self.layerNumber == 1:
            self.firstGamma = gammas
        else:
            gammas = self.firstGamma

        x_j = gammas * x_j
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class OurGammaValGCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, eachLayer: bool = False, MetaNet1=False, MetaNet2=False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(OurGammaValGCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha  # [layer-1]
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))
        # self.alpha = torch.nn.Parameter(torch.Tensor([0.1]))
        # self.beta = torch.nn.Parameter(torch.Tensor([0.5]))

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.layerNumber = layer
        self.firstGamma = 1
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))
        # self.MetaNet = Linear(channels*2, channels*2, bias=True)
        self.MetaNet = MetaNet1  # Linear(1, 2, bias=True)
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # if self.MetaNet1 is not None:
        #
        #     embedding = self.MetaNet1(x)
        #     eT = embedding.permute(1, 0)
        #     embedding = self.MetaNet2(eT)
        #     Sigmoid = torch.nn.Sigmoid()
        #     embedding = Sigmoid(embedding)
        #     self.alpha = embedding[0]
        #     self.beta = embedding[1]
        # print('startMetaNet')

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        # print(self.alpha,self.beta)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_i, x_j: Tensor, edge_weight: Tensor) -> Tensor:

        # z = torch.cat([x_i, x_j], dim=-1)

        z = x_i * x_j
        z = torch.matmul(z, torch.ones(x_i.shape[1], 1).cuda())

        embedding = self.MetaNet(z)
        Sigmoid = torch.nn.Sigmoid()
        embedding = Sigmoid(embedding)
        # # #x_j= torch.matmul(x_j, self.weight)
        # gammas, betas = torch.split(embedding, x_j.size(1), dim=-1)

        #

        gammas = embedding[:, 0:1]
        # gammas = torch.cat([gammas, 1-gammas], dim=-1)
        # Softmax = torch.nn.Softmax(dim=1)
        # gammas = F.gumbel_softmax(Softmax(gammas), tau=1, hard=False)
        # gammas = gammas[:, 0:1]

        # firstGamma

        if self.layerNumber == 1:
            self.firstGamma = gammas
        else:
            gammas = self.firstGamma

        x_j = gammas * x_j
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)
