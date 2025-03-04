import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_softmax import Sparsemax  # Assuming this is still custom
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.utils import softmax, add_self_loops
from torch_scatter import scatter
import torch_sparse
from torch_geometric.data import Data

class TwoHopNeighborhood:
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = torch.full((edge_index.size(1),), fill, dtype=torch.float, device=edge_index.device)

        index, value = torch_sparse.spspmm(edge_index, value, edge_index, value, n, n, n, coalesced=True)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            sp_tensor = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(n, n))
            sp_tensor = sp_tensor.coalesce(reduce="add")
            row, col = sp_tensor.coo()[:2]
            data.edge_index = torch.stack([row, col], dim=0)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            sp_tensor = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr, sparse_sizes=(n, n))
            sp_tensor = sp_tensor.coalesce(reduce="min")
            row, col, edge_attr = sp_tensor.coo()
            data.edge_index = torch.stack([row, col], dim=0)
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    def norm(self, edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=0, num_nodes=num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros(edge_weight.size(0), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = 1.0

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    f"Cached {self.cached_num_edges} edges, but found {edge_index.size(1)}"
                )
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class HGPSLPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slope=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slope = negative_slope

        self.att = Parameter(torch.empty(1, in_channels * 2))
        nn.init.xavier_uniform_(self.att)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()
        self.topk_pool = TopKPooling(in_channels, ratio=ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_info_score = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_info_score), dim=1)

        x, edge_index, edge_attr, batch, perm, _ = self.topk_pool(x, edge_index, edge_attr, batch, attn=score)

        if not self.sl:
            return x, edge_index, edge_attr, batch

        original_x = x
        if self.sample:
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones(
                    (edge_index.size(1),), dtype=torch.float, device=edge_index.device
                )
            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr

            # Use the same number of nodes as x for consistency
            hop_x = torch.ones((x.size(0), self.in_channels), device=x.device)
            hop_batch = batch
            _, new_edge_index, new_edge_attr, _, _, _ = self.topk_pool(
                hop_x, hop_edge_index, hop_edge_attr, hop_batch, attn=score[perm]
            )

            new_edge_index, new_edge_attr = add_self_loops(
                new_edge_index, new_edge_attr, fill_value=0, num_nodes=x.size(0)
            )
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slope) + new_edge_attr * self.lamb

            # Use a simple edge index for grouping (0 to num_edges-1)
            edge_indices = torch.arange(weights.size(0), device=weights.device)
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, edge_indices)
            else:
                new_edge_attr = softmax(weights, edge_indices, num_nodes=weights.size(0))

            mask = new_edge_attr > 0
            new_edge_index, new_edge_attr = new_edge_index[:, mask], new_edge_attr[mask]
        else:
            if edge_attr is None:
                edge_attr = torch.ones(
                    (edge_index.size(1),), dtype=x.dtype, device=x.device
                )
            num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0, reduce="sum")
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)

            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0

            new_edge_index = adj.to_sparse().indices()
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slope)

            induced_row, induced_col = edge_index
            adj[induced_row, induced_col] += edge_attr * self.lamb
            weights = adj[row, col]

            edge_indices = torch.arange(weights.size(0), device=weights.device)
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, edge_indices)
            else:
                new_edge_attr = softmax(weights, edge_indices, num_nodes=weights.size(0))

            mask = new_edge_attr > 0
            new_edge_index, new_edge_attr = new_edge_index[:, mask], new_edge_attr[mask]

        return x, new_edge_index, new_edge_attr, batch

#
# ############################
#
# # Custom coalesce function from your GitHub source
# def coalesce(index, value, m, n, op="add"):
#     from torch_sparse.storage import SparseStorage
#     storage = SparseStorage(row=index[0], col=index[1], value=value,
#                             sparse_sizes=(m, n), is_sorted=False)
#     storage = storage.coalesce(reduce=op)
#     return torch.stack([storage.row(), storage.col()], dim=0), storage.value()
#
# class TwoHopNeighborhood:
#     def __call__(self, data):
#         edge_index, edge_attr = data.edge_index, data.edge_attr
#         n = data.num_nodes
#
#         fill = 1e16  # Still used for spspmm, not coalesce
#         value = torch.full((edge_index.size(1),), fill, dtype=torch.float, device=edge_index.device)
#
#         # Sparse matrix multiplication to get two-hop neighbors
#         index, value = torch_sparse.spspmm(edge_index, value, edge_index, value, n, n, n, coalesced=True)
#
#         edge_index = torch.cat([edge_index, index], dim=1)
#         if edge_attr is None:
#             # Use custom coalesce without fill_value
#             data.edge_index, _ = coalesce(edge_index, None, n, n, op="add")
#         else:
#             # Extend edge_attr with fill values for new edges
#             value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
#             value = value.expand(-1, *list(edge_attr.size())[1:])
#             edge_attr = torch.cat([edge_attr, value], dim=0)
#             # Use custom coalesce with min operation
#             data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op="min")
#             # Manually reset large values to 0 (replacing fill_value logic)
#             edge_attr[edge_attr >= fill] = 0
#             data.edge_attr = edge_attr
#
#         return data
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}()"
#
# class NodeInformationScore(MessagePassing):
#     def __init__(self, improved=False, cached=False, **kwargs):
#         super().__init__(aggr="add", **kwargs)
#         self.improved = improved
#         self.cached = cached
#         self.cached_result = None
#         self.cached_num_edges = None
#
#     def norm(self, edge_index, num_nodes, edge_weight, dtype=None):
#         if edge_weight is None:
#             edge_weight = torch.ones(
#                 (edge_index.size(1),), dtype=dtype, device=edge_index.device
#             )
#
#         row, col = edge_index
#         deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
#
#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=0, num_nodes=num_nodes)
#
#         row, col = edge_index
#         expand_deg = torch.zeros(edge_weight.size(0), dtype=dtype, device=edge_index.device)
#         expand_deg[-num_nodes:] = 1.0
#
#         return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#
#     def forward(self, x, edge_index, edge_weight=None):
#         if self.cached and self.cached_result is not None:
#             if edge_index.size(1) != self.cached_num_edges:
#                 raise RuntimeError(
#                     f"Cached {self.cached_num_edges} edges, but found {edge_index.size(1)}"
#                 )
#         if not self.cached or self.cached_result is None:
#             self.cached_num_edges = edge_index.size(1)
#             edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
#             self.cached_result = edge_index, norm
#
#         edge_index, norm = self.cached_result
#         return self.propagate(edge_index, x=x, norm=norm)
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
#
#     def update(self, aggr_out):
#         return aggr_out
#
# class HGPSLPool(nn.Module):
#     def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slope=0.2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.ratio = ratio
#         self.sample = sample
#         self.sparse = sparse
#         self.sl = sl
#         self.lamb = lamb
#         self.negative_slope = negative_slope
#
#         self.att = Parameter(torch.empty(1, in_channels * 2))
#         nn.init.xavier_uniform_(self.att)
#         self.sparse_attention = Sparsemax()
#         self.neighbor_augment = TwoHopNeighborhood()
#         self.calc_information_score = NodeInformationScore()
#         self.topk_pool = TopKPooling(in_channels, ratio=ratio)
#
#     def forward(self, x, edge_index, edge_attr=None, batch=None):
#         if batch is None:
#             batch = edge_index.new_zeros(x.size(0))
#
#         x_info_score = self.calc_information_score(x, edge_index, edge_attr)
#         score = torch.sum(torch.abs(x_info_score), dim=1)
#
#         x, edge_index, edge_attr, batch, perm, _ = self.topk_pool(x, edge_index, edge_attr, batch, attn=score)
#
#         if not self.sl:
#             return x, edge_index, edge_attr, batch
#
#         original_x = x
#         if self.sample:
#             k_hop = 3
#             if edge_attr is None:
#                 edge_attr = torch.ones(
#                     (edge_index.size(1),), dtype=torch.float, device=edge_index.device
#                 )
#             hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
#             for _ in range(k_hop - 1):
#                 hop_data = self.neighbor_augment(hop_data)
#             hop_edge_index = hop_data.edge_index
#             hop_edge_attr = hop_data.edge_attr
#             new_edge_index, new_edge_attr = self.topk_pool.filter_adj(
#                 hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0)
#             )
#
#             new_edge_index, new_edge_attr = add_self_loops(
#                 new_edge_index, new_edge_attr, fill_value=0, num_nodes=x.size(0)
#             )
#             row, col = new_edge_index
#             weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
#             weights = F.leaky_relu(weights, self.negative_slope) + new_edge_attr * self.lamb
#
#             if self.sparse:
#                 new_edge_attr = self.sparse_attention(weights, row)
#             else:
#                 new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
#
#             mask = new_edge_attr > 0
#             new_edge_index, new_edge_attr = new_edge_index[:, mask], new_edge_attr[mask]
#         else:
#             if edge_attr is None:
#                 edge_attr = torch.ones(
#                     (edge_index.size(1),), dtype=x.dtype, device=x.device
#                 )
#             num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0, reduce="sum")
#             shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
#             cum_num_nodes = num_nodes.cumsum(dim=0)
#
#             adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
#             for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
#                 adj[idx_i:idx_j, idx_i:idx_j] = 1.0
#
#             new_edge_index = adj.to_sparse().indices()
#             row, col = new_edge_index
#             weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
#             weights = F.leaky_relu(weights, self.negative_slope)
#
#             induced_row, induced_col = edge_index
#             adj[induced_row, induced_col] += edge_attr * self.lamb
#             weights = adj[row, col]
#
#             if self.sparse:
#                 new_edge_attr = self.sparse_attention(weights, row)
#             else:
#                 new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
#
#             mask = new_edge_attr > 0
#             new_edge_index, new_edge_attr = new_edge_index[:, mask], new_edge_attr[mask]
#
#         return x, new_edge_index, new_edge_attr, batch