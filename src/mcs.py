import logging

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph

import networkx as nx

class MCSDistance:
    def __call__(self, x1, edge_index1, x2, edge_index2, lam):
        pass


class MCSDistanceSparse(MCSDistance):
    def __call__(self, x1, edge_index1, x2, edge_index2, lam):

        # if len(x1>=20) or len(x2>=20):
        #     logging.warning("distance computing for more than 20 size")
        data1 = Data(x1.clone().detach().cpu(), edge_index1.clone().detach().cpu())
        data2 = Data(x2.clone().detach().cpu(), edge_index2.clone().detach().cpu())
        G1 = to_networkx(data1)
        G2 = to_networkx(data2)

        matcher = nx.algorithms.isomorphism.ISMAGS(G1, G2, cache={})
        mcs_mappings_iter = matcher.largest_common_subgraph()

        last_min = torch.ones(1)
        last_min = last_min * 100000
        no_mapping = True

        for mapping in mcs_mappings_iter:
            dist = __compute_dist__(x1, edge_index1, x2, edge_index2, lam, mapping)
            if dist < last_min:
                last_min = dist
            no_mapping = False
        if no_mapping:
            logging.warning("no mapping found")
            last_min = __compute_dist__(x1, edge_index1, x2, edge_index2, lam, {})

        # if len(x1>=20) or len(x2>=20):
        #     logging.warning("distance computed for more that 12 size end")

        del mcs_mappings_iter
        del matcher
        del G1
        del G2
        del data1
        del data2

        gc.collect()

        return last_min


def batch_to_datalist(x, edge_index, batch) -> list[Data]:
    """
    Splits the pooled batch into individual Data objects.

    Args:
        x (Tensor): Node features after pooling [num_pooled_nodes, num_features].
        edge_index (Tensor): Edge indices after pooling [2, num_pooled_edges].
        batch (Tensor): Batch assignments after pooling [num_pooled_nodes].

    Returns:
        List[Data]: A list of Data objects, each representing a pooled graph.
    """
    data_list = []
    num_graphs = batch.max().item() + 1

    for i in range(num_graphs):
        node_mask = (batch == i)
        node_indices = node_mask.nonzero(as_tuple=False).view(-1)

        if node_indices.numel() == 0:
            continue  # Skip empty graphs if any

        # Extract node features for the current graph
        x_i = x[node_indices]

        # Extract edge indices for the current graph
        # Create a subgraph by masking edges that connect nodes within the current graph

        # Relabel edge indices to start from 0
        sub_edge_index, _ = subgraph(node_mask, edge_index, relabel_nodes=True)

        # Create a Data object for the current graph
        data_i = (x_i, sub_edge_index,)

        data_list.append(data_i)

    return data_list


def pairwise_cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine distance between two sets of vectors.

    Args:
        x1 (torch.Tensor): Tensor of shape (m, v)
        x2 (torch.Tensor): Tensor of shape (n, v)

    Returns:
        torch.Tensor: Pairwise cosine distance matrix of shape (m, n)
    """
    # Step 1: Normalize the input tensors
    x1_norm = x1 / x1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Step 2: Compute cosine similarity matrix
    # This results in a (m, n) matrix where each element [i, j] is the cosine similarity between x1[i] and x2[j]
    cosine_similarity_matrix = torch.mm(x1_norm, x2_norm.t())

    # Step 3: Convert cosine similarity to cosine distance
    cosine_distance_matrix = 1 - cosine_similarity_matrix

    return cosine_distance_matrix

def __compute_dist__(x1, edge_index1, x2, edge_index2, lam, mapping):
    """
    needs unmasked graph features and adj
    """
    num_v_1 = len(x1)
    num_v_2 = len(x2)

    # d, X, _ = self.__compute_nondif_metric(data1, adj1, data2, adj2, C)
    # d, X, _ = subgraph_isomorphism_distance_pulp(data1, adj1, data2, adj2)

    # X_adj = torch.zeros((num_v_1, num_v_2), device=x1.device)
    X = torch.zeros((num_v_1 + 1, num_v_2 + 1), device=x1.device)  # , requires_grad=False)

    C = pairwise_cosine_distance(x1, x2)

    for u, i in mapping.items():
        X[u, i] = 1.0

    for i in range(num_v_1):
        if X[i].sum() == 0: X[i, num_v_2] = 1
    for i in range(num_v_2):
        if X[:, i].sum() == 0: X[num_v_1, i] = 1

    # Manual padding for `C`
    pad_row = lam.expand(1, C.size(1))  # Create a row of `lam` with correct dimensions
    pad_col = lam.expand(C.size(0) + 1, 1)  # Create a column of `lam` with correct dimensions

    C = torch.cat([C, pad_row], dim=0)  # Add padding row
    C = torch.cat([C, pad_col], dim=1)  # Add padding column
    C[-1, -1] = 0  # Set bottom-right corner to 0

    dist = (C * X).sum()
    #     + torch.norm(adj1 - torch.matmul(X_adj, adj2 @ X_adj.T), p='fro') + torch.norm(
    # adj2 - torch.matmul(X_adj.T, adj1 @ X_adj), p='fro')
    # print('----------------')
    # print('----------------')
    # print(dist.requires_grad)
    # print('----------------')
    # print('----------------')

    return dist
