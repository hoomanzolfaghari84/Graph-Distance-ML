import logging
import time
from itertools import combinations
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
import gc

import torch
# from torch.profiler import profile as tprofile
import torch.nn.functional as F
# from memory_profiler import profile
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx, subgraph
import networkx as nx

from hgpsl_pooling.layers import GCN, HGPSLPool

class PoolingLayers(torch.nn.Module):
    def __init__(self, args):
        super(PoolingLayers, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        self.conv4 = GCN(self.nhid, self.nhid)
        self.conv5 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool3 = HGPSLPool(self.nhid, 0.3, self.sample, self.sparse, self.sl, self.lamb)
        self.pool4 = HGPSLPool(self.nhid, 0.3, self.sample, self.sparse, self.sl, self.lamb)

        self.frozen = False

    def freeze(self, frozen):
        self.frozen = frozen

    def forward(self, x, edge_index, batch):
        # print(f'x shape: {x.shape}')
        # print(f'edge index shape: {edge_index.shape}')

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if len(edge_index[0])>1000:
            logging.warning(f' graph with size {len(x)} and edge no {len(edge_index[0])} after pooling')
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv5(x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch = self.pool4(x, edge_index, edge_attr, batch)
            x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        else:
            x3 = torch.zeros_like(x2)
            x5 = torch.zeros_like(x2)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        readouts = F.relu(x1) + F.relu(x2) + F.relu(x4) + F.relu(x3)  + F.relu(x5)

        return x, edge_index, edge_attr, batch, readouts


class HPGSL_Original_Model(torch.nn.Module):
    def __init__(self, args):
        super(HPGSL_Original_Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class MCSDistanceSparse:

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



async def distance_async(graph1, graph2, lam, distance_func):
    """
    Asynchronous wrapper to run distance_func in an executor.
    """
    loop = asyncio.get_event_loop()  # Get the current event loop
    # Run the blocking distance_func in a separate thread
    return await loop.run_in_executor(
        None, distance_func, graph1[0], graph1[1], graph2[0], graph2[1], lam
    )


def compute_distance_with_timeout(graph1, graph2, lam, distance_func, timeout):
    """
    Runs distance_func with a timeout using asyncio and run_in_executor.
    """
    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current event loop
    try:
        # Run the async function with a timeout
        return loop.run_until_complete(
            asyncio.wait_for(distance_async(graph1, graph2, lam, distance_func), timeout)
        )
    except asyncio.TimeoutError:
        # logging.warning(f"distance computation timed out after {timeout} seconds")
        return -1
    finally:
        loop.close()  # Ensure the loop is closed


class MCS_HGPSL(torch.nn.Module):
    def __init__(self, args, distance_func=MCSDistanceSparse(), lam_init=2, dist_comp_timeout=45):
        """
        Initializes the Custom_HGPSL model.

        Args:
            args: Configuration arguments containing model hyperparameters.
            distance_func: A function that takes two graph representations
                           (x1, edge_index1) and (x2, edge_index2) and returns a distance.
        """
        super(MCS_HGPSL, self).__init__()

        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.dist_comp_timeout = dist_comp_timeout

        self.pooling_layer = PoolingLayers(args)

        self.lin1 = torch.nn.Linear(2 * self.nhid, self.nhid)  ## NOTICE 2*self.nhid
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        self.__fine_tuning__ = False

        self.__lam__ = torch.nn.Parameter(torch.tensor(lam_init, dtype=torch.float32))
        self.distance_func = distance_func

    def get_lam(
            self):  # constraint lambda parameter to satisfy theoretical constraints, while being differentiable in R.
        return F.softplus(self.__lam__)

    def set_fine_tuning(self, fine_tuning, freeze_pooling=False):
        self.__fine_tuning__ = fine_tuning

        for param in self.pooling_layer.parameters():
            param.requires_grad = not freeze_pooling

    def forward(self, data):
        """
           Forward pass of the model.
           Args:
               :param batch:
               :param edge_index:
               :param x:
           Returns:
               logits: Log-softmax output for classification.
               distance_matrix: Pairwise distance matrix between graphs in the batch.
        """
        # logging.info("pooling")
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_nodes_Start = len(x)
        num_graphs = batch.max().item() + 1
        x, edge_index, edge_attr, batch, readouts = self.pooling_layer(x, edge_index, batch)
        # logging.info("pooling end")
        # print(f'node reductions in batch of lenght {num_graphs}: {num_nodes_Start - len(x)}')

        if (not self.__fine_tuning__):  # and  self.training:
            # Split the pooled batch into individual Data objects

            # logging.info("post pooling")
            # x = self.__post_pooling_transforms__(x)
            # logging.info("post pooling end")

            # return self.distance_func(x, edge_index, x, edge_index, self.get_lam())
            # logging.info("splitting batch")
            data_list = split_pooled_batch(x, edge_index, batch)
            # logging.info("splitting batch end")

            num_graphs = len(data_list)

            # Initialize an empty distance matrix
            distance_matrix = torch.zeros((num_graphs, num_graphs), device=x.device)

            # # Helper function to wrap distance_func with timeout
            # def compute_distance_with_timeout(graph1, graph2, lam):
            #     with ThreadPoolExecutor(max_workers=1) as executor:
            #         future = executor.submit(self.distance_func, graph1[0], graph1[1], graph2[0], graph2[1], lam)
            #         try:
            #             return future.result(timeout=self.dist_comp_timeout)  # dist_comp_timeout seconds timeout
            #         except  TimeoutError:
            #             logging.warning(f"distance computation timed out of {self.dist_comp_timeout}s - num_nodes= {len(graph1[0])},{len(graph2[0])} - num_edges = {len(graph1[1][0])},{len(graph2[1][0])}")
            #             return 0

            # Compute pairwise distances using the custom distance function
            for i in range(num_graphs):
                patience = 3
                for j in range(i + 1, num_graphs):

                    graph1 = data_list[i]
                    graph2 = data_list[j]

                    distance = self.distance_func(graph1[0], graph1[1], graph2[0], graph2[1], self.get_lam())

                    # distance = compute_distance_with_timeout(
                    #     graph1, graph2, self.get_lam(), self.distance_func, self.dist_comp_timeout
                    # )
                    if distance == -1:
                        distance = 0
                        patience -= 1
                    if patience == 0:
                        logging.warning(f'out of patience for graph {i}')
                        break

                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # Since the distance matrix is symmetric

            data_list.clear()
            del data_list
            gc.collect()
            return distance_matrix

        x = F.relu(self.lin1(readouts))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


########################################## Utilities ##########

def split_pooled_batch(x, edge_index, batch) -> List[Data]:
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


