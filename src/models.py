import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
# import asyncio
# import gc
# from memory_profiler import profile

import torch
# from torch.profiler import profile as tprofile
import torch.nn.functional as F


from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx, subgraph

import networkx as nx
from hgpsl_implementation.layers import HGPSLPool
from mcs import MCSDistanceSparse, batch_to_datalist, MCSDistance



class PoolingModule(torch.nn.Module):
    def __init__(self, num_features, num_classes, nhid=128, lamb=1.0, structure_learning=True,
                 sparse_attention=True, sample_neighbor=True, dropout_ratio=0.0, pooling_ratio=0.5):
        super(PoolingModule, self).__init__()

        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.sample = sample_neighbor
        self.sparse = sparse_attention
        self.sl = structure_learning
        self.lamb = lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        # Define pooling layers with updated HGPSLPool
        self.pool1 = HGPSLPool(
            self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb
        )
        self.pool2 = HGPSLPool(
            self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb
        )


    def forward(self, data : Batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        self.forward(x, edge_index, batch)

    def forward(self, x : torch.Tensor, edge_index : torch.Tensor, batch : torch.Tensor):
        edge_attr = None  # Edge attributes remain optional

        # First GCN + Pooling
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Second GCN + Pooling
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Third GCN
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        readouts = F.relu(x1) + F.relu(x2) + F.relu(x3)

        return x, edge_index, edge_attr, batch, readouts


class MCSModule(torch.nn.Module):
    def __init__(self, dropout_ratio = 0.0, distance_func: MCSDistance = MCSDistanceSparse(), clustering=None, lam_init=2, dist_comp_timeout=45):

        super(MCSModule, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.dist_comp_timeout = dist_comp_timeout
        self.clustering = clustering

        self.__lam__ = torch.nn.Parameter(torch.tensor(lam_init, dtype=torch.float32))
        self.distance_func = distance_func
        self.clustering = False

    def get_lam(
            self):  # constraint lambda parameter to satisfy theoretical constraints, while being differentiable in R.
        return F.softplus(self.__lam__)

    def forward(self, x, edge_index, batch, edge_attr):

        data_list = batch_to_datalist(x, edge_index, batch)

        if self.clustering:
            _, clusters = self.clustering.predict(data_list)
        else: clusters = None

        num_graphs = len(data_list)

        # Initialize an empty distance matrix
        distance_matrix = torch.zeros((num_graphs, num_graphs), device=x.device)

        # Compute pairwise distances using the custom distance function
        for i in range(num_graphs):
            # patience = 3
            for j in range(i + 1, num_graphs):

                if self.clustering and clusters[i] != clusters[j]:
                    distance_matrix[i,j] = 0
                    continue

                graph1 = data_list[i]
                graph2 = data_list[j]

                distance = self.distance_func(graph1[0], graph1[1], graph2[0], graph2[1], self.get_lam())

                # if distance == -1:
                #     distance = 0
                #     patience -= 1
                # if patience == 0:
                #     logging.warning(f'out of patience for graph {i}')
                #     break

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Since the distance matrix is symmetric

        data_list.clear()
        del data_list
        gc.collect()

        return distance_matrix



class MCSFullModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim = 128, dropout_ratio = 0.0, dist_comp_timeout=45):
        super(MCSFullModel, self).__init__()
        self.input_dim = input_dim
        self.nhid = hidden_dim
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.dist_comp_timeout = dist_comp_timeout

        self.pooling = PoolingModule(input_dim, num_classes)
        self.mcs_module = MCSModule()

        # Dense layers
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.pre_train = True

    def freeze_pooling(self, set_value):
        self.pooling.requires_grad_(not set_value)

    def forward(self, data):
        return self.forward(data.x, data.edge_index, data.batch)

    def forward(self, x, edge_index, batch):
        x, edge_index, edge_attr, batch, readouts = self.pooling(x, edge_index, batch)

        if self.pre_train:
            distances = self.mcs_module(x , edge_index, batch)

            return  distances

        else:
            # Dense layers with dropout
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = self.lin3(x)  # Raw logits

            # Apply log_softmax for classification
            return F.log_softmax(x, dim=-1)


