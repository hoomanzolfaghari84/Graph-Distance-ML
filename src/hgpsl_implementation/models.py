import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GCNConv
from layers import HGPSLPool  # Import updated custom layers


class ModernizedHGPSL(torch.nn.Module):
    def __init__(self, num_features, num_classes, nhid=128, lamb=1.0, structure_learning=True,
                 sparse_attention=True, sample_neighbor=True, dropout_ratio=0.0, pooling_ratio=0.5):
        super(ModernizedHGPSL, self).__init__()

        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.sample = sample_neighbor
        self.sparse = sparse_attention
        self.sl = structure_learning
        self.lamb = lamb

        # Define GCN layers using updated GCN (inherits from GCNConv)
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

        # Dense layers
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
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

        # Combine pooled features
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        # Dense layers with dropout
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)  # Raw logits

        # Apply log_softmax for classification
        return F.log_softmax(x, dim=-1)