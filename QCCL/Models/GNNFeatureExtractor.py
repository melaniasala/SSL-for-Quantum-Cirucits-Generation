import torch
from .pooling import last_nodes_mean_pool, last_nodes_max_pool, global_mean_pool, global_max_pool
import torch.nn as nn
import torch_geometric.nn as gnn


pooling_strategies = {
            'global_avg': global_mean_pool,
            'global_max': global_max_pool,
            'last_avg': last_nodes_mean_pool,
            'last_max': last_nodes_max_pool
        }


class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_strategy='global_avg'):
        super(GNNFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling_strategy = pooling_strategy

        if pooling_strategy not in pooling_strategies.keys():
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")
        
        self.pooling_layer = pooling_strategies[pooling_strategy]

    # forward pass
    def forward(self, x, edge_index, batch):
        pass


class GCNFeatureExtractor(GNNFeatureExtractor):
    def __init__(self, in_channels, out_channels, pooling_strategy='global_avg'):
        super(GCNFeatureExtractor, self).__init__(in_channels, out_channels, pooling_strategy)
        self.conv1 = gnn.GCNConv(in_channels, 2 * out_channels)
        self.conv2 = gnn.GCNConv(2 * out_channels, out_channels)


    def forward(self, x, edge_index, batch):
        if x is None:  # Check if x is None
            raise ValueError("x should not be None")
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.pooling_layer(x, batch, edge_index)
        return x
