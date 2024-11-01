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
    def __init__(self, in_channels, out_channels, pooling_strategy='global_avg', num_layers=5):
        super(GNNFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling_strategy = pooling_strategy
        self.num_layers = num_layers

        if pooling_strategy not in pooling_strategies.keys():
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")
        
        self.pooling_layer = pooling_strategies[pooling_strategy]

    # forward pass
    def forward(self, sample):
        pass


class GCNFeatureExtractor(GNNFeatureExtractor):
    # N.B. GCN has issues with single-node-graphs
    def __init__(self, in_channels, out_channels, pooling_strategy='global_avg', num_layers=5): 
        super(GCNFeatureExtractor, self).__init__(in_channels, out_channels, pooling_strategy, num_layers)
        self.conv_layers = nn.ModuleList()
        max_exp = num_layers + 1
        if num_layers < 2:
            raise ValueError("Number of layers should be at least 2 (1 hidden layer)")
        for i in range(num_layers-1):
            if i == 0:
                self.conv_layers.append(gnn.GCNConv(in_channels, 2**max_exp))
            else:
                self.conv_layers.append(gnn.GCNConv(2**(max_exp-i+1), 2**(max_exp-i)))
                # print(f"Added GCNConv layer with input size {2**(max_exp-i+1)} and output size {2**(max_exp-i)}")
        self.conv_layers.append(gnn.GCNConv(2**(max_exp-num_layers+2), out_channels))

    def forward(self, sample):
        x, edge_index, batch = sample.x, sample.edge_index, sample.batch
        if x is None:  # Check if x is None
            raise ValueError("x should not be None")
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        x = self.pooling_layer(x, batch, edge_index)
        return x