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
    def __init__(self, in_channels, out_channels, pooling_strategy='global_avg', num_layers=5, add_global_node=False):
        super(GCNFeatureExtractor, self).__init__(in_channels, out_channels, pooling_strategy, num_layers)
        self.add_global_node = add_global_node  # Option to add a global node
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
        
        num_nodes = x.size(0)

        # TODO: for now the global node is added having features equal to the mean of all node features
        # and no particular edge type is implemented to differentiate it from the other nodes.
        # This might be improved in the future, by allowing the model to recognize the global node
        # and possibly learn to use it more effectively.

        # Add a global node if specified
        if self.add_global_node:
            global_node_feature = x.mean(dim=0, keepdim=True)  # Use mean of all node features for global node
            x = torch.cat([x, global_node_feature], dim=0)  # Append global node feature to x

            # Create edges for global node
            node_to_global = torch.stack([torch.arange(num_nodes), torch.full((num_nodes,), num_nodes)], dim=0)
            global_to_node = torch.stack([torch.full((num_nodes,), num_nodes), torch.arange(num_nodes)], dim=0)
            global_edges = torch.cat([node_to_global, global_to_node], dim=1)

            # Update edge_index to include global edges
            edge_index = torch.cat([edge_index, global_edges], dim=1)

            # Extend batch tensor for the global node
            batch = torch.cat([batch, batch.new_zeros(1)])

        # Apply GCN layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # Apply pooling
        x = self.pooling_layer(x, batch, edge_index)
        return x