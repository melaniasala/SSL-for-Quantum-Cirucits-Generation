import torch
from torch_geometric.utils import degree
import torch_geometric.nn as gnn


def global_mean_pool(x, batch, edge_index):
    return gnn.global_mean_pool(x, batch)


def global_max_pool(x, batch, edge_index):
    return gnn.global_max_pool(x, batch)
    

def extract_last_nodes(x, batch, row, col):
    """
    Extract the last nodes of each graph in a batch.
    """
    out_degree = degree(row, x.size(0), dtype=torch.long)
    in_degree = degree(col, x.size(0), dtype=torch.long)
    
    # A node is a "last node" if:
    # - it has no outgoing edges (out_degree == 0)
    # - it has both an outgoing and incoming edge to the same node 
    # Both conditions can be combined into a single condition: in_degree == out_degree+1
    last_nodes_mask = (in_degree == out_degree+1) | (out_degree == 0) 
    x_last = x[last_nodes_mask]
    batch_last = batch[last_nodes_mask] if batch is not None else None
    return x_last, batch_last


def last_nodes_add_pool(x, batch, edge_index):
    row, col = edge_index
    x_last, batch_last = extract_last_nodes(x, batch, row, col)
    
    return gnn.global_add_pool(x_last, batch_last)


def last_nodes_max_pool(x, batch, edge_index):
    row, col = edge_index
    x_last, batch_last = extract_last_nodes(x, batch, row, col)

    return gnn.global_max_pool(x_last, batch_last)


def last_nodes_mean_pool(x, batch, edge_index):
    row, col = edge_index
    x_last, batch_last = extract_last_nodes(x, batch, row, col)
    
    return gnn.global_mean_pool(x_last, batch_last)
