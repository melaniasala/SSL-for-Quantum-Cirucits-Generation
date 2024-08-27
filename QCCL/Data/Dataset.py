import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from Data.transforms import perform_random_transform
from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, graphs, transforms=None, pre_paired=False):
        """
        graphs: List of graphs or list of lists of paired NetworkX graphs.
        transform: Transformations to apply for augmentation (only used if pre_paired is False).
        pre_paired: Boolean flag to indicate if the dataset contains pre-paired graphs.
        """
        self.graphs = graphs
        self.transforms = transforms
        self.pre_paired = pre_paired
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.pre_paired:
            idx1, idx2 = np.random.choice(range(len(self.graphs[idx])), 2, replace=False)
            graph1, graph2 = self.graphs[idx][idx1], self.graphs[idx][idx2]
        else:
            graph1 = self.graphs[idx]
            graph2 =  perform_random_transform(graph1, self.transforms)
        return from_nx_to_geometric(graph1), from_nx_to_geometric(graph2)



def from_nx_to_geometric(graph):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    """
    x = get_attr_matrix(graph)  
    edge_index = get_edge_index(graph)  

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)

def get_attr_matrix(graph):
    nodes_list = list(graph.nodes)
    nodes_view = graph.nodes(data=True)
    return np.array([nodes_view[node]['feature_vector'] for node in nodes_list])

def get_edge_index(graph):
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    
    # Extract edges from NetworkX graph and convert them to the correct format (2, num_edges)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    return edge_index