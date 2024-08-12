import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from Data.transforms import perform_random_transform

class GraphDataset(Dataset):
    def __init__(self, graphs, transforms=None, pre_paired=False):
        """
        graphs: List of graphs or list of lists of paired graphs.
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
        return graph1, graph2
