import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from QCCL.transformations import RandomCompositeTransformation

from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, qc_graphs, pre_paired=False, num_transformations=2):
        """
        graphs: List of QuantumCircuitGraph objects or list of lists of paired QuantumCircuitGraph objects.
        transform: Transformations to apply for augmentation (only used if pre_paired is False).
        pre_paired: Boolean flag to indicate if the dataset contains pre-paired graphs.
        """
        self.quantum_circuit_graphs = qc_graphs
        self.pre_paired = pre_paired
        self.num_transformations = num_transformations
    
    def __len__(self):
        return len(self.quantum_circuit_graphs)
    
    def __getitem__(self, idx, return_graph=True, return_circuit=False):
        if self.pre_paired:
            idx1, idx2 = np.random.choice(range(len(self.quantum_circuit_graphs[idx])), 2, replace=False)
            qcg1, qcg2 = self.quantum_circuit_graphs[idx][idx1], self.quantum_circuit_graphs[idx][idx2]
        else:
            qcg1 = self.quantum_circuit_graphs[idx]

            random_composite_transform = RandomCompositeTransformation(qcg1, num_transformations=self.num_transformations)
            qcg2 =  random_composite_transform.apply()
            # print(f"Applied {num_transformations} random transformations to sample, returning pair of graphs.")

        if return_circuit and return_graph:
            return (from_nx_to_geometric(qcg1.graph), qcg1.quantum_circuit), (from_nx_to_geometric(qcg2.graph), qcg2.quantum_circuit)
        
        if return_circuit:
            return qcg1.quantum_circuit, qcg2.quantum_circuit
     
        return from_nx_to_geometric(qcg1.graph), from_nx_to_geometric(qcg2.graph)
    

    def get_equivalence_class(self, idx):
        clss = self.quantum_circuit_graphs[idx]
        return [from_nx_to_geometric(qcg.graph) for qcg in clss]



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
