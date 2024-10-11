import os
import pickle

from ...Data.QuantumCircuitGraph import QuantumCircuitGraph


def load_data(data_dir='../Data/raw/', file_name='handcrafted_dataset.pkl', subset=None):
    file_path = os.path.join(data_dir, file_name)
    graphs, qcs = [], []
    
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    if file_name == 'handcrafted_dataset.pkl':
        if subset is not None:
            dataset = dataset[subset]
            print(f"Loaded {len(dataset)} elements from subset {subset}:")
        # collect all graphs in dataset, which are stored in a nested dictionary
        dataset = collect_from_dict(dataset) 

    # extract graphs and qcs separately, if needed
    for sample in dataset:
        if isinstance(sample, QuantumCircuitGraph):  # if a QuantumCircuitGraph object
            qc = sample.quantum_circuit
            g = sample.graph
            graphs.append(g)
            qcs.append(qc)
        elif all(isinstance(s, QuantumCircuitGraph) for s in sample):  # if a list of QuantumCircuitGraph objects
            qc = [s.quantum_circuit for s in sample]
            g = [s.graph for s in sample]
            qcs.append(qc)
            graphs.append(g)

    print(f"Loaded {len(graphs)} samples and {len(qcs)} quantum circuits from subset.")
    
    return graphs, qcs

def collect_from_dict(dictionary):
    graphs = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            graphs.extend(collect_from_dict(v))
        elif isinstance(v, list):
            if all(isinstance(item, list) for item in v):  # if list of lists
                graphs.extend(v)
                print(f"\tCollected {len(v)} samples from {k}.")
            elif all(isinstance(item, tuple) for item in v):  # if list of tuples
                graphs.append(v)
                print(f"\tCollected 1 sample from {k}.")
                
    return graphs
