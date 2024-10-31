import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Data')))

from Data.QuantumCircuitGraph import QuantumCircuitGraph


def load_data(data_dir='../dataset/raw/', file_name='handcrafted_dataset.pkl', subset=None, split_circuit_graph=False):
    file_path = os.path.join(data_dir, file_name)
    graphs, qcs = [], []
    qcgs = []
    
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
        if sample.__class__.__name__ == "QuantumCircuitGraph":  # if a QuantumCircuitGraph object
            if split_circuit_graph:
                qc = sample.quantum_circuit
                g = sample.graph
                qcs.append(qc)
                graphs.append(g)
            else:
                qcgs.append(sample)
        elif all(s.__class__.__name__ == "QuantumCircuitGraph" for s in sample):  # if a list of QuantumCircuitGraph objects
            if split_circuit_graph:
                qc = [s.quantum_circuit for s in sample]
                g = [s.graph for s in sample]
                qcs.append(qc)
                graphs.append(g)
            else:
                qcgs.append([s for s in sample])

    if split_circuit_graph:
        print(f"Loaded {len(graphs)} samples and {len(qcs)} quantum circuits.")
        return graphs, qcs
    else:
        print(f"Loaded {len(qcgs)} QuantumCircuitGraph objects.")
        return qcgs

def collect_from_dict(dictionary):
    collected_items = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            collected_items.extend(collect_from_dict(v))
        elif isinstance(v, list):
            if all(isinstance(item, list) for item in v):  # if list of lists
                collected_items.extend(v)
                print(f"\tCollected {len(v)} samples from {k}.")
            elif all(isinstance(item, tuple) for item in v):  # if list of tuples
                collected_items.append(v)
                print(f"\tCollected 1 sample from {k}.")
                
    return collected_items
