import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Data')))


def load_file(file_path):
    """Loads data from a pickle file and optionally extracts a subset."""
    start_time = time.time()
    print(f"Loading file {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        if 'dataset' in data:
            data = data['dataset']
        else:
            raise ValueError("Unknown data format. Expected a dictionary with a 'dataset' key.")
    elif not isinstance(data, list):
        raise ValueError("Unknown data format. Expected a dictionary with a 'dataset' key or a list of samples.")

    print(f"Loaded {len(data)} elements from file {file_path} in {time.time() - start_time:.2f} seconds.\n")
    return data


def load_handcrafted_data(file_path, subset=None):
    """Loads data specifically for handcrafted and optionally extracts a subset."""
    start_time = time.time()
    print(f"Loading handcrafted dataset from file {file_path}...")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if subset is not None:
        if isinstance(data, dict) and subset in data:
            data = data[subset]
            print(f"Loaded subset '{subset}' containing {len(data)} elements.")
        else:
            raise ValueError(f"Subset '{subset}' not found in the dataset.")
    
    data = collect_from_dict(data)
    print(f"Loaded {len(data)} elements from handcrafted dataset in {time.time() - start_time:.2f} seconds.\n")
    return data


def process_samples(dataset, split_circuit_graph=False):
    """Processes QuantumCircuitGraph samples, optionally splitting them into components."""
    graphs, qcs, qcgs = [], [], []
    
    for sample in dataset:
        if sample.__class__.__name__ == "QuantumCircuitGraph": # Single QuantumCircuitGraph object
            if split_circuit_graph:
                qcs.append(sample.quantum_circuit)
                graphs.append(sample.graph)
            else:
                qcgs.append(sample)
        elif all(s.__class__.__name__ == "QuantumCircuitGraph" for s in sample): # List of QuantumCircuitGraph objects 
            if split_circuit_graph:
                qcs.append([s.quantum_circuit for s in sample])
                graphs.append([s.graph for s in sample])
            else:
                qcgs.append(sample)
        else:
            raise ValueError("Unknown data format. Expected a QuantumCircuitGraph object or a list of QuantumCircuitGraph objects.")
    return graphs, qcs, qcgs
    
    

def load_data(data_dir='../dataset/raw/', file_name=None, split_circuit_graph=False, handcrafted=False, subset=None):
    """Main function to load datasets from files or directories, excluding handcrafted datasets."""
    start_time = time.time()

    if handcrafted or file_name == 'handcrafted_dataset.pkl': # Load handcrafted dataset
        dataset = load_handcrafted_data(file_path=os.path.join(data_dir, file_name), subset=subset)
    elif file_name is None and os.path.isdir(data_dir):  # Load all files in the directory
        dataset = []
        for file in os.listdir(data_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(data_dir, file)
                file_dataset = load_file(file_path)
                dataset.extend(file_dataset)
        print(f"Total of {len(dataset)} elements loaded from folder {file_path} in {time.time() - start_time:.2f} seconds.")
    else:  # Load a single file
        file_path = os.path.join(data_dir, file_name)
        dataset = load_file(file_path)
        print(f"Total of {len(dataset)} elements loaded from file {file_path} in {time.time() - start_time:.2f} seconds.")

    graphs, qcs, qcgs = process_samples(dataset, split_circuit_graph)

    if split_circuit_graph:
        print(f"Final counts: {len(graphs)} graphs, {len(qcs)} quantum circuits.")
        return graphs, qcs
    else:
        print(f"Final count: {len(qcgs)} QuantumCircuitGraph objects.")
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
