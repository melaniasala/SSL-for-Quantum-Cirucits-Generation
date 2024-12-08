import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Data')))


def load_data(data_dir='../dataset/raw/', file_name=None, subset=None, split_circuit_graph=False):    
    graphs, qcs = [], []
    qcgs = []
    
    if file_name is None and os.path.isdir(data_dir):  # Handle folder case
        dataset = []
        for file in os.listdir(data_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(data_dir, file)
                print(f"Loading {file} from {file_path}...")
                with open(file_path, 'rb') as f:
                    file_dataset = pickle.load(f)
                if file == 'handcrafted_dataset.pkl' and subset is not None:
                    file_dataset = file_dataset[subset]
                    print(f"Loaded {len(file_dataset)} elements from subset {subset} in {file}.")
                file_dataset = collect_from_dict(file_dataset['dataset']) if file == 'handcrafted_dataset.pkl' else file_dataset['dataset']
                dataset.extend(file_dataset)
        print(f"Loaded {len(dataset)} total elements from folder {data_dir}.")
        print(dataset)
    else:  # Handle single file case
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        if file_name == 'handcrafted_dataset.pkl' and subset is not None:
            dataset = dataset[subset]
            print(f"Loaded {len(dataset)} elements from subset {subset}:")
        dataset = collect_from_dict(dataset) if file_name == 'handcrafted_dataset.pkl' else dataset

    # Process the dataset
    for sample in dataset:
        if sample.__class__.__name__ == "QuantumCircuitGraph":  # Single QuantumCircuitGraph object
            if split_circuit_graph:
                qc = sample.quantum_circuit
                g = sample.graph
                qcs.append(qc)
                graphs.append(g)
            else:
                qcgs.append(sample)
        elif all(s.__class__.__name__ == "QuantumCircuitGraph" for s in sample):  # List of QuantumCircuitGraph objects
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
