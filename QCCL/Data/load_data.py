import os
import pickle
import networkx as nx
import sys, os


def load_graphs(data_dir='../Data/raw/', file_name='handcrafted_dataset.pkl', subset=None):
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
        if isinstance(sample, tuple): # if a tuple (qc, graph)
            qc, g = sample
            graphs.append(g)
            qcs.append(qc)
        elif all(isinstance(graph, tuple) for graph in sample):# if a list of tuples (qc, graph) 
            qc, g = zip(*sample)
            qcs.append(list(qc))
            graphs.append(list(g))

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
                print(f"\tCollected {len(v)} items from {k}.")
            elif all(isinstance(item, tuple) for item in v):  # if list of tuples
                graphs.append(v)
                print(f"\tCollected 1 sample from {k}.")
                
    return graphs

# def collect_from_dict(dictionary):
#     graphs = []
#     for k, v in dictionary.items():
#         if isinstance(v, dict):
#             graphs.extend(collect_from_dict(v))
#         elif isinstance(v, list):
#             g, n = collect_from_list(v, k)
#             graphs.extend(g)
#             print(f"Collected {n} graphs from {k}.")
#         else:
#             graphs.extend(v)
#             print(f"Collected {len(v)} graphs from {k}.")
#     return graphs


# def collect_from_list(lst, k):
#     graphs = []
#     n = 0
#     for item in lst:
#         if isinstance(item, list):
#             g, n_new = collect_from_list(item, k)
#             graphs.extend(g)
#             n += n_new
#         else:
#             n += 1
#             graphs.append(item)
#     return graphs, n
