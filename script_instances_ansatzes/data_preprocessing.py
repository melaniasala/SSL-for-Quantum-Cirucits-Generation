import qaoa, problems
import torch_geometric as tg
import networkx as nx
import torch
import random
import networkx as nx

class QuantumCircuitGraph:
    """
    Class to represent a quantum circuit as a graph.
    The graph is built using the NetworkX library.
    """

    def __init__(self, circuit=None):
        """
        Initialize the graph representation of a quantum circuit.
        If a circuit is provided, build the graph from the circuit.
        """
        self.graph = nx.Graph()
        self.node_positions = {}
        self.node_mapping = {}
        self.node_features = None
        self.as_sequence = None

        if circuit is not None:
            self.build_graph_from_circuit(circuit)

    def build_graph_from_circuit(self, circuit):
        """
        Build the graph representation of a quantum circuit, given a QuantumCircuit object.
        """
        self.graph, self.node_positions = build_nx_graph_from_circuit(circuit)
        self.node_mapping = {node_id: i for i, node_id in enumerate(self.graph.nodes())}
        self.build_node_features()
        self.as_sequence = self.to_sequence()
    


    def build_node_features(self, include_params=False):
        """
        Build the node features matrix for the quantum circuit graph.
        The node features matrix is a tensor where each row represents the feature vector of a node in the graph.
        The feature vector consists of the one-hot encoded gate type and qubit type (control, target, or neither).
        """
        node_features = []
        for node_id in self.graph.nodes():
            gate_type = self.graph.nodes[node_id]['type']
            params = self.graph.nodes[node_id]['params']
            if 'control' in node_id:
                node_feature = create_node_feature(gate_type, ctrl_trgt='control', include_params=include_params, params=params)
            elif 'target' in node_id:
                node_feature = create_node_feature(gate_type, ctrl_trgt='target', include_params=include_params, params=params)
            else:
                node_feature = create_node_feature(gate_type, include_params=include_params, params=params)
            node_features.append(node_feature)

            self.node_mapping[node_id] = len(node_features) - 1
        self.node_features = torch.tensor(node_features, dtype=torch.float)

    # def encode_sequence(self, sequence):
    #     """
    #     Encode a sequence of nodes into a tensor representation.
    #     :param sequence: list of node_ids in the sequence
    #     :return: tensor representation of the sequence
    #     """
    #     encoded_sequence = []
    #     for node_id in sequence:
    #         node_idx = self.node_mapping[node_id]
    #         node_feature = self.node_features[node_idx]
    #         encoded_sequence.append(node_feature)
    #     encoded_sequence = torch.stack(encoded_sequence)
    #     return encoded_sequence

    def to_sequence(self, random_start=True):
        """
        Convert the quantum circuit graph to a sequence of nodes, using a Breadth-First Search (BFS) traversal.
        A random node is selected as the starting node.
        :return: list of nodes in the graph, in BFS order
        """
        start_node = random.choice(list(self.graph.nodes())) if random_start else list(self.graph.nodes())[0]
            
        bfs_edges = nx.bfs_edges(self.graph, start_node)
        bfs_nodes = [start_node]
        bfs_nodes_visited = {start_node}
        for u, v in bfs_edges:
            if v not in bfs_nodes_visited:
                bfs_nodes.append(v)
                bfs_nodes_visited.add(v)
        return bfs_nodes
    

def get_circuit_gates(circuit):
    """
    Extract gates from a QuantumCircuit and store them in a list of dictionaries.
    :param circuit: quantum circuit
    :return: list containing number of qubits and a list of dictionaries with gates information (gate type, qubits, parameters)
    """
    circuit_info = {}
    gates_list = []

    circuit_info['n_qubits'] = circuit.num_qubits
    
    for instruction in circuit.data:
        if instruction.operation.name != 'barrier':
            gate_dict = {
                'gate_type': instruction.operation.name,
                'qubits': [qubit._index for qubit in instruction.qubits],
                'params': instruction.operation.params
            }
            gates_list.append(gate_dict)
    
    circuit_info['gates'] = gates_list
    
    return circuit_info


def build_nx_graph_from_circuit(circuit):
    """
    Build a graph representation of a quantum circuit.
    :param circuit: QuantumCircuit object
    :return: networkx graph representing the circuit
    """
    # Create an empty graph
    graph = nx.Graph()

    # Dictionary to keep track of the last nodes for each qubit 
    # and custom position for each node
    last_nodes = {}
    node_positions = {}

    # Extract circuit gates
    circuit_gates = get_circuit_gates(circuit)

    # Iterate over the gates in the circuit
    for i, gate in enumerate(circuit_gates['gates']):
        type = gate['gate_type']
        
        # If CNOT gate
        if type == 'cx':
            control_qubit = gate['qubits'][0]
            target_qubit = gate['qubits'][1]
            
            # Create unique node_ids for the control and target qubits
            node_id_control = f"{type}_{control_qubit}_control_{i}"
            node_id_target = f"{type}_{target_qubit}_target_{i}"

            # Add nodes for the control and target qubits with their respective types
            graph.add_node(node_id_control, type=type, qubit=control_qubit, params=gate['params'])
            graph.add_node(node_id_target, type=type, qubit=target_qubit, params=gate['params'])

            # Add an edge between control and target
            graph.add_edge(node_id_control, node_id_target)

            # Connect control and target with the last nodes for each qubit
            for qubit in [control_qubit, target_qubit]:
                if qubit in last_nodes:
                    graph.add_edge(node_id_control if qubit == control_qubit else node_id_target, last_nodes[qubit])
                last_nodes[qubit] = node_id_control if qubit == control_qubit else node_id_target

                node_positions[node_id_control] = (i, -control_qubit) # custom position for each node
                node_positions[node_id_target] = (i, -target_qubit)

        # If not a CNOT gate
        else:
            # Create a unique node_id for the gate
            node_id = f"{type}_{gate['qubits'][0]}_{i}"

            # Add a node for the gate with its type and parameters
            graph.add_node(node_id, type=type, qubit=gate['qubits'][0], params=gate['params'])

            # Connect the gate node with the last nodes for each qubit
            for qubit in gate['qubits']:
                if qubit in last_nodes:
                    graph.add_edge(node_id, last_nodes[qubit])

                last_nodes[qubit] = node_id

                node_positions[node_id] = (i, -qubit) # custom position for each node

    return graph, node_positions


GATE_TYPE_MAP = {
    'cx': 0, 
    'h': 1, 
    'rx': 2, 
    'ry': 3, 
    'rz': 4
    # Add here all possible gate types
}


def create_node_feature(gate_type, ctrl_trgt=None, include_params=False, params=None):
    """
    Create a node feature vector for a quantum gate.
    The node feature vector consists of the one-hot encoded gate type and qubit type.
    :param gate_type: type of the quantum gate
    :param qubit_type: type of the qubit (control or target)
    :param params: parameters of the gate
    :return: node feature vector
    """
    qubit_type_map = {'control': 0, 'target': 1}
    feature_vct = [0] * len(GATE_TYPE_MAP) + [0] * len(qubit_type_map)

    gate_type_idx = GATE_TYPE_MAP[gate_type]
    feature_vct[gate_type_idx] = 1 # one-hot encoding of gate type
    
    if ctrl_trgt is not None:
        qubit_type_idx = qubit_type_map[ctrl_trgt]    
        feature_vct[len(GATE_TYPE_MAP) + qubit_type_idx] = 1 # one-hot encoding for control/target

    if include_params==True and params is not None:
        feature += params

    return feature_vct



def build_pyg_graph_from_circuit(circuit, include_params=False):
    """
    Build a PyG graph representation of a quantum circuit.
    To initialize a PyG Data object, we need to provide node features and edge index as tensors.
    :param circuit: quantum circuit
    :return: PyG Data object representing the circuit
    """
    node_features = []
    edge_index = []
    last_nodes = {}

    # Extract circuit gates
    circuit_gates = get_circuit_gates(circuit)

    current_node_idx = 0

    for i, gate in enumerate(circuit_gates['gates']):
        gate_type = gate['gate_type']
        
        if gate_type == 'cx':
            control_qubit = gate['qubits'][0]
            target_qubit = gate['qubits'][1]
            
            control_node_idx = current_node_idx
            target_node_idx = current_node_idx + 1

            # Add nodes with their respective types to node feature matrix
            node_features.append(create_node_feature(gate_type, ctrl_trgt= 'control', include_params= include_params, params= gate['params']))
            node_features.append(create_node_feature(gate_type, ctrl_trgt= 'target', include_params= include_params, params= gate['params']))
            current_node_idx += 2

            # Add an edge between control and target
            edge_index.append([control_node_idx, target_node_idx])

            # Connect control and target nodes to the last nodes for each qubit
            for qubit, node_idx in zip([control_qubit, target_qubit], [control_node_idx, target_node_idx]):
                if qubit in last_nodes:
                    edge_index.append([last_nodes[qubit], node_idx])
                last_nodes[qubit] = node_idx

        else:
            node_idx = current_node_idx
            current_node_idx += 1

            # Add a node for the gate with its type and parameters as features
            node_features.append(create_node_feature(gate_type, include_params= include_params, params= gate['params']))
            
            for qubit in gate['qubits']:
                if qubit in last_nodes:
                    edge_index.append([last_nodes[qubit], node_idx])
                last_nodes[qubit] = node_idx

    # Convert node features and edge index to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 

    # Create PyG Data object
    data = tg.data.Data(x=node_features, edge_index=edge_index)

    return data


def nx_graph_to_sequence(graph, random_start=True):
    """
    Convert a networkx graph to a sequence of nodes, using a Breadth-First Search (BFS) traversal.
    A random node is selected as the starting node.
    :param graph: networkx graph
    :return: list of nodes in the graph, in BFS order
    """
    # Check if the input graph is a networkx graph
    if not isinstance(graph, nx.Graph):
        raise ValueError('Input graph must be a NetworkX graph.')
    
    start_node = random.choice(list(graph.nodes())) if random_start else list(graph.nodes())[0]

    # Perform BFS traversal
    bfs_edges = nx.bfs_edges(graph, start_node) # edges in BFS order
    
    # Extract nodes in BFS order from the edges
    bfs_nodes = [start_node] # list to store nodes in BFS order
    bfs_nodes_visited = {start_node}
    for u, v in bfs_edges:
        if v not in bfs_nodes_visited:
            bfs_nodes.append(v)
            bfs_nodes_visited.add(v)
    
    return bfs_nodes



# given a sequence of nodes, I want to encode it into S^(pi)_i
# where S^(pi)_i is the i-th node in the sequence include node features for each of the nodes in the sequence
# to do this I need
# - the sequence, given by a succession of node_ids
# - the node_mapping, which maps the node_ids to the corresponding node index, so that I can retrieve the node features
# - the graph or node features matrix, to retrieve the node features from the node index
# the node feature matrix should be a tensor produced by the build_pyg_graph_from_circuit function
# --> take as input the pyg graph (or compute it inside the function) 
# also the node_mapping can be computed inside build_pyg_graph_from_circuit, given the graph
# better create a class that can manage all this? or just a function?



def encode_sequence(graph, index=0):
    """
    Encode a sequence of nodes into a tensor representation: each node is represented by its feature vector.
    :param graph: QuantumCircuitGraph object
    :param index: index of the sequence till which the sequence is to be encoded. Default is 0, which means the entire sequence is encoded.
    :return: tensor representation of the sequence
    """
    # Here no adjacency matrix is used, only the node features are used to encode the sequence. But the adjacency
    # matrix could still conatin useful information about the connections between the nodes. So it can be included
    # in the encoding, by two possible ways:
    # - by concatenating the adjacency matrix to the node features
    # - by 'filtering' the node features through the adjacency matrix, keeping only features of the nodes that are 
    #   connected directly to current one
    # - by using a GNN model that takes as input the adjacency matrix and the node features

    encoded_sequence = [] # list to store the encoded sequence
    end_node_idx = index if index > 0 else len(graph.as_sequence)

    for node_id in graph.as_sequence[:end_node_idx]:
        node_idx = graph.node_mapping[node_id]
        node_feature = graph.node_features[node_idx]
        encoded_sequence.append(node_feature)
    
    # Convert the encoded sequence to a PyTorch tensor
    encoded_sequence = torch.stack(encoded_sequence) 
    return encoded_sequence

