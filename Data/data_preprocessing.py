import random
import torch
import networkx as nx
import qiskit.dagcircuit.dagnode as dagnode
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from . import QuantumCircuitGraph


GATE_TYPE_MAP = {
    'cx': 0, 
    'h': 1, 
    'rx': 2, 
    'ry': 3, 
    'rz': 4,
    'x': 5
    # Add here all possible gate types
} # map gate type to index

INV_GATE_TYPE_MAP = {v: k for k, v in GATE_TYPE_MAP.items()} # map index to gate type


def encode_sequence(graph, sequence_length=None, end_index=None, use_padding=True, padding_value=0.0):
    """
    Encode a sequence of nodes into a tensor representation: each node is represented by its feature vector.

    :param graph: QuantumCircuitGraph object
    :param sequence_length: Maximum size of the sequence to encode (default is the full sequence)
    :param end_index: Index of the last node in the sequence (exclusive)
    :param use_padding: Whether to pad the sequence to the specified length
    :param padding_value: Value to use for padding
    :return: Tensor representation of the sequence
    """
    # Ensure graph is a QuantumCircuitGraph object
    if not isinstance(graph, QuantumCircuitGraph.QuantumCircuitGraph):
        raise ValueError("graph parameter must be a QuantumCircuitGraph object")

    total_nodes = graph.node_feature_matrix.shape[0]
    if sequence_length is None:
        sequence_length = total_nodes

    if end_index is None:
        end_index = len(graph.node_ids)  # end node index (exclusive)
    start_index = max(end_index - sequence_length, 0)  # start node index (inclusive)
    actual_length = end_index - start_index

    # Initialize the encoded sequence with padding if needed
    encoded_sequence = []
    if use_padding and actual_length < sequence_length:
        padding_tensor = torch.ones((1, graph.n_node_features)) * padding_value
        encoded_sequence.extend([padding_tensor] * (sequence_length - actual_length))

    for node_id in graph.to_sequence()[start_index:end_index]:
        node_idx = graph.node_mapping[node_id]
        node_feature = graph.node_feature_matrix[node_idx]
        encoded_sequence.append(node_feature)

    # Convert the encoded sequence to a PyTorch tensor
    encoded_sequence = torch.stack(encoded_sequence) 
    return encoded_sequence


def decode_sequence(encoded_sequence, graph):
    """
    Decode a tensor representation of a sequence back into a sequence of nodes.

    :param encoded_sequence: Tensor representation of the sequence
    :param graph: QuantumCircuitGraph object
    :return: List of dictionaries containing information (gate type, control/target) about each node in the sequence
    """
    # Ensure graph is a QuantumCircuitGraph object
    if not isinstance(graph, QuantumCircuitGraph.QuantumCircuitGraph):
        raise ValueError("graph parameter must be a QuantumCircuitGraph object")

    # Ensure encoded_sequence is a PyTorch tensor
    if not isinstance(encoded_sequence, torch.Tensor):
        raise ValueError("encoded_sequence parameter must be a PyTorch tensor")

    # Decode the sequence
    decoded_sequence = []
    for node_features in encoded_sequence:
        decoded_node = {}
        node_features = node_features.tolist()
        one_hot_gate_type = list(node_features[:len(GATE_TYPE_MAP)])
        one_hot_control_target = list(node_features[-2:])

        gate_type = INV_GATE_TYPE_MAP[one_hot_gate_type.index(1.)]
        decoded_node['gate_type'] = gate_type
        if gate_type == 'cx':
            control_target = 'control' if one_hot_control_target.index(1.) == 0 else 'target'
            decoded_node['ctrl_trgt'] = control_target
        
        decoded_sequence.append(decoded_node)

    return decoded_sequence


def extract_circuit_gates(circuit):
    """
    Extracts gates from a QuantumCircuit into a structured format (list of dictionaries).

    :param circuit: QuantumCircuit object
    :return: Dictionary containing number of qubits and gates information
    """
    circuit_info = {'n_qubits': circuit.num_qubits, 'gates': []}

    for instruction in circuit.data:
        if instruction.operation.name != 'barrier':
            gate_info = {
                'gate_type': instruction.operation.name,
                'qubits': [qubit._index for qubit in instruction.qubits],
                'params': instruction.operation.params
            }
            circuit_info['gates'].append(gate_info)

    return circuit_info



# def add_gate_to_graph(self, gate_type, qubits):
#     """
#     Add a gate to the quantum circuit graph.
#     :param gate_type: type of the quantum gate
#     :param qubits: qubits on which the gate acts (list of qubit indices)
#     """
    
#     # If multiqubit gate
#     n_qubits = len(qubits)
#     if n_qubits > 1:
#         for curr in range(n_qubits):
#             for prev in range(curr+1, n_qubits):
#                 node_id = f"{gate_type}_{qubits[curr]}_" # missing unique identifier
#                 if curr == 0:
#                     node_id += f"_control"




def build_graph_from_circuit(circuit):
    """
    Constructs a NetworkX graph representation from the QuantumCircuit.

        :param circuit: QuantumCircuit object
        :return: Tuple containing the graph, last nodes, node positions, and node list.
    """
    # Create an empty graph
    graph = nx.Graph()

    # Extract DAGCircuit from the QuantumCircuit, remove barriers, and get the layers
    remove_barriers = RemoveBarriers()
    
    dag = circuit_to_dag(remove_barriers(circuit))
    layers = list(dag.multigraph_layers())[1:-1] # exclude the first layer (input layer) and the last layer (output layer)

    last_nodes = {} # dictionary to store the last node for each qubit
    node_positions = {} # dictionary to store the position of each node in the graph (for visualization)
    node_list = [] # list to store the order of the nodes in the graph

    # Iterate over the layers and gates in the circuit
    for l, layer in enumerate(layers):
        gates_in_layer = []
        for gate in layer:
            # if gate is a DAGOpNode, process it. Else move to the next gate
            if type(gate) == dagnode.DAGOpNode:
                gates_in_layer.extend(process_gate(graph, gate, l, last_nodes, node_positions))
            else:
                continue

        # Order gates in the layer according to the qubit index
        gates_in_layer.sort(key=lambda x: graph.nodes[x]['qubit'])

        # Add the ordered list of gates in the layer to the node_list
        node_list.extend(gates_in_layer)

    # N.B. Nodes are added in the order given by layers and qubit index: when you call graph.nodes you 
    # will get the nodes in this same order. If you want to get the nodes in BFS order, you need to call
    # nx.bfs_nodes(graph, start_node) where start_node is the node from which you want to start the traversal

    # Notice that QuantumCircuitGraph.nodes will retrieve the list of nodes identifiers in the order given by the
    # layers and qubit index, while DiGraph.nodes will retrieve a dictionary with the nodes as keys and their
    # attributes as values. The order of the nodes in the dictionary should be the same as the order in which they
    # were added to the graph.

    return graph, node_list, last_nodes, node_positions 


def process_gate(graph, gate, layer_idx, last_nodes, node_positions):
        """
        Processes an individual gate and updates the graph accordingly.

        :param graph: The current graph
        :param gate: The gate being processed (DAGNode object)
        :param layer_idx: Current layer index
        :param last_nodes: Dictionary of last nodes for each qubit
        :param node_positions: Dictionary of node positions
        :return: List of node IDs created for this gate
        """
        gate_as_nodes = []

        if gate.name == 'cx':
            control_qubit = gate.qargs[0]._index
            target_qubit = gate.qargs[1]._index

            # Create unique node_ids for the control and target qubits
            node_id_control = f"{gate.name}_{control_qubit}_control_{gate._node_id}"
            node_id_target = f"{gate.name}_{target_qubit}_target_{gate._node_id}"

            # Add nodes to the graph
            graph.add_node(node_id_control, type=gate.name, qubit=control_qubit, params=gate.op.params, ctrl_trgt='c')
            graph.add_node(node_id_target, type=gate.name, qubit=target_qubit, params=gate.op.params, ctrl_trgt='t')

            # Add two edges (each pointing in opposite direction) between control and target
            graph.add_edge(node_id_control, node_id_target, type='cx')
            graph.add_edge(node_id_target, node_id_control, type='cx')

            # Connect control and target nodes to the last nodes for each qubit
            for qubit, node_id in zip([control_qubit, target_qubit], [node_id_control, node_id_target]):
                if qubit in last_nodes:
                    graph.add_edge(last_nodes[qubit], node_id, type='qubit', create_using=nx.DiGraph()) # edge going from last node to current node
                last_nodes[qubit] = node_id

                node_positions[node_id] = (layer_idx, -qubit) # custom position for each node
                gate_as_nodes.append(node_id)

        else:
            qubit = gate.qargs[0]._index

            # Create a unique node_id for the gate
            node_id = f"{gate.name}_{qubit}_{gate._node_id}"

            # Add the node to the graph
            graph.add_node(node_id, type=gate.name, qubit=qubit, params=gate.op.params)

            # Connect the node to the last node for the qubit
            if qubit in last_nodes:
                graph.add_edge(last_nodes[qubit], node_id, type='qubit', create_using=nx.DiGraph())
            last_nodes[qubit] = node_id

            node_positions[node_id] = (layer_idx, -qubit) # custom position for the node
            gate_as_nodes.append(node_id)

        return gate_as_nodes


# def build_pyg_graph_from_circuit(circuit, include_params=False):
#     """
#     Build a PyG graph representation of a quantum circuit.
#     To initialize a PyG Data object, we need to provide node features and edge index as tensors.
#     :param circuit: quantum circuit
#     :return: PyG Data object representing the circuit
#     """
#     node_features = []
#     edge_index = []
#     last_nodes = {}

#     # Extract circuit gates
#     circuit_gates = get_circuit_gates(circuit)

#     current_node_idx = 0

#     for i, gate in enumerate(circuit_gates['gates']):
#         gate_type = gate['gate_type']
        
#         if gate_type == 'cx':
#             control_qubit = gate['qubits'][0]
#             target_qubit = gate['qubits'][1]
            
#             control_node_idx = current_node_idx
#             target_node_idx = current_node_idx + 1

#             # Add nodes with their respective types to node feature matrix
#             node_features.append(create_node_feature(gate_type, ctrl_trgt= 'control', include_params= include_params, params= gate['params']))
#             node_features.append(create_node_feature(gate_type, ctrl_trgt= 'target', include_params= include_params, params= gate['params']))
#             current_node_idx += 2

#             # Add an edge between control and target
#             edge_index.append([control_node_idx, target_node_idx])

#             # Connect control and target nodes to the last nodes for each qubit
#             for qubit, node_idx in zip([control_qubit, target_qubit], [control_node_idx, target_node_idx]):
#                 if qubit in last_nodes:
#                     edge_index.append([last_nodes[qubit], node_idx])
#                 last_nodes[qubit] = node_idx

#         else:
#             node_idx = current_node_idx
#             current_node_idx += 1

#             # Add a node for the gate with its type and parameters as features
#             node_features.append(create_node_feature(gate_type, include_params= include_params, params= gate['params']))
            
#             for qubit in gate['qubits']:
#                 if qubit in last_nodes:
#                     edge_index.append([last_nodes[qubit], node_idx])
#                 last_nodes[qubit] = node_idx

#     # Convert node features and edge index to tensors
#     node_features = torch.tensor(node_features, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 

#     # Create PyG Data object
#     data = tg.data.Data(x=node_features, edge_index=edge_index)

#     return data


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
