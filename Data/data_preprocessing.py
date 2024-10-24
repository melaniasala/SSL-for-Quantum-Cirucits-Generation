import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import networkx as nx
from qiskit.circuit.gate import Gate



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
    # Ensure encoded_sequence is a PyTorch tensor
    if not isinstance(encoded_sequence, torch.Tensor):
        raise ValueError("encoded_sequence parameter must be a PyTorch tensor")

    # Decode the sequence
    decoded_sequence = []
    for node_features in encoded_sequence:
        decoded_node = {}
        node_features = node_features.tolist()
        one_hot_gate_type = list(node_features[:len(graph.GATE_TYPE_MAP)])
        one_hot_control_target = list(node_features[-2:])

        inv_gate_type_map = {v: k for k, v in graph.GATE_TYPE_MAP.items()}

        gate_type = inv_gate_type_map[one_hot_gate_type.index(1.)]
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




def build_graph_from_circuit(circuit, gate_type_map, include_params=False, include_id_gates=False, diff_cx=False, data=True):
    """
    Constructs a NetworkX graph representation from the QuantumCircuit.

        :param circuit: QuantumCircuit object
        :param gate_type_map: Dictionary mapping gate types to integers
        :param include_params: Whether to include gate parameters in the node feature vector
        :param include_id_gates: Whether to include identity gates in the graph
        :param diff_cx: Whether to differentiate between control and target qubits for CNOT gates
        :param data: Whether to return additional data (last nodes, node positions, node list)
        :return: Tuple containing the graph, last nodes, node positions, and node list.
    """
    # Create an empty graph
    graph = nx.DiGraph()

    last_nodes = {} # dictionary to store the last node for each qubit
    node_list = [] # list to store the order of the nodes in the graph

    for i, gate in enumerate(circuit.data):
    # Skip non-gate operations (e.g., barriers, measurements)
        if not isinstance(gate.operation, Gate):
            continue
        
        # Process the gate if it is a valid quantum gate
        gate_data = process_gate(gate, gate_type_map=gate_type_map, node_id=i, include_params=include_params, include_id_gates=include_id_gates, diff_cx=diff_cx)
        node_list.extend(insert_node(graph, gate_data, last_nodes=last_nodes))

    if data:
        return graph, node_list, last_nodes
    else:
        return graph


def process_gate(gate, gate_type_map, node_id=None, include_params=False, include_id_gates=False, diff_cx=False):
    """
    Process a quantum gate and return data necessary to add a node to the graph.

    :param gate: Quantum gate
    :param gate_type_map: Dictionary mapping gate types to integers
    :param node_id: Unique identifier for the node
    :param include_params: Whether to include gate parameters in the node feature vector
    :param include_id_gates: Whether to include identity gates in the graph
    :param diff_cx: Whether to differentiate between control and target qubits for CNOT gates

    :return: Tuple containing the node ID, qubit, and node data
    """
    if gate.name == 'cx':
        try:
            control_qubit = gate.qargs[0]._index
            target_qubit = gate.qargs[1]._index
        except AttributeError:
            control_qubit = gate.qubits[0]._index
            target_qubit = gate.qubits[1]._index

        # Create unique node_ids for the control and target qubits
        if node_id is not None:
            node_id_control = f"{gate.name}_{control_qubit}_control_{node_id}"
            node_id_target = f"{gate.name}_{target_qubit}_target_{node_id}"
        else:
            node_id_control = f"{gate.name}_{control_qubit}_control_{gate._node_id}"
            node_id_target = f"{gate.name}_{target_qubit}_target_{gate._node_id}"

    else:
        try:
            qubit = gate.qargs[0]._index
        except AttributeError:
            qubit = gate.qubits[0]._index

        # Create a unique node_id for the gate
        if node_id is not None:
            node_id = f"{gate.name}_{qubit}_{node_id}"
        else:
            node_id = f"{gate.name}_{qubit}_{gate._node_id}"

    # Get params
    try:
        params = gate.op.params
    except AttributeError:
        params = gate.params

    # Get gate type
    gate_type = gate.name

    # Get feature vector(s) and return node(s) data
    if gate_type == 'cx':
        feature_vector_control = build_node_features_vector(gate_type, node_id_control, gate_type_map, include_params=include_params, params=params, include_identity_gates=include_id_gates, differentiate_cx=diff_cx)
        feature_vector_target = build_node_features_vector(gate_type, node_id_target, gate_type_map, include_params=include_params, params=params, include_identity_gates=include_id_gates, differentiate_cx=diff_cx)

        return [(node_id_control, {'qubit': control_qubit, 'type': gate_type, 'params': params, 'feature_vector': feature_vector_control}),
                (node_id_target, {'qubit': target_qubit, 'type': gate_type, 'params': params, 'feature_vector': feature_vector_target})]
    
    else:
        feature_vector = build_node_features_vector(gate_type, node_id, gate_type_map, include_params=include_params, params=params, include_identity_gates=include_id_gates, differentiate_cx=diff_cx)

        return (node_id, {'qubit': qubit, 'type': gate_type, 'params': params, 'feature_vector': feature_vector})


def insert_node(graph, node_data, pred_succ=None, last_nodes=None):
    """
    Insert a node into the graph and update the last node and node position dictionaries.

    :param graph: NetworkX graph
    :param node_data: Data for the node to insert
    :param pred_succ: Predecessor and successor nodes for the new node. If None, last_nodes will be used.
    :param last_nodes: Dictionary containing the last node for each qubit
    :param node_positions: Dictionary containing the position of each node in the graph
    """
    gate_as_nodes = []

    # If the node data is a list, this means it is a CNOT gate: add two nodes (control and target)
    if isinstance(node_data, list):
        for node, ctrl_trgt in zip(node_data, ['control', 'target']):
            node_id, data = node
            graph.add_node(node_id, ctrl_trgt=ctrl_trgt, **data)

        control_id, target_id = [node_data[0][0], node_data[1][0]]
        control_qubit, target_qubit = [node_data[0][1]['qubit'], node_data[1][1]['qubit']]

        # If predecessor and successor nodes are provided, connect the control and target nodes to them
        if pred_succ:
            pred_ctrl, succ_ctrl = pred_succ[0]
            pred_trgt, succ_trgt = pred_succ[1]

            # check if qubit of predecessor and successor nodes are the same as the control and target qubits
            if not int(pred_ctrl.split('_')[1]) == int(succ_ctrl.split('_')[1]) == control_qubit:
                raise ValueError("Control qubit does not match predecessor or successor nodes")
            if not int(pred_trgt.split('_')[1]) == int(succ_trgt.split('_')[1]) == target_qubit:
                raise ValueError("Target qubit does not match predecessor or successor nodes")
            
            # connect control and target nodes to their predecessor and successor nodes
            graph.add_edge(pred_ctrl, control_id, type='qubit', create_using=nx.DiGraph()) # edge going from predecessor node to control node
            graph.add_edge(control_id, succ_ctrl, type='qubit', create_using=nx.DiGraph()) # edge going from control node to successor node
            graph.add_edge(pred_trgt, target_id, type='qubit', create_using=nx.DiGraph()) # edge going from predecessor node to target node
            graph.add_edge(target_id, succ_trgt, type='qubit', create_using=nx.DiGraph()) # edge going from target node to successor node

            # remove edges between predecessor and successor nodes
            graph.remove_edge(pred_ctrl, succ_ctrl)
            graph.remove_edge(pred_trgt, succ_trgt)
        
        # If no predecessor and successor nodes are provided, connect the control and target nodes to the last nodes
        else:
            for qubit, node_id in zip([control_qubit, target_qubit], [control_id, target_id]):
                if qubit in last_nodes:
                    graph.add_edge(last_nodes[qubit], node_id, type='qubit', create_using=nx.DiGraph())
                last_nodes[qubit] = node_id

        # Add edges between control and target nodes (bidirectional)
        graph.add_edge(control_id, target_id, type='cx', create_using=nx.DiGraph())
        graph.add_edge(target_id, control_id, type='cx', create_using=nx.DiGraph())

        gate_as_nodes.extend([control_id, target_id])

    # If the node data is a dictionary, this means it is a single-qubit gate   
    else:
        node_id, node_data = node_data  
        graph.add_node(node_id, ctrl_trgt=None, **node_data)
        qubit = node_data['qubit']

        # If predecessor and successor nodes are provided, connect the node to them
        if pred_succ:
            # First check if the qubit of the predecessor and successor nodes is the same as the current node
            pred_node, succ_node = pred_succ
            if not int(pred_node.split('_')[1]) == int(succ_node.split('_')[1]) == qubit:
                raise ValueError("Qubit does not match predecessor or successor nodes")
            
            # Connect the node to its predecessor and successor nodes
            graph.add_edge(pred_node, node_id, type='qubit', create_using=nx.DiGraph()) # edge going from predecessor node to current node
            graph.add_edge(node_id, succ_node, type='qubit', create_using=nx.DiGraph()) # edge going from current node to successor node

            # Remove edge between predecessor and successor nodes
            graph.remove_edge(pred_node, succ_node)

        # If no predecessor and successor nodes are provided, connect the node to the last node
        else:
            if qubit in last_nodes:
                graph.add_edge(last_nodes[qubit], node_id, type='qubit', create_using=nx.DiGraph())
            last_nodes[qubit] = node_id

        gate_as_nodes.append(node_id)

    return gate_as_nodes


def build_node_features_vector(gate_type, node_id, gate_type_map, include_params=False, params=None, include_identity_gates=False, differentiate_cx=False):
        """
        Creates a feature vector for a quantum gate.

        :param gate_type: Type of the quantum gate
        :param node_id: Unique identifier for the node
        :param include_params: Whether to include parameters in the feature
        :param params: Parameters of the gate
        :param include_identity_gates: Whether to include identity gates in the feature
        :param differentiate_cx: Whether to differentiate between control and target qubits for CNOT gates
        :return: Node feature vector
        """
        qubit_type_map = {'control': 0, 'target': 1}
        num_gate_types = len(gate_type_map) - 1 if not include_identity_gates else len(gate_type_map)
        feature_vector = [0] * (num_gate_types + (len(qubit_type_map) if differentiate_cx else 0))

        # not needed, if self.include_identity_gates is False there won't be any identity gates in the graph
        # if self.include_identity_gates or gate_type != 'id':
        
        feature_vector[gate_type_map[gate_type]] = 1  # one-hot encoding of gate type

        # one-hot encoding of target/control
        if differentiate_cx:
            if 'control' in node_id:
                feature_vector[num_gate_types + qubit_type_map['control']] = 1
            elif 'target' in node_id:
                feature_vector[num_gate_types + qubit_type_map['target']] = 1

        if include_params and params is not None:
            feature_vector += params

        return np.array(feature_vector)



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


def draw_ordering(quantum_circuit_graph, node_ids_ordered):
    """
    Draw the quantum circuit graph with the nodes ordered according to the given sequence of node IDs.
    :param quantum_circuit_graph: QuantumCircuitGraph object
    :param node_ids_ordered: List of node IDs in the order in which they should be drawn
    """
    # In ordering label we will have the index of the node in the ordered list
    ordering_labels = {node_id:(node_ids_ordered.index(node_id)+1) for node_id in quantum_circuit_graph.graph.nodes}

    # Draw the quantum circuit graph with each node with a label corresponding to its index in the ordered list
    quantum_circuit_graph.draw(ordering_labels, default_node_size=True)


def draw_circuit_and_graph(circuit_graph_tuple, axs=None):
    """
    Draws a Qiskit QuantumCircuit and a NetworkX graph side by side.
    
    Parameters:
    circuit_graph_tuple (tuple): A tuple containing a Qiskit QuantumCircuit and a NetworkX graph.
    """
    quantum_circuit, graph = circuit_graph_tuple
    
    if axs is None:
        _, axes = plt.subplots(1, 2, figsize=(14, 7))
    else:
        axes = axs

    node_labels_map = {
            'h': 'H',
            'rx': 'Rx',
            'ry': 'Ry',
            'rz': 'Rz',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'control': ' ',
            'target': '+',
            'id': 'I',
            't': 'T',
        }

    labels = {node: node_labels_map[graph.nodes[node]['type']] if graph.nodes[node]['type'] != 'cx' 
                else node_labels_map[graph.nodes[node]['ctrl_trgt']]
                for node in graph.nodes()}
    
    # Draw the QuantumCircuit
    quantum_circuit.draw(output='mpl', ax=axes[0])
    axes[0].set_title("Quantum Circuit")
    
    # Draw the NetworkX graph
    pos = nx.spring_layout(graph)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(graph, pos, with_labels=True, ax=axes[1], font_size=10, labels=labels)
    axes[1].set_title("Graph")
    
    if axs is None:
        plt.tight_layout()
        plt.show()
