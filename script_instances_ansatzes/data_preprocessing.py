import qaoa, problems
import torch_geometric as tg
import networkx as nx
import torch

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
    :param circuit: quantum circuit
    :return: networkx graph representing the circuit
    """
    # Create an empty graph
    graph = nx.Graph()

    # Dictionary to keep track of the last nodes for each qubit 
    # and custom position for each node
    last_nodes = {}
    node_positions = {}

    # Iterate over the gates in the circuit
    for i, gate in enumerate(circuit['gates']):
        type = gate['gate_type']
        
        # If CNOT gate
        if type == 'cx':
            control_qubit = gate['qubits'][0]
            target_qubit = gate['qubits'][1]
            
            # Create unique node_ids for the control and target qubits
            node_id_control = f"{type}_{control_qubit}_control_{i}"
            node_id_target = f"{type}_{target_qubit}_target_{i}"

            # Add nodes for the control and target qubits with their respective types
            graph.add_node(node_id_control, type='cx_control')
            graph.add_node(node_id_target, type='cx_target')

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
            graph.add_node(node_id, type=type, params=gate['params'])

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

    current_node_idx = 0

    for i, gate in enumerate(circuit['gates']):
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
