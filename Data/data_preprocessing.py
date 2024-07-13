from matplotlib import pyplot as plt
import torch_geometric as tg
import networkx as nx
import torch
import random
import networkx as nx
import qiskit.dagcircuit.dagnode as dagnode
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers



GATE_TYPE_MAP = {
    'cx': 0, 
    'h': 1, 
    'rx': 2, 
    'ry': 3, 
    'rz': 4,
    'x': 5
    # Add here all possible gate types
}


class QuantumCircuitGraph:
    """
    Represents a quantum circuit as a directed graph using NetworkX.
    """

    def __init__(self, circuit=None):
        """
        Initializes the graph representation of a quantum circuit.
        If a circuit is provided, builds the graph from the circuit.

        :param circuit: QuantumCircuit object

        Attributes:
        - quantum_circuit: QuantumCircuit object
        - graph: networkx graph representing the quantum circuit
        - node_positions: dictionary mapping node_id to its position for visualization
        - node_ids: list of identifiers of the nodes in the graph, in the order imposed by dividing the circuit into 
          layers and ordering the nodes in each layer according to the qubit index
        - node_mapping: dictionary mapping node_id to its index in the node_features matrix
        - node_feature_matrix: tensor containing the features of the nodes in the graph (one-hot encoded gate type and qubit type)
        (parameters can be included as well, but not implemented yet)
        - last_node_per_qubit: dictionary mapping qubit index to the last node in the graph that acts on that qubit
        """
        self.quantum_circuit = circuit
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_ids = []
        self.node_mapping = {}
        self.node_feature_matrix = None
        self.last_node_per_qubit = {}

        if circuit:
            self.build_from_circuit(circuit)

    
    def build_from_circuit(self, circuit):
        """
        Builds the graph representation of a quantum circuit, given a QuantumCircuit object.
        """
        self.graph, self.node_ids, self.last_node_per_qubit, self.node_positions = build_graph_from_circuit(circuit)
        self.build_node_feature_matrix()  


    def build_node_feature_matrix(self, include_params=False):
        """
        Builds the node features matrix for the quantum circuit graph.
        The node features matrix is a tensor where each row represents the feature vector of a node in the graph.
        The feature vector consists of the one-hot encoded gate type and qubit type (control, target, or neither).
        """
        node_features = []
        for node_id in self.graph.nodes():
            gate_type = self.graph.nodes[node_id]['type']
            if gate_type != 'barrier':
                params = self.graph.nodes[node_id]['params']
                node_feature = self.build_node_features(gate_type, node_id, include_params, params)
                node_features.append(node_feature)

                self.node_mapping[node_id] = len(node_features) - 1

                # Update the node_mapping dictionary
                self.node_mapping[node_id] = len(node_features) - 1

        self.node_feature_matrix = torch.tensor(node_features, dtype=torch.float)


    def build_node_features(self, gate_type, node_id, include_params=False, params=None):
        """
        Creates a feature vector for a quantum gate.

        :param gate_type: Type of the quantum gate
        :param node_id: Unique identifier for the node
        :param include_params: Whether to include parameters in the feature
        :param params: Parameters of the gate
        :return: Node feature vector
        """
        qubit_type_map = {'control': 0, 'target': 1}
        feature_vector = [0] * len(GATE_TYPE_MAP) + [0] * len(qubit_type_map)

        feature_vector[GATE_TYPE_MAP[gate_type]] = 1  # one-hot encoding of gate type

        # one-hot encoding of target/control
        if 'control' in node_id:
            feature_vector[len(GATE_TYPE_MAP) + qubit_type_map['control']] = 1
        elif 'target' in node_id:
            feature_vector[len(GATE_TYPE_MAP) + qubit_type_map['target']] = 1

        if include_params and params is not None:
            feature_vector += params

        return feature_vector
    

    def to_sequence(self, order='qc', random_start=True):
        """
        Converts the quantum circuit graph to a sequence of nodes.
        :param order: order in which the nodes are traversed 
                        - 'qc': quantum circuit order (layer first, qubit index second) (default)
                        - 'bfs': Breadth-First Search order
        :param random_start: whether to start the traversal from a random node or from the first node in the 
        graph, if 'bfs' order is chosen
        :return: list of nodes in the graph, in the specified order
        """
        if order == 'qc':
            return self.node_ids

        if order == 'bfs':
            start_node = random.choice(list(self.graph.nodes())) if random_start else list(self.graph.nodes())[0]
            bfs_nodes = list(nx.bfs_nodes(self.graph, start_node))
            return bfs_nodes
        
        raise ValueError('Invalid order selected. Choose between "qc" and "bfs".')
    

    def draw(self):
        """
        Visualizes the quantum circuit graph using node positions and colors.
        """
        node_colors = {
            'cx': 'purple',
            'h': 'green',
            'rx': 'red',
            'ry': 'yellow',
            'rz': 'blue',
            'x': 'orange',
        }

        node_labels_map = {
            'h': 'H',
            'rx': 'Rx',
            'ry': 'Ry',
            'rz': 'Rz',
            'x': 'X',
            'c': ' ',
            't': '+'
        }

        node_labels = {node: node_labels_map[self.graph.nodes[node]['type']] if self.graph.nodes[node]['type'] != 'cx' 
                                else node_labels_map[self.graph.nodes[node]['ctrl_trgt']]
                                for node in self.graph.nodes()}

        # nx.draw(self.graph,
        #         pos=self.node_positions,
        #         with_labels=True,
        #         labels=node_labels,
        #         node_color=[node_colors[self.graph.nodes[node]['type']] for node in self.graph.nodes()],
        #         edge_color='black',
        #         arrowstyle="->",
        #         arrowsize=10
        #         )
        
        plt.figure(figsize=(15, 4))
        
        # nodes
        options = {"alpha": 0.95}
        nx.draw_networkx_nodes(
            self.graph, 
            self.node_positions, 
            nodelist= [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'cx'],
            node_color=[node_colors[self.graph.nodes[node]['type']] for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'cx'], 
            node_shape='s',
            #node_size=450,
            **options)
        
        nx.draw_networkx_nodes(
            self.graph, 
            self.node_positions, 
            nodelist= [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] == 'cx'],
            node_color=node_colors['cx'],
            node_shape='o',
            node_size=100,
            **options)

        # edges
        nx.draw_networkx_edges(
            self.graph, 
            self.node_positions, 
            edgelist=[(u, v) for u, v, data in self.graph.edges(data=True) if data['type'] != "cx"],
            width=1.0, edge_color='black',
            arrowstyle= '-|>', arrowsize=10, arrows=True
        )

        #draw edges for cx gates as undirected
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=[(u, v) for u, v, data in self.graph.edges(data=True) if data['type'] == "cx"],
            width=2,
            alpha=0.5,
            edge_color='gray',
        )


        # labels
        nx.draw_networkx_labels(
            self.graph, 
            self.node_positions, 
            node_labels, 
            font_size=14, 
            font_color="whitesmoke", 
            verticalalignment="center_baseline", 
            horizontalalignment="center")

        
        plt.tight_layout()
        plt.axis("off")
        plt.show()
    
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

    # To Do: implement maximum size M. M is, given a node-distance d ranging from 1 to diam(G), the maximum
    # cardinality of the set of nodes v_i such that their shortest-path distance to the first node v_1
    # is equal to d

    encoded_sequence = [] # list to store the encoded sequence
    end_node_idx = index if index > 0 else len(graph.as_sequence)

    for node_id in graph.as_sequence[:end_node_idx]:
        node_idx = graph.node_mapping[node_id]
        node_feature = graph.node_features[node_idx]
        encoded_sequence.append(node_feature)
    
    # Convert the encoded sequence to a PyTorch tensor
    encoded_sequence = torch.stack(encoded_sequence) 
    return encoded_sequence

