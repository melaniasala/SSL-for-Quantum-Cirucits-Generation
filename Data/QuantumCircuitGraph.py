import networkx as nx
import torch
import matplotlib.pyplot as plt
import random
from qiskit import QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import IGate
from qiskit.circuit.barrier import Barrier
from qiskit.transpiler.passes import RemoveBarriers
from .data_preprocessing import GATE_TYPE_MAP, encode_sequence, build_graph_from_circuit, draw_circuit_and_graph


class QuantumCircuitGraph:
    """
    Represents a quantum circuit as a directed graph using NetworkX.
    """

    def __init__(self, circuit=None, include_params=False, include_identity_gates=False, differentiate_cx=False):
        """
        Initializes the graph representation of a quantum circuit.
        If a circuit is provided, builds the graph from the circuit.

        :param circuit: QuantumCircuit object
        :param include_params: Whether to include parameters in the node features (not implemented yet)
        :param include_identity_gates: Whether to include identity gates in the circuit
        :param differentiate_cx: Whether to differentiate control and target qubits of a cx gate in node features
        """
        self.quantum_circuit = circuit
        self.graph = nx.DiGraph()
        self.node_positions = {} # dictionary mapping node_id to its position in the graph for visualization
        self.node_ids = [] # list of node_ids in the graph (in in the order imposed by dividing the circuit into 
        # layers and ordering the nodes in each layer according to the qubit index)
        self.node_mapping = {} # dictionary mapping node_id to its index in the node_features matrix
        self.node_feature_matrix = None # tensor containing the features of the nodes in the graph
        self.n_node_features = None 
        self.adjacency_matrix = None
        self.last_node_per_qubit = {}

        self.include_params = include_params
        self.include_identity_gates = include_identity_gates
        self.differentiate_cx = differentiate_cx

        if self.include_params:
            raise NotImplementedError('Including parameters in the node features is not implemented yet.')

        if circuit:
            self.build_from_circuit(circuit)


    def insert_identity_gates(self, circuit):
        """
        Inserts identity gates in the circuit to create a grid-like structure.
        """
        n_qubits = circuit.num_qubits
        remove_barriers = RemoveBarriers()

        circuit_no_barriers = remove_barriers(circuit)
        dag = circuit_to_dag(circuit_no_barriers)
        layers = list(dag.multigraph_layers())[1:-1]
        
        new_layers = []

        for layer in layers:
            identities = []
            active_qubits_in_layer = [qubit._index for gate in layer if isinstance(gate, DAGOpNode) for qubit in gate.qargs]
            ops_in_layer = [gate for gate in layer if isinstance(gate, DAGOpNode)]
            for q in range(n_qubits):
                if q not in active_qubits_in_layer:
                    # Add an identity gate in correspondence to the qubit not having gates in the layer
                    id_gate = IGate()
                    id_dag_node = DAGOpNode(id_gate, qargs=[QuantumRegister(n_qubits, 'q')[q]], cargs=[])
                    identities.append(id_dag_node)
            
            new_layers.append(ops_in_layer + identities)

        new_dag = DAGCircuit()
        qr = QuantumRegister(n_qubits, 'q')
        new_dag.add_qreg(qr)

        for layer in new_layers:
            for gate in layer:
                new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)
            # add barrier between layers for better visualization
            barrier_op = Barrier(num_qubits=n_qubits) 
            new_dag.apply_operation_back(barrier_op, qr, [])

        new_circuit = dag_to_circuit(new_dag)
        return new_circuit

    
    def build_from_circuit(self, circuit):
        """
        Builds the graph representation of a quantum circuit, given a QuantumCircuit object.
        """
        if self.include_identity_gates:
            circuit = self.insert_identity_gates(circuit)
        self.graph, self.node_ids, self.last_node_per_qubit, self.node_positions = build_graph_from_circuit(circuit)
        self.build_node_feature_matrix()
        self.build_adjacency_matrix()


    def build_node_feature_matrix(self, include_params=False):
        """
        Builds the node features matrix for the quantum circuit graph.
        The node features matrix is a tensor where each row represents the feature vector of a node in the graph.
        The feature vector consists of the one-hot encoded gate type and qubit type (control, target, or neither).
        """
        node_features = []
        for node_id in self.node_ids:
            gate_type = self.graph.nodes[node_id]['type']
            if gate_type != 'barrier':
                params = self.graph.nodes[node_id]['params']
                node_feature = self.build_node_features(gate_type, node_id, include_params, params)
                node_features.append(node_feature)

                # Update the node_mapping dictionary
                self.node_mapping[node_id] = len(node_features) - 1

        self.node_feature_matrix = torch.tensor(node_features, dtype=torch.float)
        self.n_node_features = self.node_feature_matrix.size(1)


    def build_adjacency_matrix(self):
        """
        Builds the adjacency matrix (CSR format) of the quantum circuit graph.
        """
        # Extract the adjacency matrix in the node addition order
        adj_matrix = nx.adjacency_matrix(self.graph)

        # Create a mapping between self.node_ids and their corresponding indices in self.graph.nodes
        node_order_mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        ordered_indices = [node_order_mapping[node] for node in self.node_ids]

        # Reorder the rows and columns of the CSR matrix
        reordered_adj_matrix = adj_matrix[ordered_indices, :][:, ordered_indices]

        self.adjacency_matrix = reordered_adj_matrix


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
        num_gate_types = len(GATE_TYPE_MAP) - 1 if not self.include_identity_gates else len(GATE_TYPE_MAP)
        feature_vector = [0] * (num_gate_types + (len(qubit_type_map) if self.differentiate_cx else 0))

        # not needed, if self.include_identity_gates is False there won't be any identity gates in the graph
        # if self.include_identity_gates or gate_type != 'id':
        
        feature_vector[GATE_TYPE_MAP[gate_type]] = 1  # one-hot encoding of gate type

        # one-hot encoding of target/control
        if self.differentiate_cx:
            if 'control' in node_id:
                feature_vector[num_gate_types + qubit_type_map['control']] = 1
            elif 'target' in node_id:
                feature_vector[num_gate_types + qubit_type_map['target']] = 1

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

            # Convert to undirected graph to perform BFS
            graph = self.graph.to_undirected()

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
        
        raise ValueError('Invalid order selected. Choose between "qc" and "bfs".')
    

    def draw(self, custom_labels=None, default_node_size=False, ax=None):
        """
        Visualizes the quantum circuit graph using node positions and colors.

        :param custom_labels: dictionary mapping node_id to custom labels for visualization
        :param default_node_size: boolean indicating whether to use the default node size
        :param ax: matplotlib axes object to draw the graph on
        """
        node_colors = {
            'cx': 'purple',
            'h': 'green',
            'rx': 'red',
            'ry': 'yellow',
            'rz': 'blue',
            'x': 'pink',
            'y': 'orange',
            'z': 'cyan',
            'id': 'grey',
        }

        node_labels_map = {
            'h': 'H',
            'rx': 'Rx',
            'ry': 'Ry',
            'rz': 'Rz',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'c': ' ',
            't': '+',
            'id': 'I'
        }

        if custom_labels is None:
            node_labels = {node: node_labels_map[self.graph.nodes[node]['type']] if self.graph.nodes[node]['type'] != 'cx' 
                                else node_labels_map[self.graph.nodes[node]['ctrl_trgt']]
                                for node in self.graph.nodes()}
        else:
            node_labels = custom_labels

        if ax is None:
            plt.figure(figsize=(15, 4))
            ax = plt.gca() # get current axes

        # nodes
        options = {"alpha": 0.95}
        nx.draw_networkx_nodes(
            self.graph, 
            self.node_positions, 
            nodelist= [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'cx'],
            node_color=[node_colors[self.graph.nodes[node]['type']] for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'cx'], 
            node_shape='s',
            **options)
        
        nx.draw_networkx_nodes(
            self.graph, 
            self.node_positions, 
            nodelist= [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] == 'cx'],
            node_color=node_colors['cx'],
            node_shape='o',
            node_size=100 if not default_node_size else 300,
            **options)

        # edges
        nx.draw_networkx_edges(
            self.graph, 
            self.node_positions, 
            edgelist=[(u, v) for u, v, data in self.graph.edges(data=True) if data['type'] != "cx"],
            width=1.0, edge_color='black',
            arrowstyle= '-|>', arrowsize=10, arrows=True
        )

        #draw edges for cx gates as undirected (double directed edges)
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=[(u, v) for u, v, data in self.graph.edges(data=True) if data['type'] == "cx"],
            width=2,
            alpha=0.25,
            edge_color='purple',
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
        plt.title("Graph representation of the Quantum Circuit")
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
    

    def encode_sequence(self, sequence_length=None, end_index=None, use_padding=True, padding_value=0.0):
        """
        Encodes a sequence of nodes into a tensor representation.
        :param sequence_length: Maximum size of the sequence to encode (default is the full sequence)
        :param end_index: Index of the last node in the sequence (exclusive)
        :param use_padding: Whether to pad the sequence to the specified length
        :param padding_value: Value to use for padding
        :return: Tensor representation of the sequence
        """
        
        return encode_sequence(self, sequence_length, end_index, use_padding, padding_value)
    

    # define a method to draw both the circuit and the correspondent graph
    def draw_circuit_and_graph(self, circuit_like_graph=False):
        """
        Visualizes the quantum circuit and its graph representation side by side.
        """
        if circuit_like_graph: 
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            self.quantum_circuit.draw(output='mpl', ax=axes[0])
            axes[0].set_title("Quantum Circuit")
        
            self.draw(ax=axes[1])
            axes[1].set_title("Graph representation of the Quantum Circuit")
            
            # Display the combined plot
            plt.show()

        else:
            draw_circuit_and_graph((self.quantum_circuit, self.graph))
        






