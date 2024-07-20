from matplotlib import pyplot as plt
import torch_geometric as tg
import networkx as nx
import torch
import random
import networkx as nx
from .data_preprocessing import build_graph_from_circuit, encode_sequence
from .data_preprocessing import GATE_TYPE_MAP


class QuantumCircuitGraph:
    """
    Represents a quantum circuit as a directed graph using NetworkX.
    """

    def __init__(self, circuit=None, include_params=False):
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
        - n_node_features: number of features for each node in the graph
        - last_node_per_qubit: dictionary mapping qubit index to the last node in the graph that acts on that qubit
        """
        self.quantum_circuit = circuit
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_ids = []
        self.node_mapping = {}
        self.node_feature_matrix = None
        self.n_node_features = None
        self.adjacency_matrix = None
        self.last_node_per_qubit = {}

        self.include_params = include_params
        if self.include_params:
            raise NotImplementedError('Including parameters in the node features is not implemented yet.')

        if circuit:
            self.build_from_circuit(circuit)

    
    def build_from_circuit(self, circuit):
        """
        Builds the graph representation of a quantum circuit, given a QuantumCircuit object.
        """
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
    

    def draw(self, custom_labels=None, default_node_size=False):
        """
        Visualizes the quantum circuit graph using node positions and colors.

        :param custom_labels: dictionary mapping node_id to custom labels for visualization
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

        if custom_labels is None:
            node_labels = {node: node_labels_map[self.graph.nodes[node]['type']] if self.graph.nodes[node]['type'] != 'cx' 
                                else node_labels_map[self.graph.nodes[node]['ctrl_trgt']]
                                for node in self.graph.nodes()}
        else:
            node_labels = custom_labels

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
        






