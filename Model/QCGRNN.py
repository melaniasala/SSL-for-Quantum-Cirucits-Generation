import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from Data.data_preprocessing import GATE_TYPE_MAP


class QuantumCircuitGenerator(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_gate_types, update_graph_strategy='layer', update_hidden_strategy='gate'):
        super(QuantumCircuitGenerator, self).__init__()
        # Define dimensions
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gate_types = num_gate_types

        # Define udpate strategies
        self.update_graph_at_each_layer = (update_graph_strategy == 'layer')
        self.update_graph_at_each_addition = (update_graph_strategy == 'gate')

        self.update_hidden_at_each_layer = (update_hidden_strategy == 'layer')
        self.update_hidden_at_each_addition = (update_hidden_strategy == 'gate')
        
        # Define models
        self.f_CNN = ConditioningCNN(self.hidden_dim)
        self.f_adder = Adder(self.node_feature_dim, self.hidden_dim)
        self.f_gate_type = GatePredictor(self.node_feature_dim, self.hidden_dim, self.node_feature_dim)
        self.f_CNOT = self.build_CNOT_predictor()
        self.f_graph_RNN = self.build_graph_RNN()
        self.f_stopper = self.build_stopper()


    def build_gate_type_predictor(self):
        """
        Build the model for predicting gate types. 
        This should output a vector of size num_gate_types, which will be used as a probability distribution over the gate types. The model 
        should take the hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and 
        adjacency of last window_size nodes), and output the probability distribution over the gate types.

        Placeholder for actual implementation.
        """
        return nn.Linear(self.hidden_dim, self.num_gate_types)

    # def build_CNOT_predictor(self):
    #     """
    #     Build the model for predicting control and target qubits for CNOT gates.
    #     This should output a vector of size n_qubits, which will be used as a probability distribution over the qubits. The model should take 
    #     the hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and adjacency
    #     of last window_size nodes), and output the probability distribution over the qubits.
    #     Placeholder for actual implementation.
    #     """
    #     return nn.Linear(self.hidden_dim, 2)

    def build_graph_RNN(self):
        """
        Build the RNN for processing the graph, updating its hidden state, and generating the node sequence.
        This should be a simple RNN, which takes the current state of the graph (e.g., node feature matrix and adjacency of last window_size nodes)
        and the hidden state produced by the previous step of the RNN, and outputs the updated hidden state. The updated hidden state will be used
        to generate the next node in the sequence, and the process will be repeated until the stop token is sampled.

        Placeholder for actual implementation.
        """
        return nn.GRU(self.node_feature_dim, self.hidden_dim)

    def build_stopper(self):
        """
        Build the model for deciding when to stop adding nodes.
        This should output a single scalar value, which will be used as a probability of stopping the generation process. The model should take
        the hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and adjacency of
        last window_size nodes), and output the probability of stopping the generation process.

        Placeholder for actual implementation.
        """
        return nn.Linear(self.hidden_dim, 1)

    def forward(self, data, qubo_data, window_size):
        """
        Forward pass for generating a quantum circuit.
        :param graph_representation: Current state of the graph (e.g., node feature matrix)
        :param node_sequence: Sequence of nodes generated so far
        """
        # Unpack data tuples
        X, A = data.x, data.edge_index
        Q, num_qubits = qubo_data # to accomodate variable input sizes of qubo matrices, DataLoader will save the dimensions of QUBO matrices in num_qubits and then pad them dynamically to the maximum size in the batch
  
        stop_token = False
        batch_size = X.size(0)

        # Initialize hidden state for the RNN
        h_0 = self.init_hidden_state(Q)
        h = h_0
        
        # Initialize graph state as zeros
        graph_state = torch.zeros(batch_size, window_size, self.node_feature_dim)
        
        # Until stop token is sampled
        while not stop_token:
            layer = range(num_qubits)
            for qubit in layer:
                # alpha = self.f_adder(h, graph_state)
                # add = self.sample_add_action(alpha)

                # if add:
                #     gamma = self.f_gate_type(h, graph_state)
                #     gate_type = self.sample_gate_type(gamma)
                # else:
                #     gate_type = 'i'

                # GatePredictor can also predict the identity gate, so that there are no different control flows to handle
                gamma = self.f_gate_type(h, graph_state)
                gate_type = self.sample_gate_type(gamma)

                X = self.update_node_features(X, gate_type)
                A = self.update_adjacency(A, qubit)

                data = Data(x=X, edge_index=A)
                
                if self.update_graph_at_each_addition:
                    graph_state = self.update_graph_state(X, A, window_size)

                if self.update_hidden_at_each_addition:
                    h = self.f_graph_RNN(h, graph_state)

            if self.update_graph_at_each_layer:
                graph_state = self.update_graph_state(X, A, window_size)
                                                      
            if self.update_hidden_at_each_layer:
                h = self.f_graph_RNN(h, graph_state)
    
            sigma = self.f_stopper(h, graph_state, h_0)
            stop_token = self.sample_stop_action(sigma)
        
        return X, A


    def init_hidden_state(self, conditioning=None):
        """
        Initialize the hidden state for the RNN.
        If conditioning is provided, use it to initialize the hidden state.
        Else, initialize it as zeros.
        """
        if conditioning:
            h = self.f_CNN(conditioning)
        else:
            batch_size = conditioning.size(0)
            h = torch.zeros(batch_size, 1, self.hidden_dim)
        return h
    
    

    def sample_add_action(self, p):
        """
        Sample the action for adding a node.
        It simply samples from a Bernoulli distribution with parameter p.

        :param p: probability of adding a node
        :return: boolean value indicating whether to add a node (int)
        """
        return np.random.binomial(1, p)
    
    

    def sample_gate_type(self, p_vec):
        """
        Sample the gate type.
        It samples from a categorical distribution with parameters p_vec.

        Notice that the last gate type in the mapping is always the identity gate, so this sampler can deal with both the case where the identity
        gate is included in the gate types and the case where it is not. The only thing to do is to set correctly the number of gate types in the
        GateSampler instance to include the identity gate or not; the sampler will automatically deal with both cases.

        :param p_vec: probability distribution over the gate types
        :return: selected gate type (int)
        """
        gate_types = list(GATE_TYPE_MAP.keys())[:len(p_vec)]
        return np.random.choice(gate_types, p=p_vec)
    
    

    def sample_CNOT_action(self, h):
        """
        Sample the action for CNOT gate (control and target qubits).
        Placeholder for actual implementation.
        """
        return 'sample CNOT'

    def sample_stop_action(self, h):
        """
        Sample the action for stopping node addition.
        Placeholder for actual implementation.
        """
        return 'sample STOP'



    def update_node_features(self, X, gate_type):
        """
        Update the node feature matrix.
        It appends a new row (one-hot encoding of the gate type) to the node feature matrix, representing the new node added to the graph.

        :param X: node feature matrix
        :param gate_type: gate type to be added (str)

        :return: updated node feature matrix
        """
        # X: (batch_size, num_nodes, node_feature_dim)
        new_row = torch.zeros(X.size(0), 1, X.size(2)) 

        # Get the index of the gate type in the mapping and set the corresponding element to 1
        gate_idx = GATE_TYPE_MAP[gate_type]
        new_row[:, :, gate_idx] = 1

        return torch.cat((X, new_row), dim=1)



    def update_adjacency(self, A, qubit, curr_layer, num_qubits):
        """
        Update the adjacency matrix.
        In PyTorch Geometric, the adjacency matrix is represented in COO format, i.e. a tensor with shape (2, num_edges), that is num_edges
        tuples of the form (source, target). To update the adjacency matrix, we need to add a new edge from the qubit-th node in the previous
        layer to the new node in the graph.

        :param A: adjacency matrix (COO format)
        :param qubit: qubit on which the gate is applied

        :return: updated adjacency matrix
        """
        # A: (batch_size, 2, num_edges)
        # Add an edge from the qubit-th node in the previous layer to the new node (qubit-th node in the current layer)
        idx_curr_node = A.size(2) # the index of new node is the number of edges in the graph
        idx_prev_node = (curr_layer - 1) * num_qubits + qubit
       
        new_edge = torch.tensor([[idx_prev_node, idx_curr_node]], dtype=torch.long)
        return torch.cat((A, new_edge), dim=2)



class ConditioningCNN(nn.Module):
    def __init__(self, output_size):
        super(ConditioningCNN, self).__init__()
        self.model = self.build_CNN(output_size)

    def forward(self, x):
        latent_representation = self.model(x)
        return latent_representation.view(latent_representation.size(0), 1, -1) # flatten the output tensor to (batch_size, 1, output_size)

    def build_CNN(self, output_size):
        """
        Build the model for processing the input QUBO matrix (conditioning for the generation of the circuit).
        This should be a fully-convolutional CNN, so that it can accomodate QUBO matrices of different sizes, and output a fixed-size 
        latent representation of the QUBO problem instance we are trying to solve. The latent representation will be used as input to the
        graph RNN at initial step of the generation process.
        This latent representation will be learnt in an unsupervised manner, by training the whole model end-to-end on the task of generating
        quantum circuits for QUBO problems.

        Placeholder for actual implementation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # padding=1 to keep the size of the input (valid padding)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Conv2d(in_channels=64, out_channels=output_size, kernel_size=1, stride=1, padding=0) # 1x1 convolution to reduce the number of channels to output_size
        )
    
    

class Adder(nn.Module):
    '''
    Model for adding nodes to the circuit graph.
    
    The model should output a single scalar value, which will be used as a probability of adding a node to the graph. The model should take the
    hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and adjacency of last
    window_size nodes), and output the probability of adding a gate in the current layer of the circuit.
    The current state of the graph should be a tensor of shape (batch_size, window_size, node_feature_dim): then it makes sense to implement a RNN
    with sequence length equal to the window_size, and input size equal to the node_feature_dim, to easily process the graph state as a matrix-like
    structure and to mantain a notion of order in the nodes. The initial hidden state should be the hidden state produced by the graph RNN at the
    previous step of the generation process.
    '''

    def __init__(self, node_feature_dim, hidden_dim):
        super(Adder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        self.rnn = nn.RNN(self.node_feature_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1) # should output a single scalar value
        self.sigomid = nn.Sigmoid() # to output a probability
        

    def forward(self, x, h):
        """
        Forward pass for the adder model.
        :param h: hidden state produced by the graph RNN
        :param x: current state of the graph
        """
        out, _ = self.rnn(x, h)
        return self.sigmoid(self.fc(out)) 
    


class GatePredictor(nn.Module):
    '''
    Model for predicting gate types.
    
    The model should output a vector of size num_gate_types, which will be used as a probability distribution over the gate types. The model should
    take the hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and adjacency of
    last window_size nodes), and output the probability distribution over the gate types.
    '''

    def __init__(self, node_feature_dim, hidden_dim, num_gate_types):
        super(GatePredictor, self).__init__()
        self.rnn = nn.RNN(node_feature_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_gate_types) 
        self.softmax = nn.Softmax() # to output a probability distribution over the gate types
        

    def forward(self, x, h):
        """
        Forward pass for the gate predictor model.
        :param h: hidden state produced by the graph RNN
        :param x: current state of the graph
        """
        out, _ = self.rnn(x, h)
        return self.softmax(self.fc(out))
