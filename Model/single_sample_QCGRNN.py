import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from Data.data_preprocessing import GATE_TYPE_MAP

# This preliminary model will deal with a single sample, so no batsch processing is needed. The model will be later extended to deal with batches of samples.


class QuantumCircuitGenerator(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size, update_graph_strategy='layer', update_hidden_strategy='gate'):
        super(QuantumCircuitGenerator, self).__init__()
        # Define dimensions
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gate_types = len(GATE_TYPE_MAP.items()) # include the identity gate
        self.window_size = window_size

        # Define udpate strategies
        self.update_graph_at_each_layer = (update_graph_strategy == 'layer')
        self.update_graph_at_each_addition = (update_graph_strategy == 'gate')

        self.update_hidden_at_each_layer = (update_hidden_strategy == 'layer')
        self.update_hidden_at_each_addition = (update_hidden_strategy == 'gate')
        
        # Define models
        self.f_CNN = ConditioningCNN(self.hidden_dim)
        self.f_adder = Adder(self.node_feature_dim, self.hidden_dim)
        self.f_gate_predictor = GatePredictor(self.node_feature_dim, self.hidden_dim, self.num_gate_types)
        # self.f_CNOT = self.build_CNOT_predictor()
        self.f_graph_RNN = GraphRNN(self.node_feature_dim, self.hidden_dim)
        self.f_stopper = Stopper(self.node_feature_dim, self.hidden_dim)


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

    def forward(self, graph_state, h):
        """
        Forward pass for generating a quantum circuit.
        The model generates a quantum circuit for a given QUBO problem instance, conditioned on the QUBO matrix.

        :param data: initial data for the graph (node feature matrix and adjacency matrix)
        :param qubo_data: QUBO matrix and number of qubits tuple
        """
        # # Unpack data tuples
        #X, A = data.x, data.edge_index
        # Q, num_qubits = qubo_data 
  
        # stop_token = False
        # #batch_size = X.size(0)

        # # Initialize hidden state for the RNN
        # h_0 = self.init_hidden_state(Q)
        # h = h_0
        
        # # Initialize graph state as zeros
        # graph_state = torch.zeros(self.window_size, self.node_feature_dim)
        
        # # Until stop token is sampled
        # while not stop_token:

        
        # till this point should be moved to the training/inference loop (outside the model)
        # so that loss can be computed and backpropagated at each addition of a node (actually then the forward pass should contain a single node addition)
        # or at each layer (actually then the forward pass should contain a single layer addition, as implemented here)
        ##########################################
            # layer = range(num_qubits)
            # for qubit in layer:
        # dict to store parameters values
        # generation_params = {}

        # alpha = self.f_adder(graph_state, h)
        # generation_params['alpha'] = alpha
        # add = self.sample_add_action(alpha)

        # if add:
        #     gamma = self.f_gate_type(graph_state, h)
        #     gate_type = self.sample_gate_type(gamma)
        # else:
        #     gate_type = 'i' # identity gate

        gamma = self.f_gate_predictor(graph_state, h)
        # gate_type = self.sample_gate_type(gamma)

        # X = self.update_node_features(data.x, gate_type)
        # A = self.update_adjacency(data.edge_index, qubit, layer, num_qubits)
        # data = Data(x=X, edge_index=A)
        
        # if self.update_graph_at_each_addition:
        #     graph_state = self.update_graph_state(data, self.window_size)

        if self.update_hidden_at_each_addition:
            h = self.f_graph_RNN(graph_state, h)

            # if self.update_graph_at_each_layer:
            #     graph_state = self.update_graph_state(X, A, self.window_size)
                                                      
            # if self.update_hidden_at_each_layer:
            #     h = self.f_graph_RNN(graph_state, h)

        ##########################################
        # also stopper should be included in the training/inference loop
    
            # sigma = self.f_stopper(h, graph_state, h_0)
            # stop_token = self.sample_stop_action(sigma)
        
        return graph_state, h, gamma


    def init_hidden_state(self, conditioning=None):
        """
        Initialize the hidden state for the RNN.
        If conditioning is provided, use it to initialize the hidden state.
        Else, initialize it as zeros.
        """
        if conditioning is not None:
            h = self.f_CNN(conditioning)
        else:
            #batch_size = conditioning.size(0)
            h = torch.zeros(1, 1, self.hidden_dim) #
        return h
    
    

    def sample_add_action(self, p):
        """
        Sample the action for adding a node.
        It simply samples from a Bernoulli distribution with parameter p.

        :param p: probability of adding a node
        :return: boolean value indicating whether to add a node (int)
        """
        bernoulli = torch.distributions.Bernoulli(p)
    
        # Sample from the distribution
        sample = bernoulli.sample().item()
        
        # Convert the sample to an integer (0 or 1)
        return int(sample)
    
    

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

        print(p_vec)
        
        # Convert p_vec to a torch tensor if it is not already
        if not isinstance(p_vec, torch.Tensor):
            p_vec = torch.tensor(p_vec)
        
        categorical = torch.distributions.Categorical(probs=p_vec)
        sampled_gate_type_index = categorical.sample() # sample a gate type index

        return gate_types[sampled_gate_type_index]
    
    

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
        # X: (num_nodes, node_feature_dim)
        new_row = torch.zeros(1, X.size(1)) 

        # Get the index of the gate type in the mapping and set the corresponding element to 1
        gate_idx = GATE_TYPE_MAP[gate_type]
        new_row[:, gate_idx] = 1

        return torch.cat((X, new_row), dim=0)



    def update_adjacency(self, A, info):
        """
        Update the adjacency matrix.
        In PyTorch Geometric, the adjacency matrix is represented in COO format, i.e. a tensor with shape (2, num_edges), that is num_edges
        tuples of the form (source, target). To update the adjacency matrix, we need to add a new edge from the qubit-th node in the previous
        layer to the new node in the graph.

        :param A: adjacency matrix (COO format)
        :param qubit: qubit on which the gate is applied

        :return: updated adjacency matrix
        """
        curr_layer, qubit, layer, num_qubits = info

        # A: (2, num_edges)
        # Add an edge from the qubit-th node in the previous layer to the new node (qubit-th node in the current layer)
        idx_curr_node = A.size(1) # the index of new node is the number of edges in the graph
        idx_prev_node = (curr_layer - 1) * num_qubits + qubit

        new_edge = torch.tensor([[idx_prev_node, idx_curr_node]], dtype=torch.long)
        return torch.cat((A, new_edge), dim=1)     
        


    def update_graph_state(self, data, window_size):
        """
        Update the graph state.
        It keeps track of the last window_size nodes added to the graph, by keeping the last window_size rows of the node feature matrix.
        If not enough nodes have been added to the graph yet, the graph state is padded with zeros. This is not implemented here, but still it
        is what happens, since graph state is initialized as a tensor of zeros of size (window_size, node_feature_dim) at the start of the generation
        process.

        ToDo: could be important to include adjacency information in the graph state, to keep track of the connections between the nodes.      

        :param X: node feature matrix (num_nodes, node_feature_dim)
        :param window_size: size of the window to keep track of the last nodes

        :return: updated graph state (window_size, node_feature_dim)
        """
        X, A = data.x, data.edge_index
        return X[-window_size:, :]


    def update_graph_data(self, data, gate_type, current_info):
        """
        Update the graph data.
        It updates the graph data structure with the new node added to the graph.

        :param data: current graph data
        :param gate_type: gate type to be added (str)
        :param current_info: tuple containing information about the current node to be added

        :return: updated graph data
        """
        X = self.update_node_features(data.x, gate_type)
        A = self.update_adjacency(data.edge_index, current_info)
  
        return Data(x=X, edge_index=A)



class ConditioningCNN(nn.Module):
    def __init__(self, output_size):
        super(ConditioningCNN, self).__init__()
        self.output_size = output_size
        self.model = self.build_CNN(output_size)

    def forward(self, x):
        latent_representation = self.model(x)
        batch_size = latent_representation.size(0)
        return latent_representation.view(batch_size, 1, self.output_size) # flatten the output tensor to (batch_size, 1, output_size)

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
        self.softmax = nn.Softmax(dim=1) # to output a probability distribution over the gate types
            
        

    def forward(self, x, h):
        """
        Forward pass for the gate predictor model.
        :param h: hidden state produced by the graph RNN
        :param x: current state of the graph
        """
        out, _ = self.rnn(x, h)
        return self.softmax(self.fc(out))
    

class GraphRNN(nn.Module):
    '''
    Model for processing the graph, updating its hidden state, and generating the node sequence.
    
    The model should be a simple RNN, which takes the current state of the graph (e.g., node feature matrix and adjacency of last window_size nodes)
    and the hidden state produced by the previous step of the RNN, and outputs the updated hidden state. The updated hidden state will be used to
    generate the next node in the sequence, and the process will be repeated until the stop token is sampled.
    '''

    def __init__(self, node_feature_dim, hidden_dim):
        super(GraphRNN, self).__init__()
        self.rnn = nn.RNN(node_feature_dim, hidden_dim, num_layers=1, batch_first=True)
        

    def forward(self, x, h):
        """
        Forward pass for the graphRNN model.
        :param h: hidden state produced by the graphRNN at the previous step
        :param x: current state of the graph
        """
        _, hidden = self.rnn(x, h)
        return hidden
    

class Stopper(nn.Module):
    '''
    Model for deciding when to stop adding nodes.
    
    The model should output a single scalar value, which will be used as a probability of stopping the generation process. The model should take the
    hidden state produced by the graph RNN as input, as well as the current state of the graph (e.g. node feature matrix and adjacency of last
    window_size nodes), and output the probability of stopping the generation process.
    '''

    def __init__(self, node_feature_dim, hidden_dim):
        super(Stopper, self).__init__()
        self.rnn = nn.RNN(node_feature_dim, hidden_dim*2, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) # should output a single scalar value
        self.sigmoid = nn.Sigmoid() # to output a probability
        

    def forward(self, x, h, h_0):
        """
        Forward pass for the stopper model.
        :param h: hidden state produced by the graph RNN
        """
        hidden = torch.cat((h, h_0), dim=1) # concatenate the hidden state produced by the graph RNN and the initial hidden state
        out, _ = self.rnn(x, hidden)
        return self.sigmoid(self.fc(out))