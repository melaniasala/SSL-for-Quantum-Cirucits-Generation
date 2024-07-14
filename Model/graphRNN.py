import torch
import torch_geometric as tg
import torch.nn as nn

# h0 -> can be initialized in an alternative way? now all zeros
# Maybe can be initialized with a learned hidden state from the associated QUBO matrix (conditioning)

class ARGraphRNN(nn.Module):
    '''
    Autoregressive GraphRNN model
    '''

    def __init__(self, input_size, hidden_size, window_size, output_size):   
        '''
        input_size: size of the input features (conditioning)
        '''
        super(ARGraphRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.window_size = window_size
        
        self.graphRNN = GraphRNN(input_size, hidden_size, window_size, output_size)

    def forward(self, x=None, conditioning=None, training=True):
        '''
        At training time, the model should be fed with the whole sequence of nodes features and adjacency vectors
        '''
        stop_token = False
        batch_size = conditioning.size(0)

        if training and x is None:
            raise ValueError('Ground truth should be provided during training')
        
        # conditionning is a tensor of shape (batch_size, input_size) (if a vector)
        h_0 = self.init_hidden_state(batch_size, self.hidden_size, conditioning).to(conditioning.device)
        h_i_minus_1 = h_0

        
        if training:
            node_feature_matrix, adjacency_matrix = x
        else:
            node_feature_matrix = []
            adjacency_matrix = []

        t = 0

        while not stop_token:
            # How to manage first node? (no previous node to look back)
            if t == 0:
                x = None
        
            theta, phi, psi, h_i = self.graphRNN(x, h_i_minus_1)

            # sample edges
            edges = torch.bernoulli(theta)
            adjacency_matrix.append(edges)

            # sample node features using phi that contains probabilities over the gate types
            gate_type_idx = torch.multinomial(phi, 1) # sample one gate type 
            node_features = torch.zeros(self.input_size)
            node_features[gate_type_idx] = 1
            node_feature_matrix.append(node_features)

            # sample stop token (bernoulli of parameter psi)
            stop_token = torch.bernoulli(psi).bool()

            # update hidden state
            h_i_minus_1 = h_i


    def init_hidden_state(self, batch_size, hidden_size, conditioning=None):
        if conditioning is None:
            return torch.zeros(self.num_layers, batch_size, hidden_size)
        else:
            pass # conditioned initialization


    


class GraphRNN(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, output_size):
        '''
        input_size: number of features in the input (nodes features)
        window_size: number of temporal steps the model can look back
        graphRNN should take as input 
        - node_features for the sequence: a tensor of shape (batch_size, input_size) 
        (- edge_features for the sequence: a tensor of shape (batch_size, window_size) -> for now undirected graph
        - adjacency vector for the sequence: a tensor of shape (batch_size, window_size)
        
        '''
        super(GraphRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.window_size = window_size
        
        self.graph_level_RNN = graph_level_RNN(input_size, hidden_size, output_size)
        self.edge_level_MLP = edge_level_MLP(input_size, output_size)
        self.node_features_level_MLP = node_features_level_MLP(input_size, output_size)
        self.stop_token_MLP = stop_token_MLP(input_size, output_size)
    

    def forward(self, x, h_0=None):
        # x should be a tuple of (node_features, adjacency_matrix)
        # node_features: (batch_size, input_size)
        # adjacency_vct: (batch_size, window_size)
        node_features, adjacency_vct = x
        batch_size = node_features.size(0)
        window_size = adjacency_vct.size(1)
       
        # concatenate hidden state, node features and adjacency matrix for the current node
        x = torch.cat((node_features, adjacency_vct), dim=1)
        graph_h = self.graph_level_RNN(x, h_0)
        
        # concatenate hidden state, node features and adjacency matrix for the current node
        x = torch.cat((graph_h, node_features, adjacency_vct), dim=1)
        theta = self.edge_level_MLP(x)
        
        # concatenate hidden state, node features and adjacency matrix for the current node
        x = torch.cat((graph_h, node_features, adjacency_vct), dim=1)
        phi = self.node_features_level_MLP(x)
        
        # concatenate hidden state, node features and adjacency matrix for the current node
        x = torch.cat((graph_h, node_features, adjacency_vct), dim=1)
        psi = self.stop_token_MLP(x)

        return theta, phi, psi, graph_h
    
    

class graph_level_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(graph_level_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.rnn = nn.RNN(input_size, hidden_size, self.num_layers, output_size, batch_first=True)
        #self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h_0=None, conditioning=None):
        # if first node (no nodes already in the graph) or if first autoregressive step
        # so no hidden state has already been computed for the sample
        if not h_0:
            h_0 = self.init_hidden_state(x, conditioning).to(x.device)

        _, hidden = self.rnn(x, h_0) 
        return hidden # should return the hidden state, not the output here
    
    def init_hidden_state(self, batch_size, hidden_size, conditioning=None):
        if conditioning is None:
            return torch.zeros(self.num_layers, batch_size, hidden_size)
        else:
            pass # conditioned initialization


class edge_level_MLP(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        The MLP predicts the presence of edges between current node and the last window_size nodes in the graph,
        by predicting a parameter vector theta (the output), which entries correspond to the probability of the 
        existence of an edge between the current node and each of the last window_size nodes in the graph.

        input_size: corresponds to the size of concatenation of hidden state, node features and adjacency vector
        for the current node
        output_size: corresponds to window_size, the dimension of the output vector theta
        '''
        super(edge_level_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.sigmoid = nn.Sigmoid() # output probabilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)     


class node_features_level_MLP(nn.Module):
    '''
    The MLP predicts the node features for the current node, by predicting a parameter vector phi (the output) 
    containing probabilities that the current node belongs to each of the gate types available in the dataset.

    ToDo: CNOT! maybe better to consider target and control as two separate gate types, easier...
    
    input_size: corresponds to the size of concatenation of hidden state, node features and adjacency vector
    output_size: corresponds to the number of gate types available in the dataset
    '''
    def __init__(self, input_size, output_size):
        super(node_features_level_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1) # output probabilities over the gate types
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    

class stop_token_MLP(nn.Module):
    '''
    The MLP predicts the stop token for the current node, by predicting a parameter vector beta (the output) 
    containing probabilities that the current node is the last node in the graph.

    input_size: corresponds to the size of concatenation of hidden state, node features and adjacency vector
    output_size: corresponds to 1, the dimension of the output vector beta
    '''
    def __init__(self, input_size, output_size):
        super(stop_token_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.sigmoid = nn.Sigmoid() # output probabilities
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): #ToDo modify input
        h0 = self.init_hidden_state(x.size(0), self.hidden_size, self.num_layers).to(x.device)
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) # get last timestep output
        return out
    
    def init_hidden_state(batch_size, hidden_size, num_layers, qubo_matrix=None):
        hidden_state = torch.zeros(num_layers, batch_size, hidden_size)
        if qubo_matrix is not None: # conditioned initialization
            pass
        return hidden_state
    


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        h0 = self.init_hidden_state(x.size(0), self.hidden_size, self.num_layers).to(x.device)
        c0 = self.init_context(x.size(0), self.hidden_size, self.num_layers).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    
    def init_hidden_state(batch_size, hidden_size, num_layers, qubo_matrix=None):
        hidden_state = torch.zeros(num_layers, batch_size, hidden_size)
        if qubo_matrix is not None: # conditioned initialization
            pass
        return hidden_state
    
    def init_context(batch_size, hidden_size, num_layers):  
        return torch.zeros(num_layers, batch_size, hidden_size)
    
