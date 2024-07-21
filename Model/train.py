import torch
import torch.nn as nn
from torch_geometric.data import Data

#here we define a train loop to manage only a single sample for simplicity. Then we will expand it to manage a batch of samples.
def train_single_sample(data, model, optimizer, criterion, teacher_forcing_ratio=1):
    model.train()  # set the model to training mode
    window_size = model.window_size
    
    graph_data, Q = data # data is a tuple of graph and qubo data
    num_qubo_variables = Q.shape[1]
    X, A = graph_data.x, graph_data.edge_index

    # inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad() # clear the gradients of all optimized variables

    h_0 = model.init_hidden_state(Q)
    h_t = h_0
    graph_state_t = torch.zeros(1, window_size, X.shape[1])

    # Initialize graph data as empty
    graph_data_pred = Data()
    
    total_loss = 0
    for t in range(graph_data.x.shape[0]): 
        layer = range(num_qubo_variables)

        for qubit in layer:
            current_info = (graph_data_pred, qubit, layer, num_qubo_variables)

            graph_state_t, h_t, out = model(graph_state_t, h_t)

            gate_pred = model.sample_gate_type(out)
            graph_data_pred = model.update_graph_data(graph_data_pred, gate_pred, current_info)

            # # Compute and accumulate loss over gen_params
            # # loss = criterion(gen_params, target_params)  # Define `target_params` appropriately
            # # total_loss += loss

            # Determine whether to use teacher forcing for the next time step
            if model.update_graph_at_each_addition:
                if torch.rand(1).item() < teacher_forcing_ratio: # use ground truth as input
                    graph_state_t = X[t+1-window_size:t, :] if t+1 >= window_size else torch.cat([torch.zeros(window_size - t+1, X.shape[1]), X[:t+1, :]], dim=0)
                else: # else use model's own prediction as next input
                    graph_state_t = graph_data_pred.x[-window_size:, :] if t+1 >= window_size else torch.cat([torch.zeros(t+1 - window_size, X.shape[1]), graph_data_pred.x], dim=0)
            
        if model.update_hidden_at_each_layer:
            h_t = model.f_graph_RNN(graph_state_t, h_t)
            
        if model.update_graph_at_each_layer:
            # Determine whether to use teacher forcing for the next time step
            if torch.rand(1).item() < teacher_forcing_ratio: # use ground truth as input
                graph_state_t = X[t+1-window_size:t, :] if t+1 >= window_size else torch.cat([torch.zeros(window_size - t+1, X.shape[1]), X[:t+1, :]], dim=0)
            else: # else use model's own prediction as next input
                graph_state_t = graph_data_pred.x[-window_size:, :] if t+1 >= window_size else torch.cat([torch.zeros(t+1 - window_size, X.shape[1]), graph_data_pred.x], dim=0)


        stop_param = model.f_stopper(graph_state_t, h_t, h_0)
        stop_token = model.sample_stop_action(stop_param)

        # Compute and accumulate loss over stop_param
        # stop_loss = criterion(stop_param, target_stop_param)  
        # total_loss += stop_loss

    
    loss.backward()
    optimizer.step()