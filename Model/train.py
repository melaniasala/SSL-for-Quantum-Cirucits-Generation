import torch
import torch.nn as nn

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
    graph_state_t = torch.zeros(window_size, X.shape[1])
    
    loss = 0
    for t in range(graph_data.nodes): 
        layer = range(num_qubo_variables)

        for qubit in layer:
            graph_state_t, h_t, graph_data_pred, gen_params = model(graph_state_t, h_t, qubit, layer, num_qubo_variables)

            # compute loss over gen_params
            
        if model.update_graph_at_each_layer:
            graph_state_t = model.update_graph_state(X, A, window_size)
                                                      
        if model.update_hidden_at_each_layer:
            h_t = model.f_graph_RNN(h_t, graph_state_t)


        stop_param = Stopper(h_t, graph_data_t, h_0)
        stop_token = model.sample_stop_action(stop_param)

        # compute loss over stop_param
        
        # Determine whether to use teacher forcing
        if torch.rand(1).item() < teacher_forcing_ratio: # use ground truth as input
            graph_data_t = X[t+1-window_size:t, :] if t+1 >= window_size else torch.cat([torch.zeros(window_size - t+1, X.shape[1]), X[:t+1, :]], dim=0)
        else: # else use model's own prediction as next input
            graph_data_t = graph_data_pred.x[-window_size:, :] if t+1 >= window_size else torch.cat([torch.zeros(t+1 - window_size, X.shape[1]), graph_data_pred.x], dim=0)
    
    loss.backward()
    optimizer.step()