import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from .losses import NTXentLoss


def validate(model, val_loader, loss_fun, device='cuda'):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for graph1, graph2 in val_loader:
            graph1, graph2 = graph1.to(device), graph2.to(device)
            # print("Graphs collected for validation.")

            # if model class is BYOLWrapper
            if model.__class__.__name__ == 'BYOLWrapper':
                z1_online = model.online_model(graph1)
                z2_online = model.online_model(graph2)
                z1_target = model.target_model(graph1)
                z2_target = model.target_model(graph2)
                loss = 0.5*(loss_fun(z1_online, z2_target) + loss_fun(z2_online, z1_target))

            else:
                z1 = model(graph1)
                z2 = model(graph2)
                
                loss = loss_fun(z1, z2)

            total_loss += loss.item()
            # print("Loss added to total validation loss.")
            
            # Accumulate total loss and sample count
            batch_size = graph1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples  # Mean loss per sample



def train_with_profiling(model, loss, train_dataset, val_dataset=None, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda', 
                         ema_alpha=1.0, patience=None, restore_best=False, verbose=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = loss(tau)
    ema_loss = False if ema_alpha == 1.0 else True

    history = {
        'train_loss': [], 
        'val_loss': [] if val_dataset is not None else None,
        'grad_norm_l2': [],
        'avg_grad_norm_l1_per_param': []
    }
    if ema_loss:
        history['ema_val_loss'] = []

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if patience is None or patience == 'None':
        patience = epochs
  
    
    def train_step():
        model.train()
        total_loss = 0
        total_grad_norm_l2 = 0 
        total_grad_norm_l1 = 0 
        total_samples = 0
        num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for graph1, graph2 in train_loader:
            with record_function("data_loading"):
                graph1, graph2 = graph1.to(device), graph2.to(device)
            optimizer.zero_grad()

            with record_function("model_forward"):
                z1 = model(graph1)
                z2 = model(graph2)

            with record_function("loss_computation"):
                loss = nt_xent_loss(z1, z2, return_scores=False)
            loss.backward()

            grad_norm_l2 = sum(param.grad.norm(2).item() ** 2 for param in model.parameters() if param.grad is not None)
            grad_norm_l1 = sum(param.grad.norm(1).item() for param in model.parameters() if param.grad is not None)
            total_grad_norm_l2 += grad_norm_l2
            total_grad_norm_l1 += grad_norm_l1

            optimizer.step()

            batch_size = graph1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        total_grad_norm_l2 = np.sqrt(total_grad_norm_l2)
        avg_grad_norm_l1_per_param = total_grad_norm_l1 / num_params_model
        history['grad_norm_l2'].append(total_grad_norm_l2)
        history['avg_grad_norm_l1_per_param'].append(avg_grad_norm_l1_per_param)

        return total_loss / total_samples  
    
    def validate_step():
        return validate(model, val_loader, nt_xent_loss, device)
    
    def update_history():
        history['train_loss'].append(total_train_loss)
        if val_dataset is not None:
            val_loss = validate_step()
            history['val_loss'].append(val_loss)
            if ema_loss:
                prev_ema = history['ema_val_loss'][-1] if history['ema_val_loss'] else val_loss
                ema_val_loss = ema_alpha * val_loss + (1.0 - ema_alpha) * prev_ema
                history['ema_val_loss'].append(ema_val_loss)
        return

    # Begin profiling the training process
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True, profile_memory=True) as prof:
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch', disable=not verbose) as pbar:
                total_train_loss = train_step()
                pbar.update(1)
            update_history()

            if verbose:
                print(f"\t - loss: {total_train_loss:.4f} - grad_norm_l2: {history['grad_norm_l2'][-1]:.4f}", end="")
                if val_dataset is not None:
                    print(f" - val_loss: {history['val_loss'][-1]:.4f}", end="")
                    if ema_loss:
                        print(f" - ema_val_loss: {history['ema_val_loss'][-1]:.4f}")
                    else:
                        print("\n")
                        
            prof.step()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    return history




def train(model, train_dataset, val_dataset=None, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda', 
          ema_alpha=1.0, patience=None, restore_best=False, verbose=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # print("Train loader created.")
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = NTXentLoss(tau)
    ema_loss = False if ema_alpha == 1.0 else True

    print("Starting training...")
    history = {
        'train_loss': [], 
        'val_loss': [] if val_dataset is not None else None,
        'grad_norm_l2': [],
        'avg_grad_norm_l1_per_param': []
    }
    if ema_loss:
        history['ema_val_loss'] = []

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # print("Validation loader created.")

    # Initialize variables for early stopping
    if patience is None or patience == 'None':
        patience = epochs
    else:
        restore_best = True
    best_val_loss = np.inf
    patience_counter = 0  
    best_model_state = None  
    
    def train_step():
        # print("Training step started.")
        # for each epoch
        model.train()
        total_loss = 0
        total_grad_norm_l2 = 0 
        total_grad_norm_l1 = 0 
        total_samples = 0

        #extract the number of parameters in the model
        num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        for graph1, graph2 in train_loader: # for each batch
            graph1, graph2 = graph1.to(device), graph2.to(device)

            # print("Collected graphs")

            optimizer.zero_grad()

            # print("Optimizer zeroed")

            z1 = model(graph1)
            z2 = model(graph2)

            # print("Model forward pass done")

            # compute loss
            loss = nt_xent_loss(z1, z2, return_scores=False)
            loss.backward()
            # print("Loss backward pass done")

            # compute gradient norm (L2 norm)
            grad_norm_l2 = 0
            grad_norm_l1 = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_l2 += param.grad.norm(2).item() ** 2  
                    grad_norm_l1 += param.grad.norm(1).item()
            total_grad_norm_l2 += grad_norm_l2
            total_grad_norm_l1 += grad_norm_l1

            optimizer.step()
            # print("Optimizer step done")

            # Accumulate total loss and sample count
            batch_size = graph1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        total_grad_norm_l2 = np.sqrt(total_grad_norm_l2)
        avg_grad_norm_l1_per_param = total_grad_norm_l1 / num_params_model
        history['grad_norm_l2'].append(total_grad_norm_l2)
        history['avg_grad_norm_l1_per_param'].append(avg_grad_norm_l1_per_param)

        # print("Training step done.")

        return total_loss / total_samples  
    
    def validate_step():
        # print("Validation step started.")
        v=validate(model, val_loader, nt_xent_loss, device)
        # print("Validation step done.")
        return v
    
    def update_history():
        history['train_loss'].append(total_train_loss)
        if val_dataset is not None:
            val_loss = validate_step()
            history['val_loss'].append(val_loss)

            if ema_loss:
                prev_ema = history['ema_val_loss'][-1] if history['ema_val_loss'] else val_loss
                ema_val_loss = ema_alpha * val_loss + (1.0 - ema_alpha) * prev_ema
                history['ema_val_loss'].append(ema_val_loss)
        # print("History updated")
        return

    if verbose:
        for epoch in range(epochs):
            # print(f"Epoch {epoch+1}/{epochs}")
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch', disable=not verbose) as pbar:
                total_train_loss = train_step()
                pbar.update(1)

            update_history()

            print(f"\t - loss: {total_train_loss:.4f} - grad_norm_l2: {history['grad_norm_l2'][-1]:.4f}", end="")
            if val_dataset is not None:
                print(f" - val_loss: {history['val_loss'][-1]:.4f}", end="")
                if ema_loss:
                    print(f" - ema_val_loss: {history['ema_val_loss'][-1]:.4f}")
                else:
                    print("\n")

                curr_ema_val_loss = history['ema_val_loss'][-1] if ema_loss else history['val_loss'][-1]
                # check for validation loss improvement
                if curr_ema_val_loss < best_val_loss:
                    best_val_loss = curr_ema_val_loss
                    patience_counter = 0  # reset patience
                    best_model_state = model.state_dict() 
                else:
                    patience_counter += 1

                # early stopping
                if patience_counter >= patience:
                    print("Early stopping due to no improvement in validation loss. Epochs run: ", epoch+1)
                    break

    else:
        with tqdm(total=epochs, desc="Training", unit='epoch', disable=verbose) as pbar:
            for epoch in range(epochs):
                total_train_loss = train_step()
                pbar.update(1)
                pbar.set_postfix({'loss': f"{total_train_loss:.4f}"})
                update_history()

                if val_dataset is not None:
                    if ema_loss:
                        pbar.set_postfix({
                            'loss': f"{total_train_loss:.4f}",
                            'val_loss': f"{history['val_loss'][-1]:.4f}",
                            'ema_val_loss': f"{history['ema_val_loss'][-1]:.4f}"
                        })
                    else:
                        pbar.set_postfix({
                            'loss': f"{total_train_loss:.4f}",
                            'val_loss': f"{history['val_loss'][-1]:.4f}"
                        })

                    curr_ema_val_loss = history['ema_val_loss'][-1] if ema_loss else history['val_loss'][-1]
                    # check for validation loss improvement
                    if curr_ema_val_loss < best_val_loss:
                        best_val_loss = curr_ema_val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict() 
                    else:
                        patience_counter += 1

                    # early stopping
                    if patience_counter >= patience:
                        print("Early stopping due to no improvement in validation loss. Epochs run: ", epoch+1)
                        break
    
    # restore the best model parameters after training
    if best_model_state is not None and restore_best:
        print("Restoring model to the state with the best validation loss.")
        model.load_state_dict(best_model_state)

    return history


def train_byol(model, train_dataset, val_dataset=None, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda', 
          ema_alpha=1.0, patience=None, warmup=0, restore_best=False, verbose=True, compute_grads=False):
    
    online_model = model.online_model
    target_model = model.target_model

    ema_loss = False if ema_alpha == 1.0 else True
    print("EMA Loss: ", ema_loss)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = Adam(online_model.parameters(), lr=lr, weight_decay=1e-1)
    mse_loss = MSELoss()
    # cosine_similarity = CosineSimilarity()
    # mse_loss = lambda x, y: torch.mean(1 - cosine_similarity(x, y))
    # print("Using Cosine Similarity loss for BYOL training.") 

    history = {
        'train_loss': [], 
        'val_loss': [] if val_dataset is not None else None,
        'params_norm': {} # Track the L2 norm of the model parameters here
    }
    if ema_loss:
        history['ema_val_loss'] = []
    if compute_grads:
        history['grad_norm_l2'] = []
        history['avg_grad_norm_l1_per_param'] = []

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize variables for early stopping
    if patience is None or patience == 'None':
        patience = epochs
    else:
        restore_best = True
    best_val_loss = np.inf
    patience_counter = 0  
    best_model_state = None  
    
    def train_step():
        # for each epoch
        online_model.train()
        target_model.eval()
        total_loss = 0
        total_grad_norm_l2 = 0
        total_grad_norm_l1 = 0
        
        for graph1, graph2 in train_loader: # for each batch
            graph1, graph2 = graph1.to(device), graph2.to(device)

            update_target(online_model, target_model, tau)
            optimizer.zero_grad()

            z1_online = online_model(graph1)
            z2_online = online_model(graph2)

            with torch.no_grad():
                z1_target = target_model(graph1)
                z2_target = target_model(graph2) # target model won't be updated through backprop

            # compute loss
            loss = 0.5*(mse_loss(z1_online, z2_target) + mse_loss(z2_online, z1_target))

            loss.backward()

            if compute_grads:
                grad_norm_l2 = 0
                grad_norm_l1 = 0
                for param in online_model.parameters():
                    if param.grad is not None:
                        grad_norm_l2 += param.grad.norm(2).item() ** 2  
                        grad_norm_l1 += param.grad.norm(1).item()
                total_grad_norm_l2 += grad_norm_l2
                total_grad_norm_l1 += grad_norm_l1
            
            optimizer.step()
            total_loss += loss.item()

        if compute_grads:
            total_grad_norm_l2 = np.sqrt(total_grad_norm_l2)
            avg_grad_norm_l1_per_param = total_grad_norm_l1 / len(list(online_model.parameters()))
            history['grad_norm_l2'].append(total_grad_norm_l2)
            history['avg_grad_norm_l1_per_param'].append(avg_grad_norm_l1_per_param)

        return total_loss / len(train_loader)
    
    def validate_step():
        return validate(model, val_loader, mse_loss, device)
    
    def update_history():
        history['train_loss'].append(total_train_loss)
        if val_dataset is not None:
            val_loss = validate_step()
            history['val_loss'].append(val_loss)

            if ema_loss:
                prev_ema = history['ema_val_loss'][-1] if history['ema_val_loss'] else val_loss
                ema_val_loss = ema_alpha * val_loss + (1.0 - ema_alpha) * prev_ema
                history['ema_val_loss'].append(ema_val_loss)
        
        for name, param in online_model.named_parameters():
            param_norm = param.norm(2).item()
            if name not in history['params_norm']:
                history['params_norm'][name] = []
            history['params_norm'][name].append(param_norm)
    
    if verbose:
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch', disable=not verbose) as pbar:
                total_train_loss = train_step()
                pbar.update(1)

            update_history()

            print(f"\t - loss: {total_train_loss:.4f}", end="")
            if val_dataset is not None:
                print(f" - val_loss: {history['val_loss'][-1]:.4f}", end="")

                curr_val_loss = history['ema_val_loss'][-1] if ema_loss else history['val_loss'][-1]
                # check for validation loss improvement
                if epoch >= warmup:
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        patience_counter = 0  # reset patience
                        best_model_state = model.state_dict() 
                    else:
                        patience_counter += 1

                # early stopping
                if patience_counter >= patience:
                    print("Early stopping due to no improvement in validation loss. Epochs run: ", epoch+1)
                    break

    else:
        with tqdm(total=epochs, desc="Training", unit='epoch', disable=verbose) as pbar:
            for epoch in range(epochs):
                total_train_loss = train_step()
                pbar.update(1)
                pbar.set_postfix({'loss': f"{total_train_loss:.4f}"})
                update_history()

                if val_dataset is not None:
                    if ema_loss:
                        pbar.set_postfix({
                            'loss': f"{total_train_loss:.4f}",
                            'val_loss': f"{history['val_loss'][-1]:.4f}",
                            'ema_val_loss': f"{history['ema_val_loss'][-1]:.4f}"
                        })
                    else:
                        pbar.set_postfix({
                            'loss': f"{total_train_loss:.4f}",
                            'val_loss': f"{history['val_loss'][-1]:.4f}"
                        })

                    curr_val_loss = history['ema_val_loss'][-1] if ema_loss else history['val_loss'][-1]
                    # check for validation loss improvement
                    if epoch >= warmup:
                        if curr_val_loss < best_val_loss:
                            best_val_loss = curr_val_loss
                            patience_counter = 0
                            best_model_state = model.state_dict() 
                        else:
                            patience_counter += 1

                    # early stopping
                    if patience_counter >= patience:
                        print("Early stopping due to no improvement in validation loss. Epochs run: ", epoch+1)
                        break
    
    # restore the best model parameters after training
    if best_model_state is not None and restore_best:
        print("Restoring model to the state with the best validation loss.")
        model.load_state_dict(best_model_state)

    return history
    


def update_target(online_model, target_model, tau):
    for online_params, target_params in zip(online_model.gnn.parameters(), target_model.gnn.parameters()):
        target_params.data = tau * target_params.data + (1.0 - tau) * online_params.data
    for online_params, target_params in zip(online_model.projector.parameters(), target_model.projector.parameters()):
        target_params.data = tau * target_params.data + (1.0 - tau) * online_params.data
    return