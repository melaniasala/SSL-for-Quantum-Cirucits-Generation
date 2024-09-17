import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from .losses import NTXentLoss

def validate(model, val_loader, loss_fun, device='cuda'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph1, graph2 in val_loader:
            graph1, graph2 = graph1.to(device), graph2.to(device)

            z1 = model(graph1)
            z2 = model(graph2)

            loss, _, _ = loss_fun(z1, z2)
            total_loss += loss.item()
    
    return total_loss



def train(model, train_dataset, val_dataset=None, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda', 
          ema_alpha=1.0, patience=None, restore_best=False, verbose=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = NTXentLoss(tau)
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
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables for early stopping
    if patience is None:
        patience = epochs
    else:
        restore_best = True
    best_val_loss = np.inf
    patience_counter = 0  
    best_model_state = None  
    
    def train_step():
        # for each epoch
        model.train()
        total_loss = 0
        total_grad_norm_l2 = 0 
        total_grad_norm_l1 = 0 

        #extract the number of parameters in the model
        num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        for graph1, graph2 in train_loader: # for each batch
            graph1, graph2 = graph1.to(device), graph2.to(device)

            optimizer.zero_grad()

            z1 = model(graph1)
            z2 = model(graph2)

            # compute loss
            loss, _, _ = nt_xent_loss(z1, z2)
            loss.backward()

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

            total_loss += loss.item()

        total_grad_norm_l2 = np.sqrt(total_grad_norm_l2)
        avg_grad_norm_l1_per_param = total_grad_norm_l1 / num_params_model
        history['grad_norm_l2'].append(total_grad_norm_l2)
        history['avg_grad_norm_l1_per_param'].append(avg_grad_norm_l1_per_param)

        return total_loss
    
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
                
        return val_loss

    if verbose:
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch', disable=not verbose) as pbar:
                total_train_loss = train_step()
                pbar.update(1)

            val_loss = update_history()

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
                val_loss = update_history()

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
        print(f"Restoring model to the state with the best validation loss.")
        model.load_state_dict(best_model_state)

    return history