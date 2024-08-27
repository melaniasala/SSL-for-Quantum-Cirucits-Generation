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


def train(model, train_dataset, val_dataset=None, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda', verbose=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = NTXentLoss(tau)
    history = {'train_loss': [], 'val_loss': [] if val_dataset is not None else None}
    
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the training and validation steps, as well as the history update step
    def train_step():
        model.train()
        total_loss = 0
        for graph1, graph2 in train_loader:
            graph1, graph2 = graph1.to(device), graph2.to(device)

            optimizer.zero_grad()

            z1 = model(graph1)
            z2 = model(graph2)

            loss, _, _ = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss
    
    def validate_step():
        return validate(model, val_loader, nt_xent_loss, device)
    
    def update_history():
        history['train_loss'].append(total_train_loss)
        if val_dataset is not None:
            val_loss = validate_step()
            history['val_loss'].append(val_loss)
    

    if verbose:
            for epoch in range(epochs):
                with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch', disable=not verbose) as pbar:
                    total_train_loss = train_step()
                    pbar.update(1)

                update_history()

                print(f"\t - loss: {total_train_loss:.4f}", end="")
                if val_dataset is not None:
                    print(f" - val_loss: {history['val_loss'][-1]:.4f}\n")
                else:
                    print("\n")


    else:
        with tqdm(total=epochs, desc="Training", unit='epoch', disable=verbose) as pbar:
            for epoch in range(epochs):
                total_train_loss = train_step()
                pbar.update(1)
                pbar.set_postfix({'loss': f"{total_train_loss:.4f}"})

                update_history()

                if val_dataset is not None:
                    pbar.set_postfix({'loss': f"{total_train_loss:.4f}", 'val_loss': f"{history['val_loss'][-1]:.4f}"})
    
    return history