import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from .losses import NTXentLoss

def train(model, dataset, epochs=100, batch_size=32, lr=1e-3, tau=0.5, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = NTXentLoss(tau)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pf = True if epoch == epochs-1 else False
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
            for data in dataloader:
                graph1, graph2 = data

                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                optimizer.zero_grad()

                z1 = model(graph1.x, graph1.edge_index, graph1.batch)
                z2 = model(graph2.x, graph2.edge_index, graph2.batch)

                loss = nt_xent_loss(z1, z2, pf)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'Loss': total_loss / (pbar.n + 1)})
                pbar.update(1)
        
        print(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {total_loss/len(dataloader):.4f}")
