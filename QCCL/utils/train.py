import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from .losses import NTXentLoss

def train(model, dataset, epochs=100, batch_size=32, lr=1e-3, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    nt_xent_loss = NTXentLoss(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            graph1, graph2 = data
            
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            optimizer.zero_grad()
            
            z1 = model(graph1.x, graph1.edge_index)
            z2 = model(graph2.x, graph2.edge_index)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")