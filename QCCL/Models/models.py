import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

from QCCL.utils import NTXentLoss
from QCCL.Models import GCNFeatureExtractor, SimCLRWrapper, BYOLWrapper, BYOLOnlineNet, BYOLTargetNet


class CLModel(nn.Module):
    def __init__(self, gnn_input_size, embedding_size, tau=0.5, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device="cuda"):
        """
        tau (float):    Exponential moving average decay rate for the target model in BYOL,
                        while it is the temperature parameter for NTXentLoss in SimCLR.
        """
        super().__init__()
        self.tau = tau
        self.device = device
        self.embedding_size = embedding_size
        self.model = self.build_model(gnn_input_size, embedding_size, num_layers, proj_output_size, hidden_size, conv_type, device)
    
    def build_model(self, gnn_input_size, embedding_size, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device='cuda'):
        NotImplementedError

    def forward(self, inputs):
        NotImplementedError

    def train(
        self,
        train_dataset,
        val_dataset=None,
        optimizer=None,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        patience=None,
        restore_best=False,
        ema_alpha=1.0,
        verbose=True,
    ):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size)
            if val_dataset is not None
            else None
        )

        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "val_loss": [] if val_loader is not None else None,
        }
        if ema_alpha < 1.0:
            history["ema_val_loss"] = []

        # Early stopping setup
        patience = patience if patience is not None else epochs
        best_val_loss = np.inf
        patience_counter = 0
        best_model_state = None

        with tqdm(total=epochs, desc="Training", unit="epoch", disable=verbose) as pbar:
            for epoch in range(epochs):
                total_train_loss = self.train_step(train_loader, optimizer)
                self.update_history(
                    history,
                    total_train_loss,
                    self.validate(val_loader) if val_loader else None,
                    ema_alpha
                )
                pbar.update(1)

                if verbose:
                    print(
                        f"\t- Epoch {epoch+1}/{epochs} - loss: {total_train_loss:.4f}",
                        end="",
                    )
                    if "val_loss" in history:
                        print(f" - val_loss: {history['val_loss'][-1]:.4f}", end="")
                    if "ema_val_loss" in history:
                        print(f" - ema_val_loss: {history['ema_val_loss'][-1]:.4f}")
                    else:
                        print()

                else:
                    if "val_loss" in history:
                        if "ema_val_loss" in history:
                            pbar.set_postfix(
                                {
                                    "loss": f"{total_train_loss:.4f}",
                                    "val_loss": f"{history['val_loss'][-1]:.4f}",
                                    "ema_val_loss": f"{history['ema_val_loss'][-1]:.4f}",
                                }
                            )
                        else:
                            pbar.set_postfix(
                                {
                                    "loss": f"{total_train_loss:.4f}",
                                    "val_loss": f"{history['val_loss'][-1]:.4f}",
                                }
                            )

                # Early stopping logic
                curr_val_loss = (
                    history["ema_val_loss"][-1]
                    if "ema_val_loss" in history
                    else history["val_loss"][-1]
                )
                if curr_val_loss < best_val_loss:
                    best_val_loss = curr_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping due to no improvement in validation loss.")
                    break

        # Restore the best model state
        if best_model_state is not None and restore_best:
            print("Restoring model to the state with the best validation loss.")
            self.model.load_state_dict(best_model_state)

        return history

    def train_step(self, train_loader, optimizer):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def update_history(self, history, train_loss, val_loss, ema_alpha=1.0):
        history["train_loss"].append(train_loss)
        if val_loss is not None:
            history["val_loss"].append(val_loss)
            if "ema_val_loss" in history:
                prev_ema = (
                    history["ema_val_loss"][-1] if history["ema_val_loss"] else val_loss
                )
                ema_val_loss = (
                    ema_alpha * val_loss + (1.0 - ema_alpha) * prev_ema
                )
                history["ema_val_loss"].append(ema_val_loss)




class SimCLR(CLModel):
    def __init__(self, gnn_input_size, embedding_size, tau=0.5, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device="cuda"):
        super().__init__(gnn_input_size, embedding_size, tau, num_layers, proj_output_size, hidden_size, conv_type, device)
        self.loss = NTXentLoss(self.tau)
    

    def build_model(self, gnn_input_size, embedding_size, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device='cuda'):
        if proj_output_size is None:
            proj_output_size = embedding_size // 2

        print(
            f"\nBuilding SimCLR model with {num_layers} {conv_type} layers and projection size {proj_output_size}..."
        )

        if conv_type == "GCNConv":
            gnn = GCNFeatureExtractor(
                gnn_input_size, embedding_size, pooling_strategy="global_avg", num_layers=num_layers
            )
        else:
            raise ValueError(f"Invalid convolution type: {conv_type}")

        projector = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_output_size),
        )

        model = SimCLRWrapper(gnn, projector).to(device)

        return model
    
    def forward(self, inputs):
        return self.model(inputs)

    def train_step(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        for graph1, graph2 in train_loader:
            graph1, graph2 = graph1.to(self.device), graph2.to(self.device)
            optimizer.zero_grad()
            z1 = self.model(graph1)
            z2 = self.model(graph2)
            loss = self.loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for graph1, graph2 in val_loader:
                graph1, graph2 = graph1.to(self.device), graph2.to(self.device)
                z1 = self.model(graph1)
                z2 = self.model(graph2)
                loss = self.loss(z1, z2)
                total_loss += loss.item()
        return total_loss / len(val_loader)


class BYOL(CLModel):
    def __init__(self, gnn_input_size, embedding_size, tau=0.5, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device="cuda"):
        super().__init__(gnn_input_size, embedding_size, tau, num_layers, proj_output_size, hidden_size, conv_type, device)
        self.loss = MSELoss()
        self.online_model = self.model.online_model
        self.target_model = self.model.target_model

    def build_model(self, gnn_input_size, embedding_size, num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', device='cuda'):
        if proj_output_size is None:
            proj_output_size = embedding_size // 2

        print(
            f"\nBuilding BYOL model with {num_layers} {conv_type} layers and projection size {proj_output_size}..."
        )

        if conv_type == "GCNConv":
            gnn = GCNFeatureExtractor(
                gnn_input_size, embedding_size, pooling_strategy="global_avg", num_layers=num_layers
            )
        else:
            raise ValueError(f"Invalid convolution type: {conv_type}")

        projector = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_output_size),
        )
        predictor = nn.Sequential(
            nn.Linear(proj_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_output_size),
        )

        online_model = BYOLOnlineNet(gnn, projector, predictor)

        target_gnn = GCNFeatureExtractor(
            gnn_input_size, embedding_size, pooling_strategy="global_avg", num_layers=num_layers
        )
        target_projector = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_output_size),
        )
        target_model = BYOLTargetNet(target_gnn, target_projector)

        model = BYOLWrapper(online_model, target_model).to(device)

        return model
    
    def forward(self, inputs):
        return self.model(inputs)

    def train_step(self, train_loader, optimizer):
        self.model.online_model.train()
        self.model.target_model.eval()
        total_loss = 0
        for graph1, graph2 in train_loader:
            graph1, graph2 = graph1.to(self.device), graph2.to(self.device)
            optimizer.zero_grad()
            z1_online = self.model.online_model(graph1)
            z2_online = self.model.online_model(graph2)
            with torch.no_grad():
                z1_target = self.model.target_model(graph1)
                z2_target = self.model.target_model(graph2)
            loss = self.loss(z1_online, z2_target) + self.loss(z2_online, z1_target)
            loss.backward()
            optimizer.step()
            self.update_target()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.online_model.eval()
        self.model.target_model.eval()
        total_loss = 0
        with torch.no_grad():
            for graph1, graph2 in val_loader:
                graph1, graph2 = graph1.to(self.device), graph2.to(self.device)
                z1_online = self.model.online_model(graph1)
                z2_online = self.model.online_model(graph2)
                z1_target = self.model.target_model(graph1)
                z2_target = self.model.target_model(graph2)
                loss = self.loss(z1_online, z2_target) + self.loss(z2_online, z1_target)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def update_target(self):
        for online_params, target_params in zip(self.model.online_model.gnn.parameters(), self.model.target_model.gnn.parameters()):
            target_params.data = target_params.data * self.tau + online_params.data * (1.0 - self.tau)
        for online_params, target_params in zip(self.model.online_model.projector.parameters(), self.model.target_model.projector.parameters()):
            target_params.data = target_params.data * self.tau + online_params.data * (1.0 - self.tau)


def compute_grad_norms(model, loss):
    grad_norm_l2 = 0
    grad_norm_l1 = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_l2 += param.grad.norm(2).item() ** 2
            grad_norm_l1 += param.grad.norm(1).item()
    return grad_norm_l2, grad_norm_l1
