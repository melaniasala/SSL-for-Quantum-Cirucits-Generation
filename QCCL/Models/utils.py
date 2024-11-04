from . import GCNFeatureExtractor, BYOLOnlineNet, BYOLTargetNet, BYOLWrapper, SimCLRWrapper
import torch.nn as nn

def build_model(gnn_input_size, embedding_size, model_type='simclr', num_layers=5, proj_output_size=None, hidden_size=512, conv_type='GCNConv', global_node=False, device='cuda'):
    if proj_output_size is None:
        proj_output_size = embedding_size//2 

    print(f"\nBuilding model with {num_layers} {conv_type} layers and projection size {proj_output_size}...")

    if conv_type == 'GCNConv':
        gnn = GCNFeatureExtractor(gnn_input_size, embedding_size, pooling_strategy='global_avg', num_layers=num_layers, add_global_node=global_node)
    else:
        raise ValueError(f"Invalid convolution type: {conv_type}")

    if model_type == 'byol':
        projector = nn.Sequential(nn.Linear(embedding_size, hidden_size), 
                                  nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size, proj_output_size))
        predictor = nn.Sequential(nn.Linear(proj_output_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, proj_output_size))
        online_model = BYOLOnlineNet(gnn, projector, predictor)

        target_gnn = GCNFeatureExtractor(gnn_input_size, embedding_size, pooling_strategy='global_avg', num_layers=num_layers)
        target_projector = nn.Sequential(nn.Linear(embedding_size, hidden_size), 
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size, proj_output_size))
        target_model = BYOLTargetNet(target_gnn, target_projector)

        # deactivate requires_grad for the target encoder
        for param in target_model.parameters():
            param.requires_grad = False

        model = BYOLWrapper(online_model, target_model)
        print("BYOL model built successfully.")

    elif model_type == 'simclr':
        projector = nn.Sequential(nn.Linear(embedding_size, hidden_size), 
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size, proj_output_size))
        model = SimCLRWrapper(gnn, projector)
        print("SimCLR model built successfully.")

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.to(device)
    return model