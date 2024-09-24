import torch.nn as nn

# wrapper class for CL model (nn.Sequential) is not compatible with GNN
class SimCLRWrapper(nn.Module):
    def __init__(self, gnn, projector=None):
        super(SimCLRWrapper, self).__init__()
        self.gnn = gnn
        self.projection_head = projector

    def forward(self, inputs):
        gnn_output = self.gnn(inputs)
        if self.projection_head is not None:
            return self.projection_head(gnn_output)
        return gnn_output
    
    def get_gnn(self):
        return self.gnn
    

# wrapper class for BYOL model
class BYOLOnlineNet(nn.Module):
    def __init__(self, gnn, projector, predictor):
        super(BYOLOnlineNet, self).__init__()
        self.gnn = gnn
        self.projector = projector
        self.predictor = predictor

    def forward(self, inputs):
        gnn_output = self.gnn(inputs)
        projected = self.projector(gnn_output)
        return self.predictor(projected)
    
class BYOLTargetNet(nn.Module):
    def __init__(self, gnn, projector):
        super(BYOLTargetNet, self).__init__()
        self.gnn = gnn
        self.projector = projector

    def forward(self, inputs):
        gnn_output = self.gnn(inputs)
        return self.projector(gnn_output)
    
class BYOLWrapper(nn.Module):
    def __init__(self, online_model, target_model):
        super(BYOLWrapper, self).__init__()
        self.online_model = online_model
        self.target_model = target_model

    def forward(self, inputs):
        return self.online_model(inputs), self.target_model(inputs)
    
    def get_gnn(self):
        return self.online_model.gnn
    