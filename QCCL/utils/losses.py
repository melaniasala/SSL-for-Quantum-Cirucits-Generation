import torch
import torch.nn.functional as F
from torch import nn
    

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, print_flag=False, return_scores=False):
        z1 = F.normalize(z1, dim=1) # Shape: (N, D) where N is the batch size and D is the dimension of the embeddings
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0) # Shape: (2N, D)
        N = z1.size(0)

        # if print_flag: 
        #     print("Input representations:\n", representations)
        
        # Compute the cosine similarity matrix for all pairs of embeddings
        similarity_matrix = representations @ representations.t() # Shape: (2N, 2N) 
        
        # Create a mask to remove self-similarities (main diagonal)
        self_mask = torch.eye(N * 2, dtype=torch.bool).to(z1.device)  # Shape: (2N, 2N)
        
        # Create a mask to identify positive pairs
        # For each embedding in the first half, the corresponding positive is in the second half, and vice versa
        positive_mask = torch.zeros_like(self_mask)  # Shape: (2N, 2N)
        positive_mask[:N, N:] = torch.eye(N)  # First half positive pairs
        positive_mask[N:, :N] = torch.eye(N)  # Second half positive pairs
        
        # Create a mask to identify negative pairs by excluding self-similarities and positive pairs
        combined_mask = self_mask | positive_mask 
        
        negatives = similarity_matrix[~combined_mask].view(N * 2, -1)  # Shape: (2N, 2N-2)
        positives = similarity_matrix[positive_mask].view(N * 2, 1)  # Shape: (2N, 1)
        if print_flag:
            print("Positive pairs scores:\n", positives)
            print("Negative pairs scores:\n", negatives)

        loss_pos_pairs = -torch.log(torch.exp(positives / self.temperature) / 
                                    (torch.exp(positives / self.temperature) + torch.sum(torch.exp(negatives / self.temperature), dim=1).view(2*N, -1)))
        if print_flag: 
            print('losses:', loss_pos_pairs)
            print('temp:', self.temperature)
            print('N:', N)

        if return_scores:
            return torch.mean(loss_pos_pairs), positives, negatives
        
        return torch.mean(loss_pos_pairs)
    

# class MSEByolLoss(nn.Module):
#     def __init__(self):
#         super(MSEByolLoss, self).__init__()
    
#     def forward(self, z1, z2):
#         return F.mse_loss(z1, z2)
    
