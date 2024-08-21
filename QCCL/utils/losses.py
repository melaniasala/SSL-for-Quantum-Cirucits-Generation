import torch
import torch.nn.functional as F
from torch import nn
    

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, print_flag=False):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)

        if print_flag: 
            print("Input representations:\n", representations)
        
        # Compute the cosine similarity matrix for all pairs of embeddings
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        batch_size = z1.size(0)
        
        # Create a mask to remove self-similarities (main diagonal)
        self_mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z1.device)  # Shape: (2N, 2N)
        
        # Create a mask to identify positive pairs
        # For each embedding in the first half, the corresponding positive is in the second half, and vice versa
        positive_mask = torch.zeros_like(self_mask)  # Shape: (2N, 2N)
        positive_mask[:batch_size, batch_size:] = torch.eye(batch_size)  # First half positive pairs
        positive_mask[batch_size:, :batch_size] = torch.eye(batch_size)  # Second half positive pairs
        
        # Create a mask to identify negative pairs by excluding self-similarities and positive pairs
        combined_mask = self_mask | positive_mask 
        
        negatives = similarity_matrix[~combined_mask].view(batch_size * 2, -1)  # Shape: (2N, 2N-2)
        positives = similarity_matrix[positive_mask].view(batch_size * 2, 1)  # Shape: (2N, 1)
        if print_flag:
            print("Positive pairs scores:\n", positives)
            print("Negative pairs scores:\n", negatives)

        loss_pos_pairs = -torch.log(torch.exp(positives / self.temperature) / 
                                    (torch.exp(positives / self.temperature) + torch.sum(torch.exp(negatives / self.temperature), dim=1)))
        
        return torch.mean(loss_pos_pairs)
        
        # # Each row in logits: [positive_similarity, negative_similarity_1, negative_similarity_2, ...]
        # logits = torch.cat([positives, negatives], dim=1)  # Shape: (2N, 2N-1)
        
        # # Create labels (indices of correct class, the correct class is the first column in logits, zero index)
        # labels = torch.zeros(batch_size * 2).long().to(z1.device)  # Shape: (2N,)
        
        # return F.cross_entropy(logits / self.temperature, labels)
