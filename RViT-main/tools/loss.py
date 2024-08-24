import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, k=3):
        super(CustomLoss, self).__init__()
        self.k = k
        self.weights = torch.tensor([1.0, 0.7, 0.5])[:k]

    def forward(self, outputs, targets):
        # Compute the Softmax probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get the top-k predictions and their indices
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=1)
        
        # Adjust weights to match the device of outputs
        weights = self.weights.to(outputs.device)
        
        # Check if the target is within the top-k predictions
        correct_in_topk = (topk_indices == targets.unsqueeze(1))
        
        # If the correct label is not in the top-k, calculate the loss based on the max probability
        if not correct_in_topk.any():
            # Use negative log-likelihood as a penalty
            loss = -torch.log(probs[torch.arange(len(targets)), targets])
        else:
            # Calculate the weighted loss: apply weights only to positions where the correct label is in top-k
            loss = -torch.mean(torch.sum(correct_in_topk.float() * torch.log(topk_probs) * weights, dim=1))
        
        return loss.mean()