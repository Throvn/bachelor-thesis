import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Sigmoid to convert logits to probabilities
        probs = torch.where(targets == 1, inputs, 1 - inputs)  # p_t

        # Apply the focal loss formula
        loss = self.alpha * (1.0 - probs) ** self.gamma * BCE_loss
        
        # Reduction method (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Example usage:
# Initialize loss function
# focal_loss = FocalLoss(alpha=0.77, gamma=2.0)