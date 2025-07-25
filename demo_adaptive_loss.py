"""
Simplified demo of the Adaptive Loss Function for Gradient Descent
This script demonstrates the core concepts with minimal dependencies
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleAdaptiveLoss(nn.Module):
    """Simplified version of the adaptive loss function"""
    
    def __init__(self, num_features=2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weights = torch.tensor([0.1, 0.05], requires_grad=False)
        self.min_weight, self.max_weight = 0.01, 0.5
        
    def forward(self, pred, target, features):
        # Base MSE loss
        mse_loss = self.mse(pred, target)
        
        # Correlation losses for first two features
        corr_loss = 0
        for i in range(min(2, features.shape[1])):
            # Simple covariance approximation
            feat_centered = features[:, i] - features[:, i].mean()
            target_centered = target.squeeze() - target.squeeze().mean()
            cov = torch.abs(torch.mean(feat_centered * target_centered))
            corr_loss += self.weights[i] * cov
        
        return mse_loss + corr_loss
    
    def update_weights(self, features, target):
        """Simplified weight update"""
        with torch.no_grad():
            for i in range(min(2, features.shape[1])):
                feat_centered = features[:, i] - features[:, i].mean()
