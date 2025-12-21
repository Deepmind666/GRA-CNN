"""
Gray Relational Analysis (GRA) Scorer for Channel Pruning.
This module implements the core GRA scoring logic that measures the 
semantic alignment between feature maps and classification logits.
"""

import torch
import torch.nn as nn

class GrayRelationalChannelScorer:
    """
    Computes importance scores for CNN channels based on Gray Relational Analysis.
    
    Attributes:
        rho (float): The distinguishing coefficient, typically in [0.4, 0.6].
                     Balances between local granularity and global stability.
    """
    def __init__(self, rho=0.5):
        self.rho = rho

    def normalize(self, x):
        """Perform min-max normalization along the batch dimension."""
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val < 1e-8] = 1.0 
        return (x - min_val) / range_val

    def compute_score(self, feature_maps, logits, targets):
        """
        Compute relational grade between activations and decision-level logits.
        
        Args:
            feature_maps (Tensor): Activations of shape [B, C, H, W] or [B, C].
            logits (Tensor): Output logits of shape [B, num_classes].
            targets (Tensor): Ground truth labels of shape [B].
            
        Returns:
            gra_scores (Tensor): Relational grade for each channel [C].
        """
        B = feature_maps.size(0)
        
        # Global Average Pooling for spatial feature maps
        if feature_maps.dim() == 4:
            f = feature_maps.mean(dim=[2, 3])
        else:
            f = feature_maps

        # Extract logits corresponding to the ground truth class (reference sequence)
        reference = logits.gather(1, targets.view(-1, 1)).squeeze().view(1, -1) # [1, B]
        comparison = f.t() # [C, B]
        
        # Normalize sequences
        ref_norm = self.normalize(reference)
        comp_norm = self.normalize(comparison)
        
        # Compute difference sequence
        delta = torch.abs(comp_norm - ref_norm)
        delta_min = delta.min().item()
        delta_max = delta.max().item()
        
        # Compute Gray Relational Coefficient
        rho_max = self.rho * delta_max
        gamma = (delta_min + rho_max) / (delta + rho_max + 1e-8)
        
        # Average across the batch to get the Relational Grade
        gra_scores = gamma.mean(dim=1)
        
        return gra_scores
