import torch
import torch.nn as nn
import numpy as np

class GrayRelationalChannelScorer:
    def __init__(self, rho=0.5):
        """
        Args:
            rho (float): Resolution coefficient, typically 0.5.
        """
        self.rho = rho

    def normalize(self, x):
        """
        Min-Max normalization to [0, 1] for each sequence.
        Args:
            x: Tensor of shape [M, N] where M is number of sequences, N is sequence length.
        Returns:
            Normalized tensor of shape [M, N].
        """
        # Min/Max across the sequence dimension (dim=1)
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        
        range_val = max_val - min_val
        # Avoid division by zero by replacing 0 with a small epsilon or 1 (if range is 0, val is 0)
        range_val[range_val < 1e-8] = 1.0 
        
        return (x - min_val) / range_val

    def compute_score(self, feature_maps, logits, targets):
        """
        Compute GRA score for each channel.
        
        Args:
            feature_maps: Tensor [B, C, H, W] or [B, C] (if already pooled)
            logits: Tensor [B, K] (Network output)
            targets: Tensor [B] (Ground truth labels)
            
        Returns:
            scores: Tensor [C] representing importance of each channel.
        """
        B = feature_maps.size(0)
        C = feature_maps.size(1)
        
        # 1. Feature Aggregation: Global Average Pooling if 4D
        if feature_maps.dim() == 4:
            # [B, C, H, W] -> [B, C]
            f = feature_maps.mean(dim=[2, 3])
        else:
            f = feature_maps

        # Definition: 
        # Reference Sequence X0: The ideal output (logit of the target class)
        # Comparison Sequence Xi: The feature activation of channel i
        
        # 2. Construct Reference Sequence X0
        # We use the logit value corresponding to the true class for each sample.
        # logits: [B, K], targets: [B]
        # reference: [B]
        reference = logits.gather(1, targets.view(-1, 1)).squeeze().view(1, -1) # Shape [1, B]
        
        # 3. Construct Comparison Sequences Xi
        # Each channel is a sequence of length B.
        # f: [B, C] -> Transpose to [C, B]
        comparison = f.t() # Shape [C, B]
        
        # 4. Data Normalization
        # Normalize both reference and comparison sequences to [0, 1]
        ref_norm = self.normalize(reference)
        comp_norm = self.normalize(comparison)
        
        # 5. Calculate Absolute Difference Sequence Delta
        # [C, B] - [1, B] -> [C, B] (Broadcasting)
        delta = torch.abs(comp_norm - ref_norm)
        
        # 6. Global Max and Min Difference
        delta_min = delta.min().item()
        delta_max = delta.max().item()
        
        # 7. Calculate Gray Relational Coefficient gamma
        # gamma_i(k) = (delta_min + rho * delta_max) / (delta_i(k) + rho * delta_max)
        # Shape: [C, B]
        rho_max = self.rho * delta_max
        gamma = (delta_min + rho_max) / (delta + rho_max + 1e-8)
        
        # 8. Calculate Gray Relational Grade (Average over B)
        # Gamma_i = (1/B) * Sum(gamma_i(k))
        # Shape: [C]
        gra_scores = gamma.mean(dim=1)
        
        return gra_scores
