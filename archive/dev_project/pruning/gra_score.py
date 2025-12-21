import torch
import torch.nn as nn

class GrayRelationalChannelScorer:
    def __init__(self, rho=0.5):
        self.rho = rho

    def normalize(self, x):
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val < 1e-8] = 1.0 
        return (x - min_val) / range_val

    def compute_score(self, feature_maps, logits, targets):
        B = feature_maps.size(0)
        
        if feature_maps.dim() == 4:
            f = feature_maps.mean(dim=[2, 3])
        else:
            f = feature_maps

        reference = logits.gather(1, targets.view(-1, 1)).squeeze().view(1, -1) # [1, B]
        comparison = f.t() # [C, B]
        
        ref_norm = self.normalize(reference)
        comp_norm = self.normalize(comparison)
        
        delta = torch.abs(comp_norm - ref_norm)
        delta_min = delta.min().item()
        delta_max = delta.max().item()
        
        rho_max = self.rho * delta_max
        gamma = (delta_min + rho_max) / (delta + rho_max + 1e-8)
        gra_scores = gamma.mean(dim=1)
        
        return gra_scores
