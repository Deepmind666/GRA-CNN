
import torch
import torch.nn as nn

class GrayRelationalChannelScorer:
    def __init__(self, rho=0.5):
        self.rho = rho

    def normalize(self, x):
        """Min-Max normalization to [0, 1] per sequence (dim 1)"""
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1e-8
        return (x - min_val) / range_val

    def compute_score(self, feature_maps, targets, logits=None):
        """
        feature_maps: [N, C, H, W] or [N, C] (if already pooled)
        targets: [N] (class labels)
        logits: [N, NumClasses] (optional, can be used as reference)
        """
        N = feature_maps.size(0)
        C = feature_maps.size(1)
        
        # 1. Global Average Pooling if 4D
        if feature_maps.dim() == 4:
            # [N, C, H, W] -> [N, C]
            f = feature_maps.view(N, C, -1).mean(dim=2)
        else:
            f = feature_maps

        # 2. Prepare Sequences
        # Transpose to [C, N] so each channel is a sequence of length N
        sequences = f.t() # [C, N]
        
        # 3. Prepare Reference Sequence
        # Option A: Use the logit of the correct class.
        # Ideally, a good feature should correlate with the correct class score.
        if logits is not None:
            # Gather logits for the correct class
            # logits: [N, 10], targets: [N]
            # reference: [N]
            reference = logits.gather(1, targets.view(-1, 1)).squeeze() # [N]
        else:
            # Option B: Just use the target label? (Not good for classification, categorical)
            # Option C: Use the magnitude of the feature itself? (L1 norm)
            # Let's assume logits are provided.
            raise ValueError("Logits are required for GRA reference sequence.")
            
        # Reshape reference to [1, N]
        reference = reference.view(1, -1) # [1, N]
        
        # 4. Normalize
        # Normalize sequences and reference to [0, 1]
        # Note: Normalization across the batch dimension (dim 1)
        seq_norm = self.normalize(sequences)
        ref_norm = self.normalize(reference)
        
        # 5. Compute Difference Sequence
        # [C, N] - [1, N] -> [C, N]
        delta = torch.abs(seq_norm - ref_norm)
        
        # 6. Global Min and Max Difference
        min_delta = delta.min()
        max_delta = delta.max()
        
        # 7. Compute Gray Relational Coefficient
        # [C, N]
        gamma = (min_delta + self.rho * max_delta) / (delta + self.rho * max_delta)
        
        # 8. Compute Gray Relational Degree (Average over N)
        # [C]
        gra_score = gamma.mean(dim=1)
        
        return gra_score

