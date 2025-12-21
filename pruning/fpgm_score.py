import torch
import torch.nn as nn
from .l1_score import L1ChannelScorer

class FPGMChannelScorer(L1ChannelScorer):
    """
    Filter Pruning via Geometric Median (FPGM).
    
    Reference:
    He et al., "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration", CVPR 2019.
    
    Idea:
    - Filters near the geometric median of the layer are redundant (can be represented by others).
    - Prune filters with SMALL distance to the geometric median.
    - Wait, the paper says: "Filters with smaller sum of distances to other filters are more redundant".
    - Actually, the Geometric Median is the point that minimizes the sum of Euclidean distances to all other points.
    - If a filter is close to the GM, it is redundant.
    - So we calculate the GM, then calculate distance of each filter to GM.
    - Score = Distance to GM. 
    - We prune filters with SMALL scores (close to GM).
    - Large score = Outlier = Informative.
    """
    def __init__(self, model):
        super().__init__(model)

    def get_score(self, layer_name, layer_module, **kwargs):
        """
        Compute FPGM score for a single layer.
        
        Args:
            layer_module: nn.Conv2d module
        Returns:
            scores: tensor of shape (out_channels,)
        """
        if not isinstance(layer_module, nn.Conv2d):
            raise ValueError(f"Layer {layer_name} is not nn.Conv2d")
        
        weight = layer_module.weight.data # (Out, In, K, K)
        n_out = weight.shape[0]
        
        # Flatten filters: (Out, In*K*K)
        flat_filters = weight.view(n_out, -1)
        
        # Compute Geometric Median
        # Ideally, GM is found via Weiszfeld's algorithm, but that's slow.
        # In FPGM paper, they often use Euclidean distance sum as a proxy or iterate.
        # However, a common simplified implementation is:
        # Calculate distance matrix between all pairs of filters.
        # Score_i = Sum_j ||W_i - W_j||_2
        # If Score_i is SMALL, W_i is central (redundant).
        # If Score_i is LARGE, W_i is far from others (unique/important).
        # This is technically "Filter Pruning via Distance Sum", but effectively finds central filters.
        # Let's stick to the exact definition if possible, or this robust proxy.
        # The paper "Filter Pruning via Geometric Median" explicitly proposes:
        # "Calculate the Geometric Median ... " 
        # But in Eq (6) of the paper: 
        # x* = argmin_x sum ||x - w_i||
        # Then for each filter w_i, value = ||w_i - x*||.
        # Prune those with small value.
        
        # Since finding exact GM is iterative, we can use the mean as a fast initialization 
        # or just use the "Sum of Distances" proxy which is highly correlated and often used in repros.
        # BUT, let's try to implement the robust version:
        # 1. Compute pair-wise distance matrix (N, N)
        # 2. Sum over rows -> Total distance to all other filters.
        # 3. Filters with SMALL total distance are "central" -> Prune.
        # 4. Filters with LARGE total distance are "outliers" -> Keep.
        
        # Efficient pair-wise distance:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        
        r = torch.sum(flat_filters * flat_filters, dim=1).view(-1, 1) # (N, 1)
        # dist_mat = r + r.t() - 2 * flat_filters @ flat_filters.t()
        # Using cdist is safer and cleaner
        
        dist_mat = torch.cdist(flat_filters, flat_filters, p=2) # (N, N)
        
        # Sum of distances to all other filters
        scores = torch.sum(dist_mat, dim=1) # (N,)
        
        # We want to prune SMALL scores (central/redundant).
        # So score is directly proportional to importance.
        
        return scores
