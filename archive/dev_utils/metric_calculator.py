import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

class MetricCalculator:
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()
        self.model.to(device)

    def get_layer(self, layer_name):
        # Recursive get
        modules = layer_name.split('.')
        m = self.model
        for name in modules:
            m = getattr(m, name)
        return m

    def compute_scores(self, layer_name, metric='l1', num_batches=10):
        """
        Computes importance scores for channels in layer_name using specified metric.
        Returns: numpy array of scores (shape: [out_channels])
        """
        layer = self.get_layer(layer_name)
        if metric == 'l1':
            return self._compute_l1(layer)
        
        # For activation-based metrics
        return self._compute_activation_metrics(layer, metric, num_batches)

    def _compute_l1(self, layer):
        if isinstance(layer, nn.Conv2d):
            # L1 norm of weights: sum(|w|) per filter
            # Weight shape: [out_channels, in_channels, k, k]
            return layer.weight.data.abs().view(layer.weight.size(0), -1).sum(dim=1).cpu().numpy()
        else:
            raise ValueError("L1 metric only supports Conv2d")

    def _compute_activation_metrics(self, layer, metric, num_batches):
        activations = []
        targets = []
        logits_gt = [] # Logits of the ground truth class

        def hook_fn(module, input, output):
            # Global Average Pooling: [B, C, H, W] -> [B, C]
            if output.dim() == 4:
                act = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
            else:
                act = output.view(output.size(0), -1)
            activations.append(act.detach().cpu())

        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.dataloader, desc=f"Computing {metric}")):
                if i >= num_batches:
                    break
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                
                # Get logit of GT class
                # out: [B, NumClasses]
                # target: [B]
                # gather: [B, 1] -> squeeze -> [B]
                gt_logits = out.gather(1, target.view(-1, 1)).squeeze()
                
                logits_gt.append(gt_logits.detach().cpu())
                targets.append(target.cpu())
        
        handle.remove()

        # Concatenate all batches
        # A: [TotalB, C]
        A = torch.cat(activations, dim=0).numpy()
        # Z: [TotalB]
        Z = torch.cat(logits_gt, dim=0).numpy()
        
        num_channels = A.shape[1]
        scores = np.zeros(num_channels)

        if metric == 'gra':
            scores = self._calc_gra(A, Z)
        elif metric == 'cosine':
            scores = self._calc_cosine(A, Z)
        elif metric == 'correlation':
            scores = self._calc_correlation(A, Z)
        elif metric == 'mi':
            scores = self._calc_mi(A, Z)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return scores

    def _calc_gra(self, A, Z, rho=0.5):
        # A: [N, C], Z: [N]
        # Normalize Z? Usually GRA doesn't strictly require normalization if delta is relative, 
        # but typically we normalize to [0,1] or standard scale.
        # However, standard GRA formula uses absolute difference.
        # Let's keep it raw as per my paper's Eq.
        
        # Expand Z to [N, C]
        Z_expanded = Z[:, np.newaxis] # Broadcasting
        
        # Difference matrix
        # abs(a_c(i) - z(i))
        # Note: A and Z might have different scales. 
        # My GRA paper usually implies some normalization. 
        # Let's normalize both to [0,1] per sequence to be safe and standard.
        
        # Normalize columns of A
        A_min = A.min(axis=0, keepdims=True)
        A_max = A.max(axis=0, keepdims=True)
        A_norm = (A - A_min) / (A_max - A_min + 1e-8)
        
        # Normalize Z
        Z_min = Z.min()
        Z_max = Z.max()
        Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-8)
        Z_norm = Z_norm[:, np.newaxis]
        
        diff = np.abs(A_norm - Z_norm)
        
        # Global max/min diff
        delta_min = diff.min()
        delta_max = diff.max()
        
        # GRA Coefficient
        # [N, C]
        xi = (delta_min + rho * delta_max) / (diff + rho * delta_max)
        
        # Average over samples -> [C]
        gamma = xi.mean(axis=0)
        return gamma

    def _calc_cosine(self, A, Z):
        # Cosine similarity between each channel column A[:, c] and Z
        # A: [N, C], Z: [N]
        
        # Normalize
        A_norm = np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
        Z_norm = np.linalg.norm(Z) + 1e-8
        
        # Dot product: (Z . A) / (|Z| |A|)
        # Z . A -> [C]
        dot = Z.dot(A) 
        scores = dot / (Z_norm * A_norm)
        return scores.flatten()

    def _calc_correlation(self, A, Z):
        # Pearson correlation
        # Centered A
        A_mean = A.mean(axis=0, keepdims=True)
        A_centered = A - A_mean
        
        Z_mean = Z.mean()
        Z_centered = Z - Z_mean
        
        # Covariance
        cov = np.dot(Z_centered, A_centered) # [C]
        
        # Std
        A_std = np.sqrt((A_centered**2).sum(axis=0)) + 1e-8
        Z_std = np.sqrt((Z_centered**2).sum()) + 1e-8
        
        corr = cov / (Z_std * A_std)
        return np.abs(corr) # Magnitude of correlation is usually what matters for importance? 
                            # Or just raw correlation? 
                            # If neg correlation, it's also "aligned" (inversely).
                            # GRA is unsigned (similarity). 
                            # Let's return abs(correlation) as importance.

    def _calc_mi(self, A, Z):
        # Mutual Information
        # Use sklearn. 
        # A: [N, C], Z: [N]
        # This is slow for large C.
        # We can loop over C? Or sklearn handles matrix X?
        # mutual_info_regression(X, y) where X is [N, C], y is [N]
        
        # To speed up, use fewer neighbors or samples if needed.
        # But for "Lightweight", 10 batches (1280 samples) is fine.
        
        mi = mutual_info_regression(A, Z, discrete_features=False)
        return mi

if __name__ == "__main__":
    # Test
    pass
