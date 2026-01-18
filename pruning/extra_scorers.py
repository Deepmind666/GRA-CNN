"""
Taylor Importance Scorer for Channel Pruning
=============================================
First-order Taylor expansion to estimate channel importance.
Reference: Molchanov et al., "Pruning Convolutional Neural Networks for Resource Efficient Inference"

Importance = |∂L/∂a * a| where a is channel activation
"""

import torch
import torch.nn as nn
import numpy as np

class TaylorChannelScorer:
    """
    Computes channel importance using first-order Taylor expansion.
    Importance is estimated as the product of activation and its gradient.
    """
    
    def __init__(self, model, criterion=None):
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:  # ResNet
                    self.hooks.append(module.register_forward_hook(get_activation(name)))
                    self.hooks.append(module.register_backward_hook(get_gradient(name)))
                elif 'features' in name:  # VGG
                    self.hooks.append(module.register_forward_hook(get_activation(name)))
                    self.hooks.append(module.register_backward_hook(get_gradient(name)))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_scores(self, dataloader, device='cuda', num_batches=10):
        """
        Compute Taylor importance scores across multiple batches.
        
        Args:
            dataloader: Training data loader
            device: Computation device
            num_batches: Number of batches to average over
            
        Returns:
            scores: Dict[layer_name -> importance array of shape (C,)]
        """
        self.model.to(device)
        self.model.eval()
        self._register_hooks()
        
        accumulated_scores = {}
        batch_count = 0
        
        for inputs, targets in dataloader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Compute Taylor importance: |activation * gradient|
            for name in self.activations:
                if name in self.gradients:
                    act = self.activations[name]
                    grad = self.gradients[name]
                    
                    # Taylor importance per channel: mean over batch, H, W
                    importance = torch.abs(act * grad).mean(dim=[0, 2, 3])
                    
                    if name not in accumulated_scores:
                        accumulated_scores[name] = importance.detach().cpu()
                    else:
                        accumulated_scores[name] += importance.detach().cpu()
            
            batch_count += 1
        
        self._remove_hooks()
        
        # Average and normalize
        scores = {}
        for name, score in accumulated_scores.items():
            scores[name] = (score / batch_count).numpy()
        
        return scores


class ACPChannelScorer:
    """
    Approximate Channel Pruning using K-Means clustering.
    Reference: Chang et al., "ACP: Automatic Channel Pruning via Clustering and Swarm Intelligence"
    
    Simplified implementation: cluster channels by activation patterns,
    prune channels close to cluster centroids (redundant).
    """
    
    def __init__(self, model, n_clusters=None):
        self.model = model
        self.n_clusters = n_clusters  # If None, auto-determine
        self.activations = {}
        self.hooks = []
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                if name not in self.activations:
                    self.activations[name] = []
                # Store GAP representation
                feat = output.mean(dim=[2, 3]).detach().cpu()  # [B, C]
                self.activations[name].append(feat)
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    self.hooks.append(module.register_forward_hook(get_activation(name)))
                elif 'features' in name:
                    self.hooks.append(module.register_forward_hook(get_activation(name)))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_scores(self, dataloader, device='cuda', num_batches=20):
        """
        Compute ACP-style importance scores.
        Channels far from cluster centroids are considered more unique/important.
        """
        from sklearn.cluster import KMeans
        
        self.model.to(device)
        self.model.eval()
        self._register_hooks()
        
        batch_count = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                if batch_count >= num_batches:
                    break
                inputs = inputs.to(device)
                self.model(inputs)
                batch_count += 1
        
        self._remove_hooks()
        
        scores = {}
        for name, act_list in self.activations.items():
            # Concatenate all batches: [Total_samples, C]
            all_acts = torch.cat(act_list, dim=0).numpy()
            n_channels = all_acts.shape[1]
            
            # Transpose to cluster channels: [C, Total_samples]
            channel_features = all_acts.T
            
            # Determine number of clusters
            n_clusters = self.n_clusters or max(2, n_channels // 4)
            n_clusters = min(n_clusters, n_channels - 1)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(channel_features)
            
            # Distance to nearest centroid (lower = more redundant)
            distances = np.zeros(n_channels)
            for i in range(n_channels):
                cluster_id = kmeans.labels_[i]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances[i] = np.linalg.norm(channel_features[i] - centroid)
            
            # Higher distance = more unique = more important
            scores[name] = distances
        
        return scores
