"""
Prior Experiment D: Class-Specific Channel Importance Analysis
================================================================
Analyze which channels are important for specific classes vs general.

Key Insight: Some channels are "class detectors" that respond strongly
to specific semantic categories, while others are "universal features".
GRA should identify class-specific semantic channels that L1 misses.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.resnet_cifar import resnet56
from pruning.gra_score import GrayRelationalChannelScorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'APIN_Submission', 'vis'))
from pub_style import set_publication_style, get_palette, save_figure

def get_cifar10_class_loaders(batch_size=128):
    """Get CIFAR-10 loaders organized by class."""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Organize by class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(testset):
        class_indices[label].append(idx)
    
    return testset, class_indices

def compute_classwise_gra_scores(model, testset, class_indices, device='cuda'):
    """Compute GRA scores separately for each class."""
    model.eval()
    model.to(device)
    
    gra_scorer = GrayRelationalChannelScorer(rho=0.5)
    activations = {}
    layer_names = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv1' in name and 'layer' in name:
            hooks.append(module.register_forward_hook(get_activation(name)))
            layer_names.append(name)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    classwise_scores = {c: {} for c in classes}
    
    for class_id, class_name in enumerate(classes):
        indices = class_indices[class_id][:100]  # Sample 100 per class
        
        subset = torch.utils.data.Subset(testset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
        
        class_scores = {name: [] for name in layer_names}
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                
                for name in layer_names:
                    if name in activations:
                        score = gra_scorer.compute_score(activations[name], logits, targets)
                        class_scores[name].append(score.cpu())
                
                activations.clear()
        
        # Average scores for this class
        for name in layer_names:
            if class_scores[name]:
                classwise_scores[class_name][name] = torch.stack(class_scores[name]).mean(dim=0).numpy()
    
    for h in hooks:
        h.remove()
    
    return classwise_scores, layer_names

def create_classwise_heatmap(classwise_scores, layer_names, save_dir):
    """Create heatmap showing class-channel importance patterns."""
    set_publication_style()
    
    classes = list(classwise_scores.keys())
    
    # Aggregate scores across layers for each class
    # Select a representative layer (e.g., first layer of layer3)
    target_layer = None
    for name in layer_names:
        if 'layer3' in name:
            target_layer = name
            break
    
    if target_layer is None:
        target_layer = layer_names[-1]
    
    # Build matrix: classes x channels
    n_channels = len(classwise_scores[classes[0]][target_layer])
    score_matrix = np.zeros((len(classes), n_channels))
    
    for i, cls in enumerate(classes):
        scores = classwise_scores[cls][target_layer]
        # Normalize per class
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        score_matrix[i] = scores_norm
    
    # Compute class specificity: std across classes for each channel
    channel_specificity = score_matrix.std(axis=0)
    
    # Sort channels by specificity
    sorted_indices = np.argsort(channel_specificity)[::-1]
    
    # Top 20 most class-specific channels
    top_specific = sorted_indices[:20]
    
    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Full heatmap (sampled channels)
    sample_indices = np.linspace(0, n_channels-1, 50, dtype=int)
    sns.heatmap(score_matrix[:, sample_indices], ax=axes[0], cmap='RdYlBu_r',
                xticklabels=False, yticklabels=classes, cbar_kws={'shrink': 0.8})
    axes[0].set_xlabel('Channels (sampled)', fontweight='medium')
    axes[0].set_ylabel('Class', fontweight='medium')
    axes[0].set_title('(a) Channel Importance per Class', fontweight='bold')
    
    # Plot 2: Top class-specific channels
    sns.heatmap(score_matrix[:, top_specific], ax=axes[1], cmap='RdYlBu_r',
                xticklabels=[f'C{i}' for i in top_specific], yticklabels=classes,
                cbar_kws={'shrink': 0.8})
    axes[1].set_xlabel('Top Class-Specific Channels', fontweight='medium')
    axes[1].set_ylabel('Class', fontweight='medium')
    axes[1].set_title('(b) Most Class-Specific Channels', fontweight='bold')
    axes[1].tick_params(axis='x', labelsize=7, rotation=45)
    
    fig.tight_layout()
    
    output_path = os.path.join(save_dir, 'fig_classwise_importance.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    
    # Print insights
    print("\n" + "="*60)
    print("CLASS-SPECIFIC CHANNEL ANALYSIS:")
    print("="*60)
    print(f"Total channels analyzed: {n_channels}")
    print(f"Mean channel specificity: {channel_specificity.mean():.4f}")
    print(f"Max channel specificity: {channel_specificity.max():.4f}")
    print(f"Channels with high specificity (>0.2): {(channel_specificity > 0.2).sum()}")
    print("="*60)
    
    return {
        'n_channels': n_channels,
        'mean_specificity': channel_specificity.mean(),
        'high_specificity_count': (channel_specificity > 0.2).sum()
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = os.path.join(os.path.dirname(__file__), 'prior_experiments')
    os.makedirs(save_dir, exist_ok=True)
    
    model = resnet56(num_classes=10)
    pretrained_path = 'checkpoints/resnet56_cifar10_best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained: {pretrained_path}")
    else:
        print("Using randomly initialized model")
    
    testset, class_indices = get_cifar10_class_loaders()
    
    print("\nComputing class-wise GRA scores...")
    classwise_scores, layer_names = compute_classwise_gra_scores(
        model, testset, class_indices, device)
    
    print("\nCreating class-wise analysis figures...")
    results = create_classwise_heatmap(classwise_scores, layer_names, save_dir)
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == '__main__':
    main()
