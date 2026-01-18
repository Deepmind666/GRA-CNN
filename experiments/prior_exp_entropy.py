"""
Prior Experiment E: Activation Entropy and Information Theory Analysis
=======================================================================
Analyze channel importance from an information-theoretic perspective.

Key Hypothesis: GRA approximates mutual information I(channel; decision)
- High entropy channels = high information capacity
- GRA captures functional coupling to output, not just structural energy
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.resnet_cifar import resnet56
from pruning.gra_score import GrayRelationalChannelScorer
from pruning.l1_score import get_l1_scores

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'APIN_Submission', 'vis'))
from pub_style import set_publication_style, get_palette, save_figure

def get_cifar10_loader(batch_size=128):
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)

def compute_channel_entropy(activations):
    """Compute entropy of channel activations (discretized)."""
    # Global average pool
    gap = activations.mean(dim=[2, 3])  # [B, C]
    
    entropies = []
    for c in range(gap.shape[1]):
        values = gap[:, c].cpu().numpy()
        # Discretize into bins
        hist, _ = np.histogram(values, bins=20, density=True)
        hist = hist / hist.sum() + 1e-10
        entropies.append(entropy(hist))
    
    return np.array(entropies)

def collect_entropy_and_scores(model, loader, device='cuda'):
    """Collect entropy, GRA, and L1 scores for analysis."""
    model.eval()
    model.to(device)
    
    gra_scorer = GrayRelationalChannelScorer(rho=0.5)
    l1_scores = get_l1_scores(model)
    
    activations_all = {}
    layer_names = []
    
    def get_activation(name):
        def hook(module, input, output):
            if name not in activations_all:
                activations_all[name] = []
            activations_all[name].append(output.detach())
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv1' in name and 'layer' in name:
            hooks.append(module.register_forward_hook(get_activation(name)))
            layer_names.append(name)
    
    gra_scores = {name: [] for name in layer_names}
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= 10:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            for name in layer_names:
                if name in activations_all and activations_all[name]:
                    latest = activations_all[name][-1]
                    score = gra_scorer.compute_score(latest, logits, targets)
                    gra_scores[name].append(score.cpu())
    
    for h in hooks:
        h.remove()
    
    # Compute entropy for each layer
    entropy_scores = {}
    for name in layer_names:
        if activations_all[name]:
            all_acts = torch.cat(activations_all[name], dim=0)
            entropy_scores[name] = compute_channel_entropy(all_acts)
    
    # Average GRA scores
    for name in gra_scores:
        if gra_scores[name]:
            gra_scores[name] = torch.stack(gra_scores[name]).mean(dim=0).numpy()
    
    return entropy_scores, gra_scores, l1_scores, layer_names

def create_entropy_analysis_figure(entropy_scores, gra_scores, l1_scores, layer_names, save_dir):
    """Create figure showing relationship between entropy and importance."""
    set_publication_style()
    palette = get_palette('nature')
    
    # Collect all scores
    all_entropy = []
    all_gra = []
    all_l1 = []
    
    for name in layer_names:
        if name in entropy_scores and name in gra_scores and name in l1_scores:
            ent = entropy_scores[name]
            gra = gra_scores[name]
            l1 = l1_scores[name]
            
            # Normalize
            ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
            gra_norm = (gra - gra.min()) / (gra.max() - gra.min() + 1e-8)
            l1_norm = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-8)
            
            all_entropy.extend(ent_norm.tolist())
            all_gra.extend(gra_norm.tolist())
            all_l1.extend(l1_norm.tolist())
    
    all_entropy = np.array(all_entropy)
    all_gra = np.array(all_gra)
    all_l1 = np.array(all_l1)
    
    from scipy.stats import pearsonr
    
    r_gra_ent, _ = pearsonr(all_gra, all_entropy)
    r_l1_ent, _ = pearsonr(all_l1, all_entropy)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: GRA vs Entropy
    axes[0].scatter(all_entropy, all_gra, alpha=0.4, s=15, c=palette['GRA-CNN'])
    z = np.polyfit(all_entropy, all_gra, 1)
    p = np.poly1d(z)
    axes[0].plot(np.linspace(0,1,100), p(np.linspace(0,1,100)), 
                 color='black', linewidth=2, linestyle='--',
                 label=f'r = {r_gra_ent:.3f}')
    axes[0].set_xlabel('Normalized Entropy', fontweight='medium')
    axes[0].set_ylabel('Normalized GRA Score', fontweight='medium')
    axes[0].set_title('(a) GRA vs Activation Entropy', fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: L1 vs Entropy
    axes[1].scatter(all_entropy, all_l1, alpha=0.4, s=15, c=palette['L1-Norm'])
    z = np.polyfit(all_entropy, all_l1, 1)
    p = np.poly1d(z)
    axes[1].plot(np.linspace(0,1,100), p(np.linspace(0,1,100)), 
                 color='black', linewidth=2, linestyle='--',
                 label=f'r = {r_l1_ent:.3f}')
    axes[1].set_xlabel('Normalized Entropy', fontweight='medium')
    axes[1].set_ylabel('Normalized L1-Norm Score', fontweight='medium')
    axes[1].set_title('(b) L1-Norm vs Activation Entropy', fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation comparison bar chart
    methods = ['GRA', 'L1-Norm']
    correlations = [r_gra_ent, r_l1_ent]
    colors = [palette['GRA-CNN'], palette['L1-Norm']]
    
    bars = axes[2].bar(methods, correlations, color=colors, edgecolor='white', linewidth=2)
    axes[2].set_ylabel('Correlation with Entropy', fontweight='medium')
    axes[2].set_title('(c) Information Capture Comparison', fontweight='bold')
    axes[2].set_ylim(-0.1, max(correlations) + 0.1)
    axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, correlations):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.tight_layout()
    
    output_path = os.path.join(save_dir, 'fig_entropy_analysis.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    
    print("\n" + "="*60)
    print("INFORMATION THEORY ANALYSIS:")
    print("="*60)
    print(f"GRA-Entropy correlation: r = {r_gra_ent:.4f}")
    print(f"L1-Entropy correlation:  r = {r_l1_ent:.4f}")
    print(f"GRA captures {abs(r_gra_ent/r_l1_ent):.2f}x more entropy information")
    print("="*60)
    
    return {'r_gra_entropy': r_gra_ent, 'r_l1_entropy': r_l1_ent}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = os.path.join(os.path.dirname(__file__), 'prior_experiments')
    os.makedirs(save_dir, exist_ok=True)
    
    model = resnet56(num_classes=10)
    pretrained_path = 'checkpoints/resnet56_cifar10_best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print("Using randomly initialized model")
    
    loader = get_cifar10_loader()
    
    print("\nComputing entropy and importance scores...")
    entropy_scores, gra_scores, l1_scores, layer_names = collect_entropy_and_scores(
        model, loader, device)
    
    print("\nCreating entropy analysis figures...")
    results = create_entropy_analysis_figure(
        entropy_scores, gra_scores, l1_scores, layer_names, save_dir)
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == '__main__':
    main()
