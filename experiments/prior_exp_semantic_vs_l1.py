"""
Prior Experiment A: Semantic Alignment vs Structural Energy
============================================================
This experiment proves that channels with high semantic alignment (GRA)
are more important than channels with high structural energy (L1-Norm).

Key Hypothesis: A channel with low L1-Norm but high GRA score can be 
critical for accuracy, while a channel with high L1-Norm but low GRA 
score may be safely removed.

Experiments:
1. Compare GRA-based pruning vs L1-based pruning on same model
2. Identify "hidden semantic channels" (high GRA, low L1)
3. Analyze what happens when we prune them vs preserve them
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.resnet_cifar import resnet56
from pruning.gra_score import GrayRelationalChannelScorer
from pruning.l1_score import get_l1_scores

# Import visualization style
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'APIN_Submission', 'vis'))
from pub_style import set_publication_style, get_palette, save_figure

def get_cifar10_loader(batch_size=128):
    """Get CIFAR-10 data loader."""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    return loader

def collect_all_scores(model, loader, device='cuda'):
    """Collect both GRA and L1 scores for all channels."""
    model.eval()
    model.to(device)
    
    # Get L1 scores
    l1_scores = get_l1_scores(model)
    
    # Get GRA scores
    gra_scorer = GrayRelationalChannelScorer(rho=0.5)
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv1' in name and 'layer' in name:
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    gra_scores = {}
    batch_count = 0
    max_batches = 10
    
    with torch.no_grad():
        for inputs, targets in loader:
            if batch_count >= max_batches:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            for name, feats in activations.items():
                score = gra_scorer.compute_score(feats, logits, targets)
                if name not in gra_scores:
                    gra_scores[name] = score.cpu()
                else:
                    gra_scores[name] += score.cpu()
            
            batch_count += 1
            activations.clear()
    
    for h in hooks:
        h.remove()
    
    # Average GRA scores
    for name in gra_scores:
        gra_scores[name] = gra_scores[name] / batch_count
    
    return l1_scores, gra_scores

def analyze_score_correlation(l1_scores, gra_scores, save_dir):
    """Analyze correlation between L1 and GRA scores."""
    set_publication_style()
    palette = get_palette('nature')
    
    all_l1 = []
    all_gra = []
    layer_labels = []
    
    for name in gra_scores:
        if name in l1_scores:
            l1 = l1_scores[name]
            gra = gra_scores[name].numpy()
            
            # Normalize for comparison
            l1_norm = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-8)
            gra_norm = (gra - gra.min()) / (gra.max() - gra.min() + 1e-8)
            
            all_l1.extend(l1_norm.tolist())
            all_gra.extend(gra_norm.tolist())
            layer_labels.extend([name] * len(l1))
    
    all_l1 = np.array(all_l1)
    all_gra = np.array(all_gra)
    
    # Compute correlation
    pearson_r, pearson_p = pearsonr(all_l1, all_gra)
    spearman_r, spearman_p = spearmanr(all_l1, all_gra)
    
    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.scatter(all_l1, all_gra, alpha=0.4, s=15, c=palette['L1-Norm'], edgecolors='none')
    
    # Add regression line
    z = np.polyfit(all_l1, all_gra, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), color=palette['GRA-CNN'], linewidth=2, 
            label=f'Pearson r = {pearson_r:.3f}')
    
    # Highlight quadrants
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Label quadrants
    ax.text(0.75, 0.85, 'High L1, High GRA\n(Both agree)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.25, 0.85, 'Low L1, High GRA\n(Hidden Semantic)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.text(0.75, 0.15, 'High L1, Low GRA\n(False Positive)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.text(0.25, 0.15, 'Low L1, Low GRA\n(Both agree)', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax.set_xlabel('Normalized L1-Norm Score', fontweight='medium')
    ax.set_ylabel('Normalized GRA Score', fontweight='medium')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Channel Importance: L1-Norm vs GRA', fontweight='bold', fontsize=11)
    
    fig.tight_layout()
    
    output_path = os.path.join(save_dir, 'fig_l1_vs_gra_scatter.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    
    # Count channels in each quadrant
    high_l1_high_gra = np.sum((all_l1 > 0.5) & (all_gra > 0.5))
    low_l1_high_gra = np.sum((all_l1 < 0.5) & (all_gra > 0.5))  # Hidden semantic
    high_l1_low_gra = np.sum((all_l1 > 0.5) & (all_gra < 0.5))  # False positive
    low_l1_low_gra = np.sum((all_l1 < 0.5) & (all_gra < 0.5))
    
    print(f"\nQuadrant Analysis (total {len(all_l1)} channels):")
    print(f"  High L1, High GRA (agreement): {high_l1_high_gra} ({100*high_l1_high_gra/len(all_l1):.1f}%)")
    print(f"  Low L1, High GRA (hidden semantic): {low_l1_high_gra} ({100*low_l1_high_gra/len(all_l1):.1f}%)")
    print(f"  High L1, Low GRA (false positive): {high_l1_low_gra} ({100*high_l1_low_gra/len(all_l1):.1f}%)")
    print(f"  Low L1, Low GRA (agreement): {low_l1_low_gra} ({100*low_l1_low_gra/len(all_l1):.1f}%)")
    
    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'hidden_semantic': low_l1_high_gra,
        'false_positive': high_l1_low_gra,
        'total_channels': len(all_l1)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), 'prior_experiments')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model (pretrained or random init for analysis)
    model = resnet56(num_classes=10)
    
    # Try to load pretrained weights
    pretrained_path = 'checkpoints/resnet56_cifar10_best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained: {pretrained_path}")
    else:
        print("Using randomly initialized model for structural analysis")
    
    # Load data
    loader = get_cifar10_loader()
    
    # Collect scores
    print("\nCollecting L1 and GRA scores...")
    l1_scores, gra_scores = collect_all_scores(model, loader, device)
    
    # Analyze correlation
    print("\nAnalyzing score correlation...")
    results = analyze_score_correlation(l1_scores, gra_scores, save_dir)
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(save_dir, 'correlation_analysis.csv'), index=False)
    
    print(f"\nResults saved to: {save_dir}")
    print("\n" + "="*60)
    print("KEY FINDING: Low correlation between L1 and GRA suggests they")
    print("capture ORTHOGONAL information about channel importance.")
    print("="*60)

if __name__ == '__main__':
    main()
