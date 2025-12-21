import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import sys

# Add code dir
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add root for utils
from models import resnet20
from train_resnet20 import get_dataloader
from utils.metric_calculator import MetricCalculator

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet20()
    
    # Load baseline
    baseline_path = 'experiments/baseline_cifar10_resnet20.pth'
    if os.path.exists(baseline_path):
        ckpt = torch.load(baseline_path, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Baseline not found, using random init (stability might be random)")
        
    model.to(device)
    model.eval()
    
    # Get 2 different sets of batches
    loader, _ = get_dataloader(batch_size=128)
    iter_loader = iter(loader)
    
    batches1 = [next(iter_loader) for _ in range(5)]
    batches2 = [next(iter_loader) for _ in range(5)]
    
    class MultiBatchLoader:
        def __init__(self, batches):
            self.batches = batches
        def __iter__(self):
            for b in self.batches:
                yield b
        def __len__(self):
            return len(self.batches)
            
    loader1 = MultiBatchLoader(batches1)
    loader2 = MultiBatchLoader(batches2)
    
    layer = 'layer2.0.conv1'
    calc1 = MetricCalculator(model, loader1, device)
    calc2 = MetricCalculator(model, loader2, device)
    
    metrics = ['gra', 'cosine', 'correlation', 'mi']
    stability_scores = {}
    
    print(f"Analyzing stability on {layer} (5 batches)...")
    
    for m in metrics:
        s1 = calc1.compute_scores(layer, metric=m, num_batches=5)
        s2 = calc2.compute_scores(layer, metric=m, num_batches=5)
        
        corr, _ = spearmanr(s1, s2)
        stability_scores[m] = corr
        
        # Jaccard of bottom 50%
        k = len(s1) // 2
        idx1 = np.argsort(s1)[:k]
        idx2 = np.argsort(s2)[:k]
        intersection = len(set(idx1) & set(idx2))
        union = len(set(idx1) | set(idx2))
        jaccard = intersection / union
        print(f"Metric: {m}, Stability (Spearman Rho): {corr:.4f}, Jaccard (Bottom 50%): {jaccard:.4f}")
        
    # Plot
    plt.figure(figsize=(6, 5))
    names = list(stability_scores.keys())
    values = list(stability_scores.values())
    
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    plt.bar(names, values, color=colors, alpha=0.8)
    plt.ylabel('Rank Stability (Spearman $\\rho$)')
    plt.title('Score Stability across Different Batches')
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    plt.grid(axis='y', alpha=0.3)
    
    out_path = 'vis/stability_analysis.pdf'
    plt.savefig(out_path)
    print(f"Saved stability plot to {out_path}")

if __name__ == "__main__":
    main()
