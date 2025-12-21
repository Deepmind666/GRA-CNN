import torch
import torch.nn as nn
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add code dir
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

from models import resnet56
from utils.metric_calculator import MetricCalculator
from train_resnet20 import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/baseline_cifar10_resnet56.pth')
    parser.add_argument('--layer', type=str, default='layer2.0.conv1', help='Layer to analyze')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model = resnet56()
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"Checkpoint {args.checkpoint} not found. Using random init.")

    model.to(device)
    model.eval()

    # DataLoader
    trainloader, _ = get_dataloader(batch_size=128)

    # Compute Metrics
    metrics = ['l1', 'gra', 'cosine', 'correlation', 'mi']
    calculator = MetricCalculator(model, trainloader, device)
    
    results = {}
    print(f"Analyzing layer: {args.layer}")
    
    for m in metrics:
        print(f"Computing {m}...")
        try:
            scores = calculator.compute_scores(args.layer, metric=m, num_batches=10)
            results[m] = scores
        except Exception as e:
            print(f"Error computing {m}: {e}")
            results[m] = None

    # Save
    df = pd.DataFrame(results)
    output_csv = 'vis/metric_comparison_r56.csv'
    os.makedirs('vis', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved scores to {output_csv}")
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Metric Correlation (ResNet-56 {args.layer})')
    plt.tight_layout()
    plt.savefig('vis/metric_correlation_heatmap_r56.pdf')
    print("Saved heatmap.")

if __name__ == "__main__":
    main()
