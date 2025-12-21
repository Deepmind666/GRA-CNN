import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import argparse

# Import models and metric calculator
from models.resnet_cifar import resnet20, resnet56
from utils.metric_calculator import MetricCalculator

def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Check default paths
    data_roots = ['data', 'datasets', '../data']
    root = 'data'
    for r in data_roots:
        if os.path.exists(r):
            root = r
            break
            
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path to pretrained model')
    parser.add_argument('--layer', type=str, default='layer2.0.conv1', help='Layer to analyze')
    parser.add_argument('--model', type=str, default='resnet20', help='Model architecture')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    if args.model == 'resnet20':
        model = resnet20()
    elif args.model == 'resnet56':
        model = resnet56()
    else:
        raise ValueError("Unknown model")

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Handle if it's a dict with 'state_dict' or just state_dict
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif 'net' in ckpt:
            model.load_state_dict(ckpt['net'])
        else:
            try:
                model.load_state_dict(ckpt)
            except:
                print("Could not load state dict directly. Using random init (for pipeline verification only).")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Using random init.")

    model.to(device)
    model.eval()

    # 2. DataLoader
    trainloader = get_dataloader()

    # 3. Compute Metrics
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

    # 4. Save & Visualize
    df = pd.DataFrame(results)
    output_csv = 'vis/metric_comparison.csv'
    os.makedirs('vis', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved scores to {output_csv}")
    
    # Correlation Matrix
    corr_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Save Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Metric Correlation ({args.layer})')
    plt.tight_layout()
    plt.savefig('vis/metric_correlation_heatmap.pdf')
    print("Saved heatmap to vis/metric_correlation_heatmap.pdf")

if __name__ == "__main__":
    main()
