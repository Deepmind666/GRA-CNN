import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 1. Load Data
    # Assuming results.csv is in the same directory or parent
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    if not os.path.exists(csv_path):
        csv_path = 'results.csv' # Try current dir
    
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-20 on CIFAR-10
    df = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-20')]
    
    # 2. Setup Style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.figure(figsize=(6, 5))
    
    # 3. Plot
    methods = {
        'L1-Norm': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        'FPGM': {'color': 'green', 'linestyle': '--', 'marker': 's'},
        'HRank': {'color': 'orange', 'linestyle': '-.', 'marker': '^'},
        'GRA-CNN': {'color': 'red', 'linestyle': '-', 'marker': 'D'}
    }
    
    # Baseline (approximate if not in CSV, usually around 91.6-92.0 for ResNet-20)
    baseline_acc = 91.65 # Standard ResNet-20 baseline
    plt.axhline(y=baseline_acc, color='black', linestyle=':', label='Baseline')
    
    for method, style in methods.items():
        subset = df[df['method'] == method].sort_values('prune_ratio')
        # Filter for relevant ratios (e.g. 0.3 to 0.8)
        subset = subset[(subset['prune_ratio'] >= 0.2) & (subset['prune_ratio'] <= 0.9)]
        
        if not subset.empty:
            plt.plot(subset['prune_ratio'], subset['accuracy'], label=method, **style)
            
    # 4. Axes & Ticks
    # plt.xticks([0.3, 0.5, 0.7])
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('ResNet-20 on CIFAR-10')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 5. Save
    save_path = os.path.join(os.path.dirname(__file__), '../fig2.pdf')
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
