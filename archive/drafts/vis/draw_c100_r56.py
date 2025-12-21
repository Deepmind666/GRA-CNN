import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-56 on CIFAR-100
    df = df[(df['dataset'] == 'CIFAR-100') & (df['model'] == 'ResNet-56')]
    
    ratios = [0.3, 0.5, 0.7]
    methods = ['L1-Norm', 'GRA-CNN']
    
    # Prepare data for plotting
    acc_l1 = []
    acc_gra = []
    
    for r in ratios:
        # L1
        row_l1 = df[(df['method'] == 'L1-Norm') & (df['prune_ratio'] == r)]
        if not row_l1.empty:
            acc_l1.append(row_l1['accuracy'].iloc[-1])
        else:
            acc_l1.append(0)
            
        # GRA
        row_gra = df[(df['method'] == 'GRA-CNN') & (df['prune_ratio'] == r)]
        if not row_gra.empty:
            acc_gra.append(row_gra['accuracy'].iloc[-1])
        else:
            acc_gra.append(0)
            
    x = np.arange(len(ratios))
    width = 0.35
    
    plt.figure(figsize=(6, 5))
    
    # Plot Bars
    plt.bar(x - width/2, acc_l1, width, label='L1-Norm', color=palette['L1-Norm'])
    plt.bar(x + width/2, acc_gra, width, label='GRA-CNN', color=palette['GRA-CNN'])
    
    # Baseline
    baseline_acc = 71.5 # ResNet-56 CIFAR-100 Baseline
    plt.axhline(y=baseline_acc, color='black', linestyle='--', label='Baseline')
    
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xticks(x, ratios)
    
    # Zoom Y-axis
    # Filter out 0s for min calculation
    valid_accs = [a for a in acc_l1 + acc_gra if a > 0]
    if valid_accs:
        y_min = min(valid_accs)
        plt.ylim(y_min - 2.0, baseline_acc + 2.0)
    
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig8.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
