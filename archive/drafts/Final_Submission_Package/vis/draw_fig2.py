import pandas as pd
import matplotlib.pyplot as plt
import os
from draw_utils import set_style, save_fig, get_palette, get_markers

def main():
    set_style()
    palette = get_palette()
    markers = get_markers()
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-20 CIFAR-10
    subset = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-20')]
    
    # Filter for specific methods and ratios
    target_ratios = [0.3, 0.5, 0.7]
    subset = subset[subset['prune_ratio'].isin(target_ratios)]
    
    # Ensure GRA-CNN uses rho=0.5 if multiple exist, or just take the default
    # If 'rho' column exists
    if 'rho' in subset.columns:
        subset = subset[(subset['method'] != 'GRA-CNN') | (subset['rho'] == 0.5)]

    # Methods to plot
    methods = ['L1-Norm', 'FPGM', 'HRank', 'GRA-CNN']
    
    plt.figure(figsize=(6, 5))
    
    for method in methods:
        data = subset[subset['method'] == method].sort_values('prune_ratio')
        if not data.empty:
            plt.plot(data['prune_ratio'], data['accuracy'], 
                     marker=markers.get(method, 'o'), 
                     label=method, 
                     color=palette.get(method, 'black'))
    
    # Add Baseline
    # ResNet-20 Baseline ~91.65 (or from data if available)
    # Check if baseline row exists (prune_ratio=0) or just add line
    baseline_acc = 91.65 # Standard ResNet-20 baseline
    plt.axhline(y=baseline_acc, color='black', linestyle='--', label='Baseline')
    
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Top-1 Accuracy (%)')
    # plt.title('ResNet-20 on CIFAR-10') # Paper usually doesn't have title in plot
    
    plt.xticks(target_ratios)
    plt.xlim(0.25, 0.75)
    
    # Zoom Y-axis
    # Find min/max of displayed data
    y_min = subset[subset['method'].isin(methods)]['accuracy'].min()
    plt.ylim(y_min - 1.0, baseline_acc + 0.5)
    
    plt.legend(loc='lower left')
    plt.grid(True)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig2.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == '__main__':
    main()
