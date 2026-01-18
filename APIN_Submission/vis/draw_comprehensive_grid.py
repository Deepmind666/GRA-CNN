"""
12-Panel Grid Figure Generator for GRA-CNN
============================================
Creates publication-quality 4x3 grid figure matching reference standard.
Each panel shows accuracy vs. pruning ratio for a specific dataset/architecture combination.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pub_style import set_publication_style, get_palette, get_markers, get_linestyles, save_figure

# =============================================================================
# SIMULATED DATA (Replace with real experimental results)
# =============================================================================

def generate_simulated_data():
    """
    Generate simulated experimental data for visualization development.
    This will be replaced with actual experimental results.
    """
    np.random.seed(42)
    
    configs = [
        ('CIFAR-10', 'ResNet-20'),
        ('CIFAR-10', 'ResNet-56'),
        ('CIFAR-10', 'ResNet-110'),
        ('CIFAR-10', 'VGG-16'),
        ('CIFAR-100', 'ResNet-20'),
        ('CIFAR-100', 'ResNet-56'),
        ('CIFAR-100', 'ResNet-110'),
        ('CIFAR-100', 'VGG-16'),
        ('Tiny-200', 'ResNet-18'),
        ('Tiny-200', 'MobileV2'),
        ('Summary', 'Heatmap'),
        ('Summary', 'Pareto'),
    ]
    
    methods = ['L1-Norm', 'FPGM', 'HRank', 'Taylor', 'ACP', 'GRA-CNN']
    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Baseline accuracies
    baselines = {
        ('CIFAR-10', 'ResNet-20'): 91.65,
        ('CIFAR-10', 'ResNet-56'): 93.68,
        ('CIFAR-10', 'ResNet-110'): 94.20,
        ('CIFAR-10', 'VGG-16'): 93.50,
        ('CIFAR-100', 'ResNet-20'): 68.50,
        ('CIFAR-100', 'ResNet-56'): 71.80,
        ('CIFAR-100', 'ResNet-110'): 73.50,
        ('CIFAR-100', 'VGG-16'): 72.00,
        ('Tiny-200', 'ResNet-18'): 67.92,
        ('Tiny-200', 'MobileV2'): 65.00,
    }
    
    # Generate degradation curves (GRA-CNN degrades slowest)
    data = []
    for config in configs[:10]:  # First 10 configs
        baseline = baselines.get(config, 90.0)
        for method in methods:
            for ratio in ratios:
                # Different degradation rates
                if method == 'GRA-CNN':
                    drop_rate = 0.08  # Best
                elif method == 'Taylor':
                    drop_rate = 0.12
                elif method == 'ACP':
                    drop_rate = 0.10
                elif method == 'FPGM':
                    drop_rate = 0.15
                elif method == 'HRank':
                    drop_rate = 0.14
                else:  # L1-Norm
                    drop_rate = 0.18  # Worst
                
                acc_mean = baseline * (1 - drop_rate * ratio)
                acc_std = np.random.uniform(0.2, 0.4)
                
                data.append({
                    'dataset': config[0],
                    'architecture': config[1],
                    'method': method,
                    'ratio': ratio,
                    'acc_mean': acc_mean,
                    'acc_std': acc_std,
                    'baseline': baseline
                })
    
    return pd.DataFrame(data)

# =============================================================================
# 12-PANEL GRID FIGURE
# =============================================================================

def create_12_panel_grid():
    """Create publication-quality 12-panel grid figure."""
    set_publication_style()
    palette = get_palette('nature')
    markers = get_markers()
    
    # Add missing methods to palette
    palette['Taylor'] = '#9B59B6'  # Purple
    palette['ACP'] = '#F39B7F'     # Salmon
    
    markers['Taylor'] = 'v'  # Triangle down
    markers['ACP'] = 'p'     # Pentagon
    
    df = generate_simulated_data()
    
    # Create 4x3 grid
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    
    # Panel configurations
    configs = [
        ('CIFAR-10', 'ResNet-20'),
        ('CIFAR-10', 'ResNet-56'),
        ('CIFAR-10', 'ResNet-110'),
        ('CIFAR-10', 'VGG-16'),
        ('CIFAR-100', 'ResNet-20'),
        ('CIFAR-100', 'ResNet-56'),
        ('CIFAR-100', 'ResNet-110'),
        ('CIFAR-100', 'VGG-16'),
        ('Tiny-200', 'ResNet-18'),
        ('Tiny-200', 'MobileV2'),
    ]
    
    methods = ['L1-Norm', 'FPGM', 'HRank', 'Taylor', 'ACP', 'GRA-CNN']
    
    # Plot first 10 panels with line charts
    for idx, (dataset, arch) in enumerate(configs):
        ax = axes[idx]
        subset = df[(df['dataset'] == dataset) & (df['architecture'] == arch)]
        
        for method in methods:
            method_data = subset[subset['method'] == method].sort_values('ratio')
            if not method_data.empty:
                ax.errorbar(
                    method_data['ratio'],
                    method_data['acc_mean'],
                    yerr=method_data['acc_std'],
                    marker=markers.get(method, 'o'),
                    color=palette.get(method, 'gray'),
                    label=method,
                    capsize=2,
                    capthick=0.8,
                    elinewidth=0.8,
                    markersize=5,
                    linewidth=1.5,
                    markeredgecolor='white',
                    markeredgewidth=0.3,
                    alpha=0.9 if method == 'GRA-CNN' else 0.7
                )
        
        # Baseline
        baseline = subset['baseline'].iloc[0] if not subset.empty else 90
        ax.axhline(y=baseline, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        # Styling
        ax.set_title(f'{dataset} / {arch}', fontsize=9, fontweight='medium', pad=5)
        ax.set_xlabel('Pruning Ratio', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.15, 0.75)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Only show legend in first panel
        if idx == 0:
            ax.legend(fontsize=6, loc='lower left', framealpha=0.9, ncol=2)
    
    # Panel 11: Summary heatmap (GRA-CNN improvement over L1)
    ax_heatmap = axes[10]
    improvement_data = []
    for config in configs:
        subset = df[(df['dataset'] == config[0]) & (df['architecture'] == config[1])]
        gra_data = subset[subset['method'] == 'GRA-CNN']
        l1_data = subset[subset['method'] == 'L1-Norm']
        if not gra_data.empty and not l1_data.empty:
            for ratio in [0.3, 0.5, 0.7]:
                gra_acc = gra_data[gra_data['ratio'] == ratio]['acc_mean'].values
                l1_acc = l1_data[l1_data['ratio'] == ratio]['acc_mean'].values
                if len(gra_acc) > 0 and len(l1_acc) > 0:
                    improvement_data.append({
                        'config': f'{config[0][:4]}-{config[1][:5]}',
                        'ratio': f'{int(ratio*100)}%',
                        'improvement': gra_acc[0] - l1_acc[0]
                    })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        # Aggregate duplicates before pivot
        imp_df = imp_df.groupby(['config', 'ratio']).agg({'improvement': 'mean'}).reset_index()
        pivot = imp_df.pivot(index='config', columns='ratio', values='improvement')
        im = ax_heatmap.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=3)
        ax_heatmap.set_xticks(range(len(pivot.columns)))
        ax_heatmap.set_xticklabels(pivot.columns, fontsize=7)
        ax_heatmap.set_yticks(range(len(pivot.index)))
        ax_heatmap.set_yticklabels(pivot.index, fontsize=6)
        ax_heatmap.set_title('GRA vs L1 Gain (%)', fontsize=9, fontweight='medium', pad=5)
        plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    
    # Panel 12: Pareto front (Accuracy vs FLOPs)
    ax_pareto = axes[11]
    for method in ['L1-Norm', 'GRA-CNN']:
        subset = df[df['method'] == method]
        flops_reduction = subset['ratio'] * 100
        accuracy = subset['acc_mean']
        ax_pareto.scatter(flops_reduction, accuracy, 
                         color=palette[method], label=method, 
                         alpha=0.5, s=20, marker=markers[method])
    ax_pareto.set_xlabel('FLOPs Reduction (%)', fontsize=8)
    ax_pareto.set_ylabel('Accuracy (%)', fontsize=8)
    ax_pareto.set_title('Accuracy-Efficiency Pareto', fontsize=9, fontweight='medium', pad=5)
    ax_pareto.legend(fontsize=7, loc='lower left')
    ax_pareto.grid(True, alpha=0.2, linestyle='--')
    ax_pareto.tick_params(labelsize=7)
    
    # Adjust layout
    fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
    
    # Add overall title
    fig.suptitle('Comprehensive Performance Comparison Across Datasets and Architectures', 
                 fontsize=12, fontweight='bold', y=1.01)
    
    return fig

def main():
    fig = create_12_panel_grid()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'fig_comprehensive_grid.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    
    plt.close(fig)
    print("12-Panel Grid Figure created successfully!")
    print("Note: Using SIMULATED data. Replace with actual experimental results.")

if __name__ == '__main__':
    main()
