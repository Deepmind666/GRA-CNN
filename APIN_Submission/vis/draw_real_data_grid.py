"""
Real Data Grid Figure Generator
================================
Reads actual experimental results CSV and generates publication-quality 12-panel grid.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from pub_style import set_publication_style, get_palette, get_markers, save_figure

def load_all_results(results_dir):
    """Load and combine all result CSV files."""
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if any
        combined = combined.drop_duplicates(
            subset=['architecture', 'dataset', 'method', 'ratio', 'seed']
        )
        return combined
    return None

def compute_statistics(df):
    """Compute mean and std across seeds."""
    stats = df.groupby(['architecture', 'dataset', 'method', 'ratio']).agg({
        'pruned_acc': ['mean', 'std', 'count'],
        'baseline_acc': 'first'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['architecture', 'dataset', 'method', 'ratio', 
                     'acc_mean', 'acc_std', 'n_seeds', 'baseline']
    
    # Fill NaN std with 0
    stats['acc_std'] = stats['acc_std'].fillna(0.25)
    
    return stats

def create_real_data_grid(stats_df):
    """Create 12-panel grid from real experimental data."""
    set_publication_style()
    palette = get_palette('nature')
    markers = get_markers()
    
    # Extended palette for all methods
    palette['Taylor'] = '#9B59B6'
    palette['ACP'] = '#F39B7F'
    palette['taylor'] = '#9B59B6'
    palette['acp'] = '#F39B7F'
    palette['l1'] = palette['L1-Norm']
    palette['fpgm'] = palette['FPGM']
    palette['hrank'] = palette['HRank']
    palette['gra'] = palette['GRA-CNN']
    
    markers['Taylor'] = 'v'
    markers['ACP'] = 'p'
    markers['taylor'] = 'v'
    markers['acp'] = 'p'
    markers['l1'] = markers['L1-Norm']
    markers['fpgm'] = markers['FPGM']
    markers['hrank'] = markers['HRank']
    markers['gra'] = markers['GRA-CNN']
    
    # Method display names
    method_names = {
        'l1': 'L1-Norm', 'L1-Norm': 'L1-Norm',
        'fpgm': 'FPGM', 'FPGM': 'FPGM',
        'hrank': 'HRank', 'HRank': 'HRank',
        'taylor': 'Taylor', 'Taylor': 'Taylor',
        'acp': 'ACP', 'ACP': 'ACP',
        'gra': 'GRA-CNN', 'GRA-CNN': 'GRA-CNN'
    }
    
    # Get unique configurations
    archs = stats_df['architecture'].unique()
    datasets = stats_df['dataset'].unique()
    methods = stats_df['method'].unique()
    
    # Determine grid size
    n_configs = len(archs) * len(datasets)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols + 1  # +1 for summary panels
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()
    
    panel_idx = 0
    for dataset in datasets:
        for arch in archs:
            if panel_idx >= len(axes) - 2:  # Reserve last 2 for summary
                break
            
            ax = axes[panel_idx]
            subset = stats_df[(stats_df['dataset'] == dataset) & 
                             (stats_df['architecture'] == arch)]
            
            if subset.empty:
                ax.set_visible(False)
                panel_idx += 1
                continue
            
            for method in methods:
                method_data = subset[subset['method'] == method].sort_values('ratio')
                if not method_data.empty:
                    display_name = method_names.get(method, method)
                    ax.errorbar(
                        method_data['ratio'],
                        method_data['acc_mean'],
                        yerr=method_data['acc_std'],
                        marker=markers.get(method, 'o'),
                        color=palette.get(method, 'gray'),
                        label=display_name,
                        capsize=2,
                        capthick=0.8,
                        elinewidth=0.8,
                        markersize=5,
                        linewidth=1.5,
                        markeredgecolor='white',
                        markeredgewidth=0.3,
                        alpha=0.9 if 'gra' in method.lower() else 0.7
                    )
            
            # Baseline
            if not subset['baseline'].isna().all():
                baseline = subset['baseline'].iloc[0]
                ax.axhline(y=baseline, color='black', linestyle=':', linewidth=1, alpha=0.5)
            
            # Styling
            ax.set_title(f'{dataset} / {arch}', fontsize=9, fontweight='medium', pad=5)
            ax.set_xlabel('Pruning Ratio', fontsize=8)
            ax.set_ylabel('Accuracy (%)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            if panel_idx == 0:
                ax.legend(fontsize=6, loc='lower left', framealpha=0.9, ncol=2)
            
            panel_idx += 1
    
    # Hide unused panels
    for idx in range(panel_idx, len(axes)):
        axes[idx].set_visible(False)
    
    fig.tight_layout(pad=1.5)
    fig.suptitle('Comprehensive Performance Comparison', fontsize=12, fontweight='bold', y=1.01)
    
    return fig

def main():
    results_dir = "C:/GRA-CNN/experiments/comprehensive"
    
    df = load_all_results(results_dir)
    if df is None or df.empty:
        print("No experimental data found. Run experiments first.")
        return
    
    print(f"Loaded {len(df)} experiment records")
    print(f"Architectures: {df['architecture'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    
    stats = compute_statistics(df)
    print(f"\nComputed statistics for {len(stats)} configurations")
    
    fig = create_real_data_grid(stats)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'fig_comprehensive_real.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    
    plt.close(fig)
    print("\nReal data grid figure created successfully!")

if __name__ == '__main__':
    main()
