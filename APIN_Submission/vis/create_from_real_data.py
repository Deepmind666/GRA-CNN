"""
Generate Figures from REAL Experimental Data
============================================
Uses actual experiment results from final_consolidated_results.csv
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# ============================================================================
# LOAD REAL EXPERIMENTAL DATA
# ============================================================================

print("Loading real experimental data...")
df = pd.read_csv(r'C:\GRA-CNN\experiments\final_consolidated_results.csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Architectures: {df['architecture'].unique()}")
print(f"Datasets: {df['dataset'].unique()}")
print(f"Methods: {df['method'].unique()}")
print(f"Ratios: {sorted(df['ratio'].unique())}")
print()
print("Sample data:")
print(df.head(20))
print()

# ============================================================================
# COLOR PALETTE
# ============================================================================

COLORS = {
    'gra': '#E64B35',
    'l1': '#4DBBD5',
    'fpgm': '#00A087',
    'hrank': '#3C5488',
    'baseline': '#8E8E93',
}

MARKERS = {'gra': 'o', 'l1': 's', 'fpgm': '^', 'hrank': 'D'}
LINESTYLES = {'gra': '-', 'l1': '--', 'fpgm': '-.', 'hrank': ':'}

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_data(arch, dataset, method):
    """Extract accuracy data for given configuration."""
    subset = df[
        (df['architecture'].str.lower().str.contains(arch.lower())) &
        (df['dataset'].str.lower().str.contains(dataset.lower())) &
        (df['method'].str.lower() == method.lower())
    ].copy()
    subset = subset.sort_values('ratio')
    return subset['ratio'].values, subset['pruned_acc'].values

def get_baseline(arch, dataset):
    """Get baseline accuracy."""
    subset = df[
        (df['architecture'].str.lower().str.contains(arch.lower())) &
        (df['dataset'].str.lower().str.contains(dataset.lower())) &
        (df['method'].str.lower() == 'baseline')
    ]
    if len(subset) > 0:
        return subset['pruned_acc'].values[0]
    return None

# ============================================================================
# ANALYZE AVAILABLE DATA
# ============================================================================

print("="*60)
print("AVAILABLE EXPERIMENTAL DATA SUMMARY")
print("="*60)

for arch in df['architecture'].unique():
    for dataset in df['dataset'].unique():
        subset = df[(df['architecture']==arch) & (df['dataset']==dataset)]
        if len(subset) > 0:
            print(f"\n{arch} on {dataset}:")
            for method in subset['method'].unique():
                m_subset = subset[subset['method']==method]
                ratios = sorted(m_subset['ratio'].unique())
                accs = m_subset.sort_values('ratio')['pruned_acc'].values
                print(f"  {method}: ratios={ratios}, accs={accs}")

print("\n" + "="*60)

# ============================================================================
# CREATE FIGURE FROM REAL DATA
# ============================================================================

def create_real_data_figure():
    """Create figure using REAL experimental data."""
    set_style()
    
    # Find all unique arch-dataset combinations
    combos = df.groupby(['architecture', 'dataset']).size().reset_index()
    combos = [(row['architecture'], row['dataset']) for _, row in combos.iterrows()]
    
    # Filter to have at least 2 methods with pruning data
    valid_combos = []
    for arch, dataset in combos:
        subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['ratio']>0)]
        methods = subset['method'].unique()
        if len(methods) >= 2:
            valid_combos.append((arch, dataset))
    
    print(f"\nValid combinations for plotting: {valid_combos}")
    
    if len(valid_combos) == 0:
        print("ERROR: No valid data combinations found!")
        return
    
    n_panels = len(valid_combos)
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4*n_cols, 3.5*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.35, wspace=0.28)
    
    for idx, (arch, dataset) in enumerate(valid_combos):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['ratio']>0)]
        methods = subset['method'].unique()
        
        for method in methods:
            m_data = subset[subset['method']==method].sort_values('ratio')
            ratios = m_data['ratio'].values
            accs = m_data['pruned_acc'].values
            
            color = COLORS.get(method.lower(), 'gray')
            marker = MARKERS.get(method.lower(), 'o')
            ls = LINESTYLES.get(method.lower(), '-')
            
            ax.plot(ratios, accs, marker=marker, color=color, linestyle=ls,
                   label=method.upper(), linewidth=1.8, markersize=5,
                   markeredgecolor='white', markeredgewidth=0.6)
        
        # Baseline
        baseline = get_baseline(arch, dataset)
        if baseline:
            ax.axhline(y=baseline, color=COLORS['baseline'], linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_title(f'({chr(97+idx)}) {arch} on {dataset}', fontweight='bold', fontsize=9)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        if row == n_rows - 1:
            ax.set_xlabel('Pruning Ratio')
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=6, ncol=2)
    
    fig.suptitle('Figure: Pruning Results from Real Experiments', 
                fontweight='bold', fontsize=11, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_data.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_data.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Created: fig_real_data.pdf/png with {n_panels} panels")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    create_real_data_figure()
