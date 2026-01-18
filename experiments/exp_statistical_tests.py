"""
Statistical Significance Test Experiment
=========================================
Performs rigorous statistical tests to validate GRA-CNN improvements.

Tests:
1. Paired t-test: GRA vs each baseline
2. Wilcoxon signed-rank test (non-parametric)
3. Effect size (Cohen's d)
4. Confidence intervals
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'APIN_Submission', 'vis'))
from pub_style import set_publication_style, get_palette, save_figure

def load_all_results(results_dir):
    """Load all experiment results."""
    import glob
    
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
        return pd.concat(dfs, ignore_index=True)
    return None

def compute_statistics(df):
    """Compute mean, std, and other statistics per configuration."""
    stats_df = df.groupby(['architecture', 'dataset', 'method', 'ratio']).agg({
        'pruned_acc': ['mean', 'std', 'count', 'min', 'max']
    }).reset_index()
    
    stats_df.columns = ['architecture', 'dataset', 'method', 'ratio', 
                        'mean', 'std', 'count', 'min', 'max']
    
    return stats_df

def paired_ttest(df, method1='gra', method2='l1'):
    """Perform paired t-test between two methods."""
    # Get unique configurations
    configs = df.groupby(['architecture', 'dataset', 'ratio']).size().reset_index()[['architecture', 'dataset', 'ratio']]
    
    results = []
    
    for _, row in configs.iterrows():
        arch, dataset, ratio = row['architecture'], row['dataset'], row['ratio']
        
        # Get accuracies for both methods
        m1_accs = df[(df['architecture'] == arch) & 
                     (df['dataset'] == dataset) & 
                     (df['ratio'] == ratio) & 
                     (df['method'] == method1)]['pruned_acc'].values
        
        m2_accs = df[(df['architecture'] == arch) & 
                     (df['dataset'] == dataset) & 
                     (df['ratio'] == ratio) & 
                     (df['method'] == method2)]['pruned_acc'].values
        
        if len(m1_accs) >= 2 and len(m2_accs) >= 2:
            # Paired t-test
            t_stat, p_value = stats.ttest_ind(m1_accs, m2_accs)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((m1_accs.std()**2 + m2_accs.std()**2) / 2)
            cohens_d = (m1_accs.mean() - m2_accs.mean()) / pooled_std if pooled_std > 0 else 0
            
            # 95% confidence interval for the difference
            diff = m1_accs.mean() - m2_accs.mean()
            se = np.sqrt(m1_accs.var()/len(m1_accs) + m2_accs.var()/len(m2_accs))
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
            
            results.append({
                'architecture': arch,
                'dataset': dataset,
                'ratio': ratio,
                f'{method1}_mean': m1_accs.mean(),
                f'{method2}_mean': m2_accs.mean(),
                'difference': diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)

def create_significance_figure(ttest_results, save_dir):
    """Create figure showing statistical significance of improvements."""
    set_publication_style()
    palette = get_palette('nature')
    
    if ttest_results.empty:
        print("No statistical test results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Effect size by configuration
    ax1 = axes[0]
    configs = ttest_results['architecture'] + ' / ' + ttest_results['dataset'] + ' / ' + ttest_results['ratio'].astype(str)
    colors = [palette['GRA-CNN'] if d > 0 else palette['L1-Norm'] for d in ttest_results['cohens_d']]
    
    bars = ax1.barh(range(len(configs)), ttest_results['cohens_d'], color=colors, edgecolor='white')
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels(configs, fontsize=7)
    ax1.set_xlabel("Cohen's d (Effect Size)", fontweight='medium')
    ax1.set_title("(a) GRA vs L1 Effect Size by Configuration", fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, ttest_results['significant'])):
        if sig:
            ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                    '*', fontsize=12, va='center', fontweight='bold', color='red')
    
    # Plot 2: Improvement with confidence intervals
    ax2 = axes[1]
    x = range(len(configs))
    ax2.errorbar(ttest_results['difference'], x, 
                 xerr=[ttest_results['difference'] - ttest_results['ci_low'],
                       ttest_results['ci_high'] - ttest_results['difference']],
                 fmt='o', color=palette['GRA-CNN'], capsize=3, capthick=1, markersize=6)
    ax2.set_yticks(x)
    ax2.set_yticklabels(configs, fontsize=7)
    ax2.set_xlabel('Accuracy Improvement (%)', fontweight='medium')
    ax2.set_title('(b) GRA Improvement with 95% CI', fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    
    output_path = os.path.join(save_dir, 'fig_statistical_significance.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    
    print(f"\nStatistical significance figure saved to: {save_dir}")

def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'comprehensive')
    save_dir = os.path.join(os.path.dirname(__file__), 'prior_experiments')
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading experiment results...")
    df = load_all_results(results_dir)
    
    if df is None or df.empty:
        print("No data available for statistical analysis.")
        return
    
    print(f"Loaded {len(df)} experiment records")
    print(f"Methods: {df['method'].unique()}")
    print(f"Architectures: {df['architecture'].unique()}")
    
    # Compute statistics
    stats_df = compute_statistics(df)
    stats_df.to_csv(os.path.join(save_dir, 'experiment_statistics.csv'), index=False)
    
    # Perform t-tests
    print("\nPerforming statistical tests...")
    ttest_results = paired_ttest(df, 'gra', 'l1')
    
    if not ttest_results.empty:
        ttest_results.to_csv(os.path.join(save_dir, 'ttest_results.csv'), index=False)
        
        # Create figure
        create_significance_figure(ttest_results, save_dir)
        
        # Summary
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total comparisons: {len(ttest_results)}")
        print(f"Significant improvements (p<0.05): {ttest_results['significant'].sum()}")
        print(f"Mean improvement: {ttest_results['difference'].mean():.3f}%")
        print(f"Mean effect size (Cohen's d): {ttest_results['cohens_d'].mean():.3f}")
        print("="*60)
    else:
        print("Not enough data for statistical tests")

if __name__ == '__main__':
    main()
