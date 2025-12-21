import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    
    df = pd.read_csv('vis/results.csv')
    
    # Filter
    subset = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-20') & (df['method'] == 'GRA-CNN')]
    subset = subset[subset['prune_ratio'] == 0.5] # Fix ratio to 0.5
    
    if subset.empty:
        # Try to find any ratio if 0.5 missing, or just plot what we have
        subset = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-20') & (df['method'] == 'GRA-CNN')]
        # If we have multiple ratios, this plot is confusing. We need Acc vs Rho for a fixed ratio.
        # Let's group by ratio
        ratios = subset['prune_ratio'].unique()
        target_ratio = 0.5 if 0.5 in ratios else ratios[0]
        subset = subset[subset['prune_ratio'] == target_ratio]
    
    subset = subset.sort_values('rho')
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(subset['rho'], subset['accuracy'], marker='o', color='#d62728')
    
    plt.xlabel(r'Distinguishing Coefficient ($\rho$)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Sensitivity to $\\rho$ (ResNet-20, Ratio {subset["prune_ratio"].iloc[0]})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_fig(plt.gcf(), 'APIN_Submission/fig_rho_ablation.pdf')

if __name__ == '__main__':
    main()
