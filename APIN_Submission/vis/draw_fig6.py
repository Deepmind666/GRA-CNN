import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from draw_utils import set_style, save_fig

def main():
    set_style()
    
    # Load data
    df = pd.read_csv('vis/correlation_scores.csv')
    
    # Filter layer
    layer_name = 'layer2.0.conv1'
    subset = df[df['layer'] == layer_name]
    
    if subset.empty:
        print(f"Layer {layer_name} not found, using first layer available")
        layer_name = df['layer'].iloc[0]
        subset = df[df['layer'] == layer_name]
        
    # Calculate Correlation
    p_corr, _ = pearsonr(subset['l1_score'], subset['gra_score'])
    s_corr, _ = spearmanr(subset['l1_score'], subset['gra_score'])
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=subset, x='l1_score', y='gra_score', color='#1f77b4', alpha=0.7, s=80)
    
    # Add regression line? No, scatter is enough to show orthogonality (lack of correlation)
    # But maybe a faint line
    sns.regplot(data=subset, x='l1_score', y='gra_score', scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.5})
    
    plt.xlabel('L1-Norm Score (Normalized)')
    plt.ylabel('GRA Score (Normalized)')
    plt.title(f'Metric Correlation ({layer_name})')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Add text box
    text = f'Pearson: {p_corr:.2f}\nSpearman: {s_corr:.2f}'
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    save_fig(plt.gcf(), 'APIN_Submission/fig_correlation.pdf')

if __name__ == '__main__':
    main()
