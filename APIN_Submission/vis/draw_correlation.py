import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    # Load Data
    csv_path = os.path.join(os.path.dirname(__file__), 'correlation_scores.csv')
    if not os.path.exists(csv_path):
        print("correlation_scores.csv not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Assume columns 'l1_score' and 'gra_score'
    # Normalize if needed, or assume they are normalized
    l1 = df['l1_score']
    gra = df['gra_score']
    
    # Calculate Correlations
    p_corr, _ = pearsonr(l1, gra)
    s_corr, _ = spearmanr(l1, gra)
    
    plt.figure(figsize=(6, 6))
    
    # Scatter plot
    plt.scatter(l1, gra, 
                s=20, # slightly larger than default 7? User said marker size 7. 
                      # Wait, user said "Marker size: 7" for line plots. 
                      # For scatter, usually smaller or similar. Let's use 20 area (~radius 2.5) or follow user strictness?
                      # User said "Marker size: 7". In matplotlib scatter, s is area in points^2. 
                      # plot(markersize=7) means diameter approx 7 points. Area ~ 38. 
                      # Let's use s=30.
                alpha=0.3, 
                color='purple', 
                label='Channels',
                edgecolors='none') 
    
    # Quadrants (Median lines)
    l1_med = l1.median()
    gra_med = gra.median()
    
    plt.axvline(x=l1_med, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.axhline(y=gra_med, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Annotate Quadrants
    # High L1, High GRA (Top Right)
    plt.text(0.95, 0.95, 'Important\nin Both', ha='right', va='top', transform=plt.gca().transAxes, fontsize=11)
    # Low L1, Low GRA (Bottom Left)
    plt.text(0.05, 0.05, 'Redundant\nin Both', ha='left', va='bottom', transform=plt.gca().transAxes, fontsize=11)
    # Low L1, High GRA (Top Left) - Semantic but Low Magnitude
    plt.text(0.05, 0.95, 'GRA-Specific\nImportance', ha='left', va='top', transform=plt.gca().transAxes, color='#d62728', fontsize=11, fontweight='bold')
    
    # Annotate Correlation
    plt.title(f'Pearson: {p_corr:.2f}, Spearman: {s_corr:.2f}', fontsize=12)
    
    plt.xlabel('L1-Norm Score (Normalized)')
    plt.ylabel('GRA Score (Normalized)')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig6.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
