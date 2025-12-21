import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, default='gra_vs_l1_scores.csv', help='input csv file')
    parser.add_argument('--output-dir', type=str, default='../APIN_Submission', help='output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return

    df = pd.read_csv(args.input_csv)
    
    # Filter for a specific layer to make the plot clear (e.g., layer2.0.conv1)
    # If layer names are generic, pick one with enough channels.
    # ResNet20 has layer1, layer2, layer3.
    target_layer = 'layer2.0.conv1' 
    
    # Check if target layer exists
    if target_layer not in df['Layer'].unique():
        # Fallback to first available layer
        target_layer = df['Layer'].unique()[0]
        
    subset = df[df['Layer'] == target_layer]
    
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    
    # Scatter plot
    sns.scatterplot(data=subset, x='L1_Score', y='GRA_Score', s=100, color='b', alpha=0.7, edgecolor='k')
    
    # Add regression line to check correlation
    sns.regplot(data=subset, x='L1_Score', y='GRA_Score', scatter=False, color='r', line_kws={'linestyle':'--'})
    
    plt.title(f'Comparison of Feature Importance Metrics\n({target_layer})', fontsize=14)
    plt.xlabel('Normalized L1-Norm Score', fontsize=12)
    plt.ylabel('Normalized GRA Score', fontsize=12)
    
    # Annotate low L1 but high GRA points (if any)
    # This highlights the "Difference"
    # Find points where L1 < 0.3 and GRA > 0.6
    interesting_points = subset[(subset['L1_Score'] < 0.4) & (subset['GRA_Score'] > 0.6)]
    for _, row in interesting_points.iterrows():
        plt.text(row['L1_Score']+0.02, row['GRA_Score'], f"Ch{int(row['Channel_Index'])}", fontsize=9)

    plt.tight_layout()
    
    out_path = os.path.join(args.output_dir, 'fig_correlation.pdf')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    # Also save png for preview
    plt.savefig(out_path.replace('.pdf', '.png'))

if __name__ == '__main__':
    main()
