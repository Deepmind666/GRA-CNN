import pandas as pd
import matplotlib.pyplot as plt
import os
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    csv_path = os.path.join(os.path.dirname(__file__), 'results_rho_ablation_detailed.csv')
    if not os.path.exists(csv_path): 
        print("results_rho_ablation_detailed.csv not found")
        return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(6, 5))
    
    # ResNet-20
    subset_r20 = df[df['model'] == 'resnet20'].sort_values('rho')
    plt.plot(subset_r20['rho'], subset_r20['acc'], 
             marker='o', 
             label='ResNet-20 (Ratio=0.5)',
             color='#1f77b4', # Blue
             linewidth=2.2)
    
    # ResNet-56
    subset_r56 = df[df['model'] == 'resnet56'].sort_values('rho')
    plt.plot(subset_r56['rho'], subset_r56['acc'], 
             marker='s', 
             label='ResNet-56 (Ratio=0.5)',
             color='#d62728', # Red
             linewidth=2.2)
    
    plt.xlabel(r'Resolution Coefficient ($\rho$)')
    plt.ylabel('Top-1 Accuracy (%)')
    
    plt.legend()
    plt.grid(True)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig7_detailed.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
