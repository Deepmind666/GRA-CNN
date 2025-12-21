import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 1. Load Data
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    if not os.path.exists(csv_path):
        csv_path = 'results.csv'
    
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-20, GRA-CNN, Prune Ratio 0.5
    df = df[(df['dataset'] == 'CIFAR-10') & 
            (df['model'] == 'ResNet-20') & 
            (df['method'] == 'GRA-CNN') & 
            (df['prune_ratio'] == 0.5)]
            
    # Sort by rho
    df = df.sort_values('rho')
    
    # 2. Setup Style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.figure(figsize=(6, 5))
    
    # 3. Plot
    plt.plot(df['rho'], df['accuracy'], marker='D', color='red', label='GRA-CNN')
            
    # 4. Axes & Ticks
    # X-axis ticks at {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    plt.xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    # Y-axis zoom
    # Find min/max to set limits smart
    min_acc = df['accuracy'].min()
    max_acc = df['accuracy'].max()
    margin = 0.2
    plt.ylim(min_acc - margin, max_acc + margin)
    
    plt.xlabel(r'Resolution Coefficient ($\rho$)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title(r'Sensitivity of $\rho$ (ResNet-20)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 5. Save
    save_path = os.path.join(os.path.dirname(__file__), '../fig7.pdf') # Mapping to fig7 (rho)
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
