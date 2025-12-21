import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data
data = pd.read_csv('final_results.csv')

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Filter for ResNet-20 comparison
df_r20 = data[data['Architecture'] == 'resnet20']

# Plot ResNet-20 Pruning Ratio vs Accuracy
plt.plot(df_r20[df_r20['Method'] == 'gra']['PruningRatio'], 
         df_r20[df_r20['Method'] == 'gra']['Accuracy'], 
         marker='o', label='GRA-CNN (Ours)', linewidth=2.5, markersize=8)

# Plot Baseline L1 (only one point for now, extend line for visualization if needed)
# For visualization, let's assume L1 drops faster (simulated for visual if only 1 point exists)
# But here we plot the actual point
plt.scatter(df_r20[df_r20['Method'] == 'l1']['PruningRatio'], 
            df_r20[df_r20['Method'] == 'l1']['Accuracy'], 
            color='red', marker='x', s=100, label='L1-Norm (Baseline)', zorder=5)

plt.title('Pruning Sensitivity Analysis on ResNet-20 (CIFAR-10)', fontsize=14)
plt.xlabel('Pruning Ratio', fontsize=12)
plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Save plot
plt.savefig('../APIN_Submission/fig_results.pdf', bbox_inches='tight')
plt.savefig('fig_results.png', bbox_inches='tight')
print("Figure saved to ../APIN_Submission/fig_results.pdf")
