"""分析GRA-Gradient改进效果"""
import pandas as pd
df = pd.read_csv('experiments/supplementary_results.csv')

print("=" * 70)
print("GRA-Gradient vs L1 完整对比")
print("=" * 70)
print()

configs = [
    ("ResNet-20", "CIFAR-10", 0.5),
    ("ResNet-56", "CIFAR-10", 0.3),
    ("ResNet-56", "CIFAR-10", 0.5),
    ("ResNet-56", "CIFAR-10", 0.7),
    ("VGG-16", "CIFAR-10", 0.5),
    ("ResNet-56", "CIFAR-100", 0.5),
]

print(f"{'配置':<35} {'GRA-Grad':>10} {'L1':>10} {'差值':>10}  状态")
print("-" * 70)

total_diff = 0
wins = 0
for arch, dataset, ratio in configs:
    gra = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
             (df['method']=='GRA') & (df['ratio']==ratio)]
    l1 = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
            (df['method']=='L1') & (df['ratio']==ratio)]
    
    if len(gra) > 0 and len(l1) > 0:
        gra_acc = gra.iloc[-1]['final_acc']  # 最新结果
        l1_acc = l1['final_acc'].mean()  # L1平均
        diff = gra_acc - l1_acc
        total_diff += diff
        
        if diff > 0:
            status = "✓ GRA优"
            wins += 1
        elif diff > -0.5:
            status = "≈ 接近"
        else:
            status = "✗ L1优"
        
        print(f"{arch}/{dataset}@{int(ratio*100)}%{'':<10} {gra_acc:>9.2f}% {l1_acc:>9.2f}% {diff:>+9.2f}%  {status}")

print("-" * 70)
print(f"平均差距: {total_diff/len(configs):+.2f}%")
print(f"GRA胜出配置: {wins}/{len(configs)}")
print()
print("=" * 70)
print("结论: GRA-Gradient 算法改进效果")
print("=" * 70)
