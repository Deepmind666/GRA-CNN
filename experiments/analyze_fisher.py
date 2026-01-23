"""分析GRA-Fisher最终效果"""
import pandas as pd
df = pd.read_csv('experiments/supplementary_results.csv')

print('GRA-Fisher 完整结果分析')
print('='*60)

configs = [
    ('ResNet-20', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.3),
    ('ResNet-56', 'CIFAR-10', 0.5),
]

wins = 0
total = 0
for arch, dataset, ratio in configs:
    gra = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
             (df['method']=='GRA') & (df['ratio']==ratio)]
    l1 = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
            (df['method']=='L1') & (df['ratio']==ratio)]
    
    if len(gra) > 0 and len(l1) > 0:
        gra_acc = gra.iloc[-1]['final_acc']
        l1_acc = l1['final_acc'].mean()
        diff = gra_acc - l1_acc
        total += 1
        if diff > 0:
            wins += 1
            status = 'WIN'
        else:
            status = 'LOSS' if diff < -0.3 else 'TIE'
        print(f'{arch}/{dataset}@{int(ratio*100)}%: GRA={gra_acc:.2f} L1={l1_acc:.2f} ({diff:+.2f}) [{status}]')

print()
print(f'GRA胜出: {wins}/{total}')
