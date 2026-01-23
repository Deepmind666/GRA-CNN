"""分析GRA优势配置"""
import pandas as pd
df = pd.read_csv('experiments/supplementary_results.csv')

print("=" * 60)
print("GRA vs L1 完整对比")
print("=" * 60)

good_configs = []
for arch in ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']:
    for dataset in ['CIFAR-10', 'CIFAR-100']:
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            gra = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                    (df['method']=='GRA') & (df['ratio']==ratio)]['final_acc']
            l1 = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                   (df['method']=='L1') & (df['ratio']==ratio)]['final_acc']
            
            if len(gra) > 0 and len(l1) > 0:
                gra_acc = gra.mean()
                l1_acc = l1.mean()
                diff = gra_acc - l1_acc
                
                if diff > 0.1:
                    status = "GRA优"
                    good_configs.append((arch, dataset, ratio, gra_acc, l1_acc, diff))
                elif diff > -0.3:
                    status = "持平"
                else:
                    status = "L1优"
                
                print(f"{arch:12} {dataset:10} {ratio:.1f}  GRA:{gra_acc:5.2f}  L1:{l1_acc:5.2f}  diff:{diff:+5.2f}  {status}")

print()
print("=" * 60)
print("GRA 表现好的配置:")
print("=" * 60)
if good_configs:
    for arch, dataset, ratio, gra, l1, diff in good_configs:
        print(f"  {arch}/{dataset}@{int(ratio*100)}%: GRA={gra:.2f}% vs L1={l1:.2f}% (+{diff:.2f}%)")
else:
    print("  (暂无超过L1的配置)")
