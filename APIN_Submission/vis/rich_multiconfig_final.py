"""
GRA vs L1 正交性分析 - 2×3 标准学术图表布局
==============================================
6个图表面板，无纯文字区域
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from scipy import stats

sys.path.insert(0, r'C:\GRA-CNN')
from models.resnet_cifar import resnet20, resnet56

# 设置学术期刊字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'


def analyze_config(model_fn, num_classes, dataset_cls, mean, std, name):
    """分析单个配置"""
    model = model_fn(num_classes=num_classes)
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = dataset_cls(root=DATA_DIR, train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    conv_layers = [(n, m) for n, m in model.named_modules() 
                   if isinstance(m, nn.Conv2d) and m.out_channels > 3]
    
    activations = {}
    def make_hook(n):
        def hook(m, i, o): activations[n] = o.detach()
        return hook
    
    hooks = [m.register_forward_hook(make_hook(n)) for n, m in conv_layers]
    
    all_acts = {n: [] for n, _ in conv_layers}
    all_logits = []
    
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            if i >= 8: break
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            all_logits.append(out[torch.arange(len(lbls)), lbls].cpu())
            for n, _ in conv_layers:
                all_acts[n].append(activations[n].mean(dim=(2,3)).cpu())
    
    for h in hooks: h.remove()
    
    all_logits = torch.cat(all_logits).numpy()
    logits_norm = (all_logits - all_logits.min()) / (all_logits.max() - all_logits.min() + 1e-8)
    
    all_gra, all_l1 = [], []
    layer_corrs = []
    
    for n, m in conv_layers:
        acts = torch.cat(all_acts[n], dim=0).numpy()
        weights = m.weight.data.cpu().numpy()
        l1 = np.abs(weights).sum(axis=(1,2,3))
        l1 = l1 / l1.max()
        
        gra = np.zeros(acts.shape[1])
        for c in range(acts.shape[1]):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            delta = np.abs(act_norm - logits_norm)
            xi = (delta.min() + 0.5 * delta.max()) / (delta + 0.5 * delta.max() + 1e-8)
            gra[c] = xi.mean()
        
        # 计算该层的相关系数
        if len(gra) > 2:
            layer_r = np.corrcoef(gra, l1)[0, 1]
            layer_corrs.append(layer_r if not np.isnan(layer_r) else 0)
        
        all_gra.extend(gra)
        all_l1.extend(l1)
    
    all_gra, all_l1 = np.array(all_gra), np.array(all_l1)
    r = np.corrcoef(all_gra, all_l1)[0, 1]
    
    return {
        'gra': all_gra, 'l1': all_l1, 'r': r, 'n': len(all_gra), 
        'name': name, 'layer_corrs': np.array(layer_corrs)
    }


def main():
    print("=" * 60)
    print("GRA vs L1 正交性分析 - 2×3 标准学术布局")
    print("=" * 60)
    
    configs = [
        (resnet20, 10, torchvision.datasets.CIFAR10, 
         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 'ResNet-20\nCIFAR-10'),
        (resnet56, 10, torchvision.datasets.CIFAR10,
         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 'ResNet-56\nCIFAR-10'),
        (resnet20, 100, torchvision.datasets.CIFAR100,
         (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 'ResNet-20\nCIFAR-100'),
        (resnet56, 100, torchvision.datasets.CIFAR100,
         (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 'ResNet-56\nCIFAR-100'),
    ]
    
    results = []
    for model_fn, num_classes, dataset_cls, mean, std, name in configs:
        print(f"\n分析: {name.replace(chr(10), ' / ')}...")
        try:
            r = analyze_config(model_fn, num_classes, dataset_cls, mean, std, name)
            results.append(r)
            print(f"  ✓ n={r['n']}, r={r['r']:.4f}")
        except Exception as e:
            print(f"  ✗ {e}")
    
    if not results:
        print("No results!")
        return
    
    # ========== 创建 2×3 标准学术图表布局 ==========
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    
    # 统一 Y 轴范围
    all_gra_values = np.concatenate([r['gra'] for r in results])
    y_min = max(0.45, np.percentile(all_gra_values, 1) - 0.05)
    y_max = min(0.85, np.percentile(all_gra_values, 99) + 0.05)
    
    # ===== 面板 (a)-(d): 4个散点图 =====
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ['(a)', '(b)', '(c)', '(d)']
    
    for idx, r in enumerate(results[:4]):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        ax.scatter(r['l1'], r['gra'], s=12, alpha=0.5, c='#D62728', edgecolors='none')
        slope, intercept, rv, _, _ = stats.linregress(r['l1'], r['gra'])
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, slope * x_line + intercept, 'b--', linewidth=2)
        
        ax.set_xlabel('L1 Score (Weight Magnitude)')
        ax.set_ylabel('GRA Score (Semantic Alignment)')
        ax.set_title(f"{labels[idx]} {r['name']}\nn = {r['n']}, r = {r['r']:.3f}", fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(y_min, y_max)
    
    # ===== 面板 (e): 相关系数柱状图 =====
    ax_bar = fig.add_subplot(gs[0, 2])
    names = [r['name'].replace('\n', '/') for r in results]
    corrs = [r['r'] for r in results]
    colors = ['#E74C3C' if abs(c) < 0.05 else '#F39C12' for c in corrs]
    
    bars = ax_bar.barh(range(len(corrs)), corrs, color=colors, edgecolor='black', height=0.6)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel('Pearson Correlation (r)')
    ax_bar.set_title('(e) GRA-L1 Correlation\nAcross Configurations', fontweight='bold')
    ax_bar.axvline(x=0, color='black', linewidth=0.8)
    ax_bar.axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_bar.axvline(x=-0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_bar.set_xlim(-0.15, 0.15)
    ax_bar.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for i, c in enumerate(corrs):
        ax_bar.text(c + 0.008 if c >= 0 else c - 0.03, i, f'{c:.3f}', va='center', fontweight='bold', fontsize=10)
    
    # ===== 面板 (f): 逐层相关系数分布直方图 =====
    ax_hist = fig.add_subplot(gs[1, 2])
    all_layer_corrs = np.concatenate([r['layer_corrs'] for r in results])
    
    ax_hist.hist(all_layer_corrs, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
    ax_hist.axvline(x=0, color='red', linewidth=2, linestyle='-', label='r = 0')
    ax_hist.axvline(x=np.mean(all_layer_corrs), color='green', linewidth=2, linestyle='--', 
                    label=f'Mean = {np.mean(all_layer_corrs):.3f}')
    ax_hist.set_xlabel('Layer-wise Correlation (r)')
    ax_hist.set_ylabel('Number of Layers')
    ax_hist.set_title(f'(f) Distribution of Layer-wise\nGRA-L1 Correlation (n={len(all_layer_corrs)} layers)', fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=9)
    ax_hist.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 数据来源标注
    total_channels = sum(r['n'] for r in results)
    fig.text(0.5, 0.01, 
             f'Data: Real analysis on {total_channels} channels from CIFAR-10/100 test sets (512 images per config)',
             fontsize=9, style='italic', color='gray', ha='center')
    
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_rich_multiconfig.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_rich_multiconfig.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"完成: 2×3 标准学术布局")
    print(f"- 4个散点图 (a-d)")
    print(f"- 1个柱状图 (e)")
    print(f"- 1个直方图 (f)")
    print("=" * 60)


if __name__ == "__main__":
    main()
