"""
ρ参数敏感性实验 + 逐层剪枝效果可视化
=========================================
基于评审意见：分析ρ对剪枝效果的影响，并展示逐层GRA分数分布
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from scipy import stats

# 设置学术期刊字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

sys.path.insert(0, r'C:\GRA-CNN')
from models.resnet_cifar import resnet56

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'


def compute_gra_scores(model, loader, rho=0.5, num_batches=8):
    """计算所有通道的GRA分数"""
    model.eval()
    
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
            if i >= num_batches: break
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            all_logits.append(out[torch.arange(len(lbls)), lbls].cpu())
            for n, _ in conv_layers:
                all_acts[n].append(activations[n].mean(dim=(2,3)).cpu())
    
    for h in hooks: h.remove()
    
    all_logits = torch.cat(all_logits).numpy()
    logits_norm = (all_logits - all_logits.min()) / (all_logits.max() - all_logits.min() + 1e-8)
    
    layer_scores = {}
    
    for n, m in conv_layers:
        acts = torch.cat(all_acts[n], dim=0).numpy()
        gra_scores = np.zeros(acts.shape[1])
        
        for c in range(acts.shape[1]):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            delta = np.abs(act_norm - logits_norm)
            delta_min = delta.min()
            delta_max = delta.max()
            xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
            gra_scores[c] = xi.mean()
        
        layer_scores[n] = gra_scores
    
    return layer_scores


def experiment_rho_sensitivity():
    """实验1: ρ参数敏感性分析"""
    print("=" * 60)
    print("实验1: ρ参数敏感性分析")
    print("=" * 60)
    
    model = resnet56(num_classes=10)
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for rho in rho_values:
        print(f"  测试 ρ = {rho}...")
        scores = compute_gra_scores(model, loader, rho=rho, num_batches=8)
        
        # 计算统计量
        all_scores = np.concatenate([s for s in scores.values()])
        results[rho] = {
            'mean': all_scores.mean(),
            'std': all_scores.std(),
            'min': all_scores.min(),
            'max': all_scores.max(),
            'scores': all_scores
        }
        print(f"    Mean GRA: {all_scores.mean():.4f}, Std: {all_scores.std():.4f}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) GRA分数均值和标准差随ρ变化
    ax1 = axes[0]
    means = [results[r]['mean'] for r in rho_values]
    stds = [results[r]['std'] for r in rho_values]
    
    ax1.errorbar(rho_values, means, yerr=stds, fmt='o-', color='#2E86AB', 
                 linewidth=2, markersize=8, capsize=5, capthick=2, label='Mean ± Std')
    ax1.fill_between(rho_values, np.array(means)-np.array(stds), 
                     np.array(means)+np.array(stds), alpha=0.2, color='#2E86AB')
    ax1.axvspan(0.4, 0.6, alpha=0.15, color='green', label='Stable region')
    ax1.set_xlabel('Sensitivity Coefficient ρ', fontsize=12)
    ax1.set_ylabel('Mean GRA Score', fontsize=12)
    ax1.set_title('(a) GRA Score vs ρ', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.05, 0.95)
    
    # (b) 不同ρ下的分数分布箱线图
    ax2 = axes[1]
    box_data = [results[r]['scores'] for r in rho_values]
    bp = ax2.boxplot(box_data, labels=[f'{r}' for r in rho_values], patch_artist=True)
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(rho_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Sensitivity Coefficient ρ', fontsize=12)
    ax2.set_ylabel('GRA Score Distribution', fontsize=12)
    ax2.set_title('(b) Score Distribution across ρ', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # (c) 通道区分度随ρ变化 (用标准差/均值衡量)
    ax3 = axes[2]
    cv = [results[r]['std'] / results[r]['mean'] for r in rho_values]  # 变异系数
    ax3.plot(rho_values, cv, 'o-', color='#E74C3C', linewidth=2, markersize=8)
    ax3.axhline(y=np.mean(cv), color='gray', linestyle='--', linewidth=1.5, 
                label=f'Mean CV = {np.mean(cv):.3f}')
    ax3.set_xlabel('Sensitivity Coefficient ρ', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation', fontsize=12)
    ax3.set_title('(c) Channel Discriminability', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0.05, 0.95)
    
    plt.tight_layout()
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_rho_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_rho_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ 保存: fig_rho_sensitivity.pdf/png")
    return results


def experiment_layerwise_analysis():
    """实验2: 逐层GRA分数可视化"""
    print("\n" + "=" * 60)
    print("实验2: 逐层剪枝效果分析")
    print("=" * 60)
    
    model = resnet56(num_classes=10)
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 获取GRA和L1分数
    gra_scores = compute_gra_scores(model, loader, rho=0.5, num_batches=8)
    
    # 计算L1分数
    l1_scores = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.out_channels > 3:
            l1 = m.weight.data.abs().sum(dim=(1,2,3)).cpu().numpy()
            l1 = l1 / l1.max()  # 归一化
            l1_scores[n] = l1
    
    # 创建逐层可视化
    layer_names = list(gra_scores.keys())
    n_layers = len(layer_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) 逐层GRA分数均值
    ax1 = axes[0, 0]
    gra_means = [gra_scores[n].mean() for n in layer_names]
    l1_means = [l1_scores[n].mean() for n in layer_names]
    
    x = np.arange(n_layers)
    width = 0.35
    ax1.bar(x - width/2, gra_means, width, label='GRA', color='#E74C3C', alpha=0.8)
    ax1.bar(x + width/2, l1_means, width, label='L1', color='#3498DB', alpha=0.8)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mean Importance Score', fontsize=12)
    ax1.set_title('(a) Layer-wise Importance: GRA vs L1', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xticks(x[::5])
    ax1.set_xticklabels([str(i) for i in x[::5]])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # (b) 逐层相关性
    ax2 = axes[0, 1]
    layer_corrs = []
    for n in layer_names:
        r = np.corrcoef(gra_scores[n], l1_scores[n])[0, 1]
        layer_corrs.append(r if not np.isnan(r) else 0)
    
    colors = ['#E74C3C' if abs(c) < 0.1 else '#F39C12' if abs(c) < 0.3 else '#27AE60' for c in layer_corrs]
    ax2.bar(x, layer_corrs, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axhline(y=np.mean(layer_corrs), color='green', linestyle='--', linewidth=2,
                label=f'Mean r = {np.mean(layer_corrs):.3f}')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Pearson Correlation (GRA vs L1)', fontsize=12)
    ax2.set_title('(b) Layer-wise GRA-L1 Correlation', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xticks(x[::5])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # (c) 深度依赖性分析
    ax3 = axes[1, 0]
    depth_groups = [
        ('Early (1-18)', layer_names[:18]),
        ('Middle (19-36)', layer_names[18:36]),
        ('Late (37-54)', layer_names[36:])
    ]
    
    gra_by_depth = []
    l1_by_depth = []
    for _, layers in depth_groups:
        gra_by_depth.append(np.mean([gra_scores[n].mean() for n in layers if n in gra_scores]))
        l1_by_depth.append(np.mean([l1_scores[n].mean() for n in layers if n in l1_scores]))
    
    x_depth = np.arange(len(depth_groups))
    ax3.bar(x_depth - 0.2, gra_by_depth, 0.35, label='GRA', color='#E74C3C')
    ax3.bar(x_depth + 0.2, l1_by_depth, 0.35, label='L1', color='#3498DB')
    ax3.set_xticks(x_depth)
    ax3.set_xticklabels([g[0] for g in depth_groups])
    ax3.set_xlabel('Network Depth', fontsize=12)
    ax3.set_ylabel('Mean Importance Score', fontsize=12)
    ax3.set_title('(c) Depth-dependent Importance Pattern', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # (d) 通道重要性热力图 (选取部分层)
    ax4 = axes[1, 1]
    selected_layers = layer_names[::8][:6]  # 每8层选1个
    
    heatmap_data = []
    for n in selected_layers:
        gra_norm = (gra_scores[n] - gra_scores[n].min()) / (gra_scores[n].max() - gra_scores[n].min() + 1e-8)
        heatmap_data.append(gra_norm[:16])  # 取前16个通道
    
    heatmap_data = np.array(heatmap_data)
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r')
    ax4.set_yticks(range(len(selected_layers)))
    ax4.set_yticklabels([f'L{i*8+1}' for i in range(len(selected_layers))])
    ax4.set_xlabel('Channel Index (first 16)', fontsize=12)
    ax4.set_ylabel('Layer', fontsize=12)
    ax4.set_title('(d) GRA Score Heatmap', fontweight='bold', fontsize=13)
    plt.colorbar(im, ax=ax4, label='Normalized GRA')
    
    plt.tight_layout()
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_layerwise_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_layerwise_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 分析了 {n_layers} 层")
    print(f"✓ 平均层间相关性: {np.mean(layer_corrs):.4f}")
    print("✓ 保存: fig_layerwise_analysis.pdf/png")


if __name__ == "__main__":
    print("=" * 60)
    print("GRA-CNN 高质量实验可视化生成")
    print("=" * 60)
    
    experiment_rho_sensitivity()
    experiment_layerwise_analysis()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
