"""
科学价值可视化: GRA vs L1 深度对比分析
========================================
展示两种方法选择通道的本质差异
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
import os

sys.path.insert(0, r'C:\GRA-CNN')
from models.resnet_cifar import resnet56

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'


def compute_gra_and_l1_scores(model, dataloader, num_batches=10, rho=0.5):
    """同时计算 GRA 和 L1 分数"""
    model.eval()
    model.to(DEVICE)
    
    conv_layers = []
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 3:
            conv_layers.append((name, module))
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # 收集激活和 logits
    all_activations = {name: [] for name, _ in conv_layers}
    all_logits = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            batch_logits = outputs[torch.arange(len(labels)), labels]
            all_logits.append(batch_logits.cpu())
            
            for name, _ in conv_layers:
                act = activations[name]
                act_pooled = act.mean(dim=(2, 3))
                all_activations[name].append(act_pooled.cpu())
    
    for h in hooks:
        h.remove()
    
    all_logits = torch.cat(all_logits).numpy()
    
    gra_scores = {}
    l1_scores = {}
    
    for name, module in conv_layers:
        acts = torch.cat(all_activations[name], dim=0).numpy()
        n_samples, n_channels = acts.shape
        
        # L1 分数 (权重 L1 范数)
        weights = module.weight.data.cpu().numpy()
        l1 = np.abs(weights).sum(axis=(1, 2, 3))
        l1_scores[name] = l1 / l1.max()  # 归一化到 0-1
        
        # GRA 分数
        logits_norm = (all_logits - all_logits.min()) / (all_logits.max() - all_logits.min() + 1e-8)
        gra = np.zeros(n_channels)
        for c in range(n_channels):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            delta = np.abs(act_norm - logits_norm)
            delta_min, delta_max = delta.min(), delta.max()
            xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
            gra[c] = xi.mean()
        gra_scores[name] = gra
    
    return gra_scores, l1_scores, conv_layers


def create_comparison_figure(gra_scores, l1_scores, conv_layers, prune_ratio=0.5):
    """创建 GRA vs L1 对比分析图"""
    
    # 选择一个有代表性的中间层
    target_layer = conv_layers[len(conv_layers)//2][0]
    gra = gra_scores[target_layer]
    l1 = l1_scores[target_layer]
    n_channels = len(gra)
    n_prune = int(n_channels * prune_ratio)
    
    # 各方法选择要剪枝的通道 (分数最低的)
    gra_prune_idx = set(np.argsort(gra)[:n_prune])
    l1_prune_idx = set(np.argsort(l1)[:n_prune])
    
    # 分析重叠和差异
    both_prune = gra_prune_idx & l1_prune_idx  # 都剪
    only_gra = gra_prune_idx - l1_prune_idx    # 只有 GRA 剪
    only_l1 = l1_prune_idx - gra_prune_idx      # 只有 L1 剪
    neither = set(range(n_channels)) - gra_prune_idx - l1_prune_idx  # 都保留
    
    # ========== 图 1: 散点图 + 象限分析 ==========
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8])
    
    # 子图 1: GRA vs L1 散点图
    ax1 = fig.add_subplot(gs[0])
    
    colors = []
    for i in range(n_channels):
        if i in both_prune:
            colors.append('#888888')  # 灰色: 都剪
        elif i in only_gra:
            colors.append('#D62728')  # 红色: 只有 GRA 剪 (语义冗余)
        elif i in only_l1:
            colors.append('#1F77B4')  # 蓝色: 只有 L1 剪 (结构弱但语义重要!)
        else:
            colors.append('#2CA02C')  # 绿色: 都保留
    
    ax1.scatter(l1, gra, c=colors, alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
    
    # 添加阈值线
    l1_thresh = np.sort(l1)[n_prune]
    gra_thresh = np.sort(gra)[n_prune]
    ax1.axvline(x=l1_thresh, color='blue', linestyle='--', alpha=0.5, label='L1 threshold')
    ax1.axhline(y=gra_thresh, color='red', linestyle='--', alpha=0.5, label='GRA threshold')
    
    # 标注四个象限
    ax1.text(0.15, 0.85, 'Semantic\nFilters', fontsize=10, fontweight='bold', 
             color='#1F77B4', transform=ax1.transAxes)
    ax1.text(0.75, 0.15, 'Redundant\nChannels', fontsize=10, fontweight='bold',
             color='#D62728', transform=ax1.transAxes)
    
    ax1.set_xlabel('L1 Score (Weight Magnitude)', fontsize=11)
    ax1.set_ylabel('GRA Score (Semantic Alignment)', fontsize=11)
    ax1.set_title(f'(a) Channel Selection Comparison\n{target_layer}', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图 2: 韦恩图式柱状图
    ax2 = fig.add_subplot(gs[1])
    
    categories = ['Both\nPrune', 'Only\nGRA', 'Only\nL1', 'Both\nKeep']
    counts = [len(both_prune), len(only_gra), len(only_l1), len(neither)]
    colors_bar = ['#888888', '#D62728', '#1F77B4', '#2CA02C']
    
    bars = ax2.bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Channels', fontsize=11)
    ax2.set_title('(b) Channel Selection Overlap', fontweight='bold')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_ylim(0, max(counts) * 1.15)
    
    # 子图 3: 关键发现
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    text = f"""
Key Findings:

• {len(only_l1)} channels are
  "Semantic Filters"
  (L1 says weak, 
   GRA says important)

• {len(only_gra)} channels are
  "Redundant"
  (L1 says strong, 
   GRA says useless)

• Agreement: {len(both_prune)+len(neither)}/{n_channels}
  ({100*(len(both_prune)+len(neither))/n_channels:.0f}%)
"""
    ax3.text(0, 0.95, text, fontsize=10, verticalalignment='top', 
             fontfamily='monospace', transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_gra_vs_l1_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_gra_vs_l1_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return len(only_l1), len(only_gra), target_layer


def main():
    print("=" * 60)
    print("科学价值可视化: GRA vs L1 深度对比")
    print("=" * 60)
    
    # 加载模型
    model = resnet56(num_classes=10)
    ckpt_path = r'C:\GRA-CNN\experiments\baseline_cifar10_resnet56.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"✓ 模型: {ckpt_path}")
    
    # 加载数据
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 计算分数
    print("\n计算 GRA 和 L1 分数...")
    gra_scores, l1_scores, conv_layers = compute_gra_and_l1_scores(model, dataloader)
    print(f"✓ 分析了 {len(conv_layers)} 个卷积层")
    
    # 生成对比图
    print("\n生成对比分析图...")
    n_semantic, n_redundant, layer = create_comparison_figure(gra_scores, l1_scores, conv_layers)
    
    print(f"\n关键发现 ({layer}):")
    print(f"  • 语义滤波器: {n_semantic} 个 (L1 低分但 GRA 高分)")
    print(f"  • 冗余通道: {n_redundant} 个 (L1 高分但 GRA 低分)")
    
    print("\n✓ 保存: fig_gra_vs_l1_analysis.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
