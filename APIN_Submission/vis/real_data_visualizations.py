"""
真实数据可视化 - 从实际模型提取 GRA 分数和特征图
==================================================
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
import os

sys.path.insert(0, r'C:\GRA-CNN')

from models.resnet_cifar import resnet56

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'

# ============================================================================
# 1. 真实 GRA 分数计算
# ============================================================================

def compute_real_gra_scores(model, dataloader, num_batches=5, rho=0.5):
    """
    计算真实的 GRA 通道重要性分数
    """
    model.eval()
    model.to(DEVICE)
    
    # 收集所有卷积层
    conv_layers = []
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 3:
            conv_layers.append(name)
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # 收集激活和 logits
    all_activations = {name: [] for name in conv_layers}
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            # 收集真实类别的 logit
            batch_logits = outputs[torch.arange(len(labels)), labels]
            all_logits.append(batch_logits.cpu())
            all_labels.append(labels.cpu())
            
            # 收集各层激活
            for name in conv_layers:
                # Global average pooling
                act = activations[name]
                act_pooled = act.mean(dim=(2, 3))  # [B, C]
                all_activations[name].append(act_pooled.cpu())
    
    # 移除 hooks
    for h in hooks:
        h.remove()
    
    # 合并数据
    all_logits = torch.cat(all_logits)  # [N]
    for name in conv_layers:
        all_activations[name] = torch.cat(all_activations[name], dim=0)  # [N, C]
    
    # 计算每层每通道的 GRA 分数
    gra_scores = {}
    for name in conv_layers:
        acts = all_activations[name].numpy()  # [N, C]
        logits = all_logits.numpy()  # [N]
        
        n_samples, n_channels = acts.shape
        scores = np.zeros(n_channels)
        
        # 归一化
        logits_norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
        
        for c in range(n_channels):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            
            # GRA 公式
            delta = np.abs(act_norm - logits_norm)
            delta_min, delta_max = delta.min(), delta.max()
            xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
            scores[c] = xi.mean()
        
        gra_scores[name] = scores
    
    return gra_scores, conv_layers


# ============================================================================
# 2. 真实特征图提取
# ============================================================================

def extract_real_feature_maps(model, image, layer_name='layer2.0.conv1'):
    """从真实模型提取特征图"""
    model.eval()
    model.to(DEVICE)
    
    activations = {}
    
    def hook(module, input, output):
        activations['target'] = output.detach()
    
    # 找到目标层
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        # 使用第一个 conv 层
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 3:
                target_module = module
                layer_name = name
                break
    
    handle = target_module.register_forward_hook(hook)
    
    with torch.no_grad():
        image = image.to(DEVICE)
        _ = model(image)
    
    handle.remove()
    
    return activations['target'].cpu(), layer_name


# ============================================================================
# 3. 主可视化函数 - 真实数据版
# ============================================================================

def create_real_visualizations():
    """使用真实数据生成可视化"""
    
    print("=" * 60)
    print("真实数据可视化生成")
    print("=" * 60)
    
    # 加载模型
    model = resnet56(num_classes=10)
    ckpt_path = r'C:\GRA-CNN\experiments\baseline_cifar10_resnet56.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✓ 加载模型: {ckpt_path}")
    else:
        print("✗ 未找到模型，使用随机权重")
    
    # 加载数据
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # 计算真实 GRA 分数
    print("\n[1/3] 计算真实 GRA 分数...")
    gra_scores, conv_layers = compute_real_gra_scores(model, dataloader, num_batches=10)
    print(f"  ✓ 分析了 {len(conv_layers)} 个卷积层")
    
    # ========== 图1: 真实通道重要性热力图 ==========
    print("\n[2/3] 生成通道重要性热力图...")
    
    # 选择有代表性的层
    selected_layers = conv_layers[::3][:12]  # 每隔3层选一个
    max_channels = max(len(gra_scores[l]) for l in selected_layers)
    
    # 构建矩阵
    heatmap_data = np.full((len(selected_layers), max_channels), np.nan)
    for i, layer in enumerate(selected_layers):
        scores = gra_scores[layer]
        heatmap_data[i, :len(scores)] = scores
    
    fig, ax = plt.subplots(figsize=(14, 6))
    masked_data = np.ma.masked_invalid(heatmap_data)
    im = ax.imshow(masked_data, aspect='auto', cmap='RdYlBu_r', vmin=0.3, vmax=0.9)
    
    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Real GRA Channel Importance Scores (ResNet-56 on CIFAR-10)', fontweight='bold', fontsize=13)
    ax.set_yticks(range(len(selected_layers)))
    ax.set_yticklabels([l.split('.')[-1] if '.' in l else l for l in selected_layers], fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('GRA Score (γ)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_real_importance_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_real_importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 保存: fig_real_importance_heatmap.pdf")
    
    # ========== 图2: GRA 分数分布 (直方图) ==========
    print("\n[3/3] 生成 GRA 分数分布图...")
    
    # 合并所有分数
    all_scores = np.concatenate([gra_scores[l] for l in conv_layers])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 直方图
    ax = axes[0]
    ax.hist(all_scores, bins=50, color='#D62728', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.median(all_scores), color='blue', linestyle='--', linewidth=2, label=f'Median={np.median(all_scores):.3f}')
    ax.axvline(x=0.5, color='green', linestyle=':', linewidth=2, label='Threshold (0.5)')
    ax.set_xlabel('GRA Score (γ)', fontsize=12)
    ax.set_ylabel('Number of Channels', fontsize=12)
    ax.set_title('(a) Distribution of GRA Scores', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 逐层平均分数
    ax = axes[1]
    layer_means = [gra_scores[l].mean() for l in conv_layers]
    layer_stds = [gra_scores[l].std() for l in conv_layers]
    x = range(len(conv_layers))
    
    ax.bar(x, layer_means, yerr=layer_stds, color='#1F77B4', alpha=0.7, capsize=2)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean GRA Score', fontsize=12)
    ax.set_title('(b) Layer-wise Mean GRA Score', fontweight='bold')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f'L{i+1}' for i in x[::5]])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_real_gra_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\GRA-CNN\APIN_Submission\fig_real_gra_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 保存: fig_real_gra_distribution.pdf")
    
    # 保存原始数据
    import pandas as pd
    rows = []
    for layer in conv_layers:
        for c, score in enumerate(gra_scores[layer]):
            rows.append({'layer': layer, 'channel': c, 'gra_score': score})
    df = pd.DataFrame(rows)
    df.to_csv(r'C:\GRA-CNN\experiments\real_gra_scores.csv', index=False)
    print("\n  ✓ 原始数据: experiments/real_gra_scores.csv")
    
    print("\n" + "=" * 60)
    print("真实数据可视化完成!")
    print("=" * 60)


if __name__ == "__main__":
    create_real_visualizations()
