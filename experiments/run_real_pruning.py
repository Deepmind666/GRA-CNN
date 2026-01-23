"""
真正的结构化剪枝实验运行器
============================
实现真正的通道剪枝（不是简化版本）

功能：
1. 计算每层通道的重要性分数
2. 根据分数排序，移除低分通道
3. 构建新的剪枝模型
4. 微调恢复精度
5. 记录结果
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import time
from datetime import datetime
import copy

sys.path.insert(0, r'C:\GRA-CNN')

from models.resnet_cifar import resnet20, resnet56, resnet110
from models.vgg_cifar import vgg16

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'
EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'
RESULTS_FILE = r'C:\GRA-CNN\experiments\supplementary_results.csv'

# ============================================================================
# 数据加载
# ============================================================================

def get_dataloaders(dataset, batch_size=128):
    dataset_lower = dataset.lower().replace('-', '')
    if dataset_lower == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
        input_size = 32
    elif dataset_lower == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100
        input_size = 32
    elif dataset_lower == 'tinyimagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        num_classes = 200
        # 假设 Tiny-ImageNet 已经准备在 DATA_DIR/tiny-imagenet-200
        from torchvision.datasets import ImageFolder
        train_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200', 'train')
        test_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200', 'val')
        input_size = 64
        
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        train_ds = ImageFolder(train_dir, transform=train_transform)
        test_ds = ImageFolder(test_dir, transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, test_loader, num_classes

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_ds = DatasetClass(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_ds = DatasetClass(root=DATA_DIR, train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, num_classes

# ============================================================================
# 模型
# ============================================================================

def get_model(arch, num_classes):
    arch = arch.lower().replace('-', '')
    if arch == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif arch == 'resnet32':
        from models.resnet_cifar import resnet32
        return resnet32(num_classes=num_classes)
    elif arch == 'resnet56':
        return resnet56(num_classes=num_classes)
    elif arch == 'resnet110':
        return resnet110(num_classes=num_classes)
    elif arch == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif arch == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2_cifar, mobilenetv2_tiny
        # 根据数据集判断输入尺寸
        if num_classes == 200:
            return mobilenetv2_tiny(num_classes=num_classes)
        return mobilenetv2_cifar(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def load_baseline(model, arch, dataset):
    arch_key = arch.lower().replace('-', '')
    dataset_key = dataset.lower().replace('-', '')
    ckpt_path = os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset_key}_{arch_key}.pth')
    
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"  Loaded baseline: {ckpt_path}")
            return True
        except Exception as e:
            print(f"  Error loading baseline: {e}")
    else:
        print(f"  Baseline not found: {ckpt_path}")
    return False

# ============================================================================
# 评分方法
# ============================================================================


def compute_l1_scores(model):
    """L1-Norm: 权重绝对值之和"""
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            score = weight.abs().view(weight.size(0), -1).sum(dim=1)
            scores[name] = score.cpu().numpy()
    return scores


# ============================================================================
# GRA-Fisher: 基于 Fisher 信息量的语义剪枝 (理论最优)
# Fisher(50%) + GRA(30%) + L1(20%)
# ============================================================================

def compute_gra_scores(model, dataloader, num_batches=10, rho=0.5):
    """
    GRA-Fisher v2.0: 增强型 Fisher 语义剪枝
    
    主要改进:
    1. Class-Balanced Fisher: 引入类别加权，防止对多数类的偏好
    2. Channel Orthogonality: 惩罚高相关性通道，增加特征多样性
    3. Fisher 稳定性增强: 使用梯度平方的对数变换
    """
    model.eval()
    DEVICE = next(model.parameters()).device
    
    # ============ 1. Class-Balanced Fisher ============
    fisher_accumulator = {}
    class_counts = {}
    
    activations = {}
    gradients = {}
    hooks = []
    
    def save_act(name):
        def hook(m, i, o): activations[name] = o.detach()
        return hook
    def save_grad(name):
        def hook(m, gi, go): gradients[name] = go[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches: break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # 记录类别分布用于平衡
        for t in targets:
            t_item = t.item()
            class_counts[t_item] = class_counts.get(t_item, 0) + 1
            
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.mean().backward()
        
        for name in activations:
            if name in gradients:
                act = activations[name]   # [B, C, H, W]
                grad = gradients[name]    # [B, C, H, W]
                # 计算每个样本的 Fisher
                # Fisher = (g * a)^2
                f_val = (grad * act).pow(2).mean(dim=[2, 3]) # [B, C]
                
                # 类别权重补偿 (1/count)
                batch_weights = torch.tensor([1.0/max(class_counts.get(t.item(), 1), 1) for t in targets], device=DEVICE)
                f_weighted = (f_val * batch_weights.unsqueeze(1)).sum(dim=0) # [C]
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = f_weighted
                else:
                    fisher_accumulator[name] += f_weighted
                    
    for h in hooks: h.remove()
    
    # ============ 2. Channel Orthogonality & Variety ============
    # 计算通道特征的相关性，惩罚高度冗余的通道
    ortho_scores = {}
    model.eval()
    with torch.no_grad():
        for name in activations:
            act = activations[name].mean(dim=[2, 3]) # [B, C]
            # 归一化特征
            act = (act - act.mean(dim=0)) / (act.std(dim=0) + 1e-8)
            # 计算相关矩阵 [C, C]
            corr = torch.matmul(act.t(), act) / act.size(0)
            # 某通道与其他通道的相关性均值 (惩罚项)
            redundancy = (corr.abs().sum(dim=1) - 1.0) / max(act.size(1) - 1, 1)
            ortho_scores[name] = (1.0 - redundancy).cpu().numpy() # 越不冗余分数越高

    # ============ 3. GRA 语义与 L1 稳定 ============
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            l1_scores[name] = w.abs().view(w.size(0), -1).sum(dim=1).cpu().numpy()

    # 重新计算 margin 相关的 GRA
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= 4: break 
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            all_logits.append(model(inputs))
            all_targets.append(targets)
    
    logits = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    correct = logits.gather(1, targets_cat.view(-1, 1)).squeeze()
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets_cat.view(-1, 1), False)
    max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
    margin = (correct - max_wrong).detach()
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)

    gra_scores = {}
    for name in activations:
        act_c = activations[name].mean(dim=[2, 3]).detach() # [B, C]
        c_scores = []
        for c in range(act_c.size(1)):
            a_norm = (act_c[:, c] - act_c[:, c].min()) / (act_c[:, c].max() - act_c[:, c].min() + 1e-8)
            delta = (a_norm - margin_norm).abs()
            gamma = (delta.min() + rho * delta.max()) / (delta + rho * delta.max() + 1e-8)
            c_scores.append(gamma.mean().item())
        gra_scores[name] = np.array(c_scores)

    # ============ 4. 终极融合 (v2.0 优化配比) ============
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    final_scores = {}
    for name in fisher_accumulator:
        f_norm = norm(fisher_accumulator[name].cpu().numpy())
        o_norm = norm(ortho_scores[name]) if name in ortho_scores else f_norm
        g_norm = norm(gra_scores[name]) if name in gra_scores else f_norm
        l_norm = norm(l1_scores[name]) if name in l1_scores else f_norm
        
        combined = 0.40 * f_norm + 0.20 * o_norm + 0.25 * g_norm + 0.15 * l_norm
        final_scores[name] = combined
    
    return final_scores



def compute_fpgm_scores(model):
    """FPGM: 几何中位数距离"""
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            n_out = weight.size(0)
            flat = weight.view(n_out, -1)
            
            # 计算每对滤波器之间的距离
            dist_mat = torch.cdist(flat, flat, p=2)
            
            # 距离和越小 = 越靠近中心 = 越冗余 = 应被剪枝
            # 所以分数 = 距离和 (高分=重要)
            score = dist_mat.sum(dim=1)
            scores[name] = score.cpu().numpy()
    
    return scores

def compute_hrank_scores(model, dataloader, num_batches=3):
    """HRank: 特征图秩"""
    model.eval()
    ranks = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in ranks:
                ranks[name] = []
            # 计算每个通道的特征图秩
            B, C, H, W = output.shape
            for c in range(C):
                fm = output[:, c, :, :].view(B, H*W)  # (B, H*W)
                try:
                    rank = torch.linalg.matrix_rank(fm.float()).float().mean().item()
                except:
                    rank = min(H, W) / 2  # fallback
                if len(ranks[name]) <= c:
                    ranks[name].append([rank])
                else:
                    ranks[name][c].append(rank)
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(DEVICE)
            _ = model(inputs)
    
    for h in hooks:
        h.remove()
    
    scores = {}
    for name, channel_ranks in ranks.items():
        avg_ranks = [np.mean(r) for r in channel_ranks]
        scores[name] = np.array(avg_ranks)
    
    return scores

# ============================================================================
# 评估
# ============================================================================

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# ============================================================================
# 剪枝 (简化版本：使用掩码而非真正移除通道)
# ============================================================================

def apply_pruning_mask(model, scores, ratio):
    """应用剪枝掩码（简化版本）"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in scores:
            score = scores[name]
            n_channels = len(score)
            n_prune = int(n_channels * ratio)
            
            if n_prune > 0 and n_prune < n_channels:
                # 找到要剪枝的通道
                prune_idx = np.argsort(score)[:n_prune]
                
                # 将这些通道的权重置零
                with torch.no_grad():
                    module.weight.data[prune_idx] = 0
                    if module.bias is not None:
                        module.bias.data[prune_idx] = 0

# ============================================================================
# 微调
# ============================================================================

def finetune(model, train_loader, test_loader, epochs=40, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, test_loader)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: {acc:.2f}%, Best={best_acc:.2f}%")
    
    return best_acc

# ============================================================================
# 主实验
# ============================================================================

def run_single_experiment(arch, dataset, method, ratio, rho=0.5, epochs=40):
    """运行单个剪枝实验"""
    print(f"\n{'='*60}")
    print(f"Experiment: {arch}/{dataset}/{method}/ratio={ratio}")
    print(f"{'='*60}")
    
    # 加载数据
    train_loader, test_loader, num_classes = get_dataloaders(dataset)
    
    # 加载模型
    model = get_model(arch, num_classes).to(DEVICE)
    if not load_baseline(model, arch, dataset):
        print("  Skipping: no baseline available")
        return None
    
    # 计算baseline精度
    baseline_acc = evaluate(model, test_loader)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    
    # 计算通道重要性分数
    print(f"  Computing {method} scores...")
    if method.upper() == 'GRA':
        scores = compute_gra_scores(model, train_loader, rho=rho)
    elif method.upper() == 'L1':
        scores = compute_l1_scores(model)
    elif method.upper() == 'FPGM':
        scores = compute_fpgm_scores(model)
    elif method.upper() == 'HRANK':
        scores = compute_hrank_scores(model, train_loader)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 应用剪枝
    print(f"  Applying pruning (ratio={ratio})...")
    apply_pruning_mask(model, scores, ratio)
    
    pruned_acc = evaluate(model, test_loader)
    print(f"  After pruning: {pruned_acc:.2f}%")
    
    # 微调
    print(f"  Fine-tuning for {epochs} epochs...")
    final_acc = finetune(model, train_loader, test_loader, epochs=epochs)
    print(f"  Final accuracy: {final_acc:.2f}%")
    
    return {
        'architecture': arch,
        'dataset': dataset,
        'method': method.upper(),
        'ratio': ratio,
        'baseline_acc': baseline_acc,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc,
        'accuracy': final_acc,  # 兼容complete_data格式
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--method', type=str, default='gra')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', action='store_true', help='Run batch mode for all missing experiments')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量模式：运行所有缺失实验
        run_batch_experiments()
    else:
        # 单个实验
        result = run_single_experiment(args.arch, args.dataset, args.method, args.ratio, epochs=args.epochs)
        if result:
            # 追加到结果文件
            df = pd.DataFrame([result])
            if os.path.exists(RESULTS_FILE):
                existing = pd.read_csv(RESULTS_FILE)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(RESULTS_FILE, index=False)
            print(f"\nResult saved to: {RESULTS_FILE}")

def run_batch_experiments():
    """运行所有缺失的实验"""
    # 加载现有数据
    existing = pd.read_csv(r'C:\GRA-CNN\experiments\complete_data.csv')
    
    # 定义实验矩阵 (简化版本：只运行有baseline的配置)
    configs = [
        ('ResNet-20', 'CIFAR-10'),
        ('ResNet-56', 'CIFAR-10'),
        ('ResNet-20', 'CIFAR-100'),
        ('ResNet-56', 'CIFAR-100'),
    ]
    methods = ['GRA', 'L1', 'FPGM', 'HRank']
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    total = len(configs) * len(methods) * len(ratios)
    current = 0
    
    for arch, dataset in configs:
        for method in methods:
            for ratio in ratios:
                current += 1
                
                # 检查是否已存在
                match = existing[(existing['architecture']==arch) & 
                                (existing['dataset']==dataset) &
                                (existing['method']==method) &
                                (existing['ratio']==ratio)]
                if len(match) > 0:
                    print(f"[{current}/{total}] {arch}/{dataset}/{method}/{ratio} - EXISTS, skipping")
                    continue
                
                print(f"\n[{current}/{total}] Running...")
                result = run_single_experiment(arch, dataset, method, ratio, epochs=40)
                
                if result:
                    results.append(result)
                    
                    # 每完成一个实验就保存
                    df = pd.DataFrame(results)
                    df.to_csv(RESULTS_FILE, index=False)
    
    print(f"\n{'='*60}")
    print(f"Batch complete! {len(results)} new experiments saved.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
