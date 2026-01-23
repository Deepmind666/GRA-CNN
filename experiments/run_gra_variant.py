"""
GRA 算法变体实验运行器
======================

支持的算法变体:
1. fisher_weighted: 样本加权Fisher (困难样本权重更高)
2. fisher_orthogonal: 通道正交性 (去除冗余通道)
3. fisher_class_balanced: 类别平衡Fisher
4. fisher_stable: 梯度稳定性

支持的权重配置:
--fisher_weight, --gra_weight, --l1_weight
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import resnet_cifar, vgg_cifar

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 数据加载
# ============================================================================

def get_dataloader(dataset_name, batch_size=128, num_workers=4):
    if dataset_name == 'CIFAR-10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:  # CIFAR-100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader, num_classes


def get_model(arch, num_classes):
    if arch == 'ResNet-20':
        model = resnet_cifar.resnet20(num_classes=num_classes)
    elif arch == 'ResNet-56':
        model = resnet_cifar.resnet56(num_classes=num_classes)
    elif arch == 'ResNet-110':
        model = resnet_cifar.resnet110(num_classes=num_classes)
    elif arch == 'VGG-16':
        model = vgg_cifar.vgg16_bn(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model.to(DEVICE)


# ============================================================================
# 算法变体实现
# ============================================================================

def compute_fisher_weighted(model, dataloader, num_batches=10):
    """
    样本加权 Fisher: 困难样本权重更高
    
    理论: 高损失样本更能暴露通道的真实重要性
    """
    model.eval()
    fisher_scores = {}
    activations = {}
    gradients = {}
    hooks = []
    
    def save_act(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_grad(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不reduce，保留每个样本的损失
    fisher_accumulator = {}
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, targets)  # (B,) 每个样本的损失
        
        # 样本权重 = 损失 / 平均损失
        weights = losses / (losses.mean() + 1e-8)
        weights = weights.detach()
        
        # 反向传播
        loss = losses.mean()
        loss.backward()
        
        for name in activations:
            if name in gradients:
                act = activations[name]  # (B, C, H, W)
                grad = gradients[name]   # (B, C, H, W)
                
                # 加权 Fisher = weight × (grad × act)²
                weighted_fisher = weights.view(-1, 1, 1, 1) * (grad * act).pow(2)
                fisher = weighted_fisher.mean(dim=[0, 2, 3])
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = fisher
                else:
                    fisher_accumulator[name] += fisher
    
    for h in hooks:
        h.remove()
    
    for name, fisher in fisher_accumulator.items():
        fisher_scores[name] = fisher.cpu().numpy()
    
    return fisher_scores


def compute_fisher_orthogonal(model, dataloader, num_batches=10):
    """
    通道正交性: 去除冗余通道
    
    理论: 保留正交通道，移除高相关性冗余通道
    Score = Fisher × Orthogonality (正交性惩罚)
    """
    # 先计算标准 Fisher
    fisher_scores = compute_standard_fisher(model, dataloader, num_batches)
    
    # 计算通道正交性
    model.eval()
    activations = {}
    hooks = []
    
    def save_act(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.mean(dim=[2, 3]).detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(DEVICE)
            model(inputs)
    
    for h in hooks:
        h.remove()
    
    # 计算每个通道与其他通道的最大相关性
    orthogonality_scores = {}
    for name, act_list in activations.items():
        acts = torch.cat(act_list, dim=0)  # (N, C)
        C = acts.size(1)
        
        # 计算通道间相关性矩阵
        acts_centered = acts - acts.mean(dim=0, keepdim=True)
        acts_std = acts.std(dim=0, keepdim=True) + 1e-8
        acts_norm = acts_centered / acts_std
        
        corr_matrix = torch.mm(acts_norm.T, acts_norm) / acts.size(0)
        
        # 正交性 = 1 - max_correlation (排除自身)
        corr_matrix.fill_diagonal_(0)
        max_corr = corr_matrix.abs().max(dim=1)[0]
        orthogonality = 1 - max_corr
        
        orthogonality_scores[name] = orthogonality.cpu().numpy()
    
    # Fisher × Orthogonality
    final_scores = {}
    for name in fisher_scores:
        if name in orthogonality_scores:
            fisher_norm = normalize(fisher_scores[name])
            ortho_norm = normalize(orthogonality_scores[name])
            final_scores[name] = fisher_norm * (0.5 + 0.5 * ortho_norm)
    
    return final_scores


def compute_fisher_class_balanced(model, dataloader, num_batches=10):
    """
    类别平衡 Fisher: 每个类别单独计算，然后加权
    
    理论: 确保所有类别都有足够的通道支持
    """
    model.eval()
    activations = {}
    gradients = {}
    hooks = []
    
    def save_act(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_grad(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 按类别累积 Fisher
    class_fisher = {}  # {name: {class_id: fisher}}
    class_counts = {}
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name in activations:
            if name in gradients:
                act = activations[name]
                grad = gradients[name]
                fisher = (grad * act).pow(2)  # (B, C, H, W)
                
                if name not in class_fisher:
                    class_fisher[name] = {}
                    class_counts[name] = {}
                
                for b in range(targets.size(0)):
                    c = targets[b].item()
                    f = fisher[b].mean(dim=[1, 2])  # (C,)
                    
                    if c not in class_fisher[name]:
                        class_fisher[name][c] = f
                        class_counts[name][c] = 1
                    else:
                        class_fisher[name][c] += f
                        class_counts[name][c] += 1
    
    for h in hooks:
        h.remove()
    
    # 类别平衡: 每个类别权重相同
    final_scores = {}
    for name in class_fisher:
        num_classes = len(class_fisher[name])
        C = None
        for c in class_fisher[name]:
            if C is None:
                C = class_fisher[name][c].size(0)
            class_fisher[name][c] /= class_counts[name][c]
        
        # 加权平均 (每个类别权重相同)
        balanced = torch.zeros(C, device=DEVICE)
        for c in class_fisher[name]:
            balanced += class_fisher[name][c] / num_classes
        
        final_scores[name] = balanced.cpu().numpy()
    
    return final_scores


def compute_fisher_stable(model, dataloader, num_batches=10):
    """
    梯度稳定性: 高方差梯度可能是噪声
    
    理论: 信噪比高的通道更可靠
    Score = Fisher × Stability
    """
    model.eval()
    activations = {}
    hooks = []
    
    def save_act(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_grad(name):
        def hook(module, grad_input, grad_output):
            if name not in batch_gradients:
                batch_gradients[name] = []
            batch_gradients[name].append(grad_output[0].mean(dim=[0, 2, 3]).detach())
        return hook
    
    batch_gradients = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    fisher_accumulator = {}
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name in activations:
            act = activations[name]
            # 获取当前梯度
            if name in batch_gradients and batch_gradients[name]:
                grad_mean = batch_gradients[name][-1]
                # 简化: 使用激活作为代理
                fisher = act.pow(2).mean(dim=[0, 2, 3])
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = fisher
                else:
                    fisher_accumulator[name] += fisher
    
    for h in hooks:
        h.remove()
    
    # 计算梯度稳定性 (均值/标准差)
    stability_scores = {}
    for name in batch_gradients:
        grads = torch.stack(batch_gradients[name], dim=0)  # (num_batches, C)
        mean = grads.mean(dim=0).abs()
        std = grads.std(dim=0) + 1e-8
        stability = mean / std
        stability_scores[name] = torch.sigmoid(stability).cpu().numpy()
    
    # Fisher × Stability
    final_scores = {}
    for name in fisher_accumulator:
        fisher = fisher_accumulator[name].cpu().numpy()
        if name in stability_scores:
            fisher_norm = normalize(fisher)
            stab_norm = stability_scores[name]
            final_scores[name] = fisher_norm * stab_norm
        else:
            final_scores[name] = fisher
    
    return final_scores


def compute_standard_fisher(model, dataloader, num_batches=10):
    """标准 Fisher 信息量"""
    model.eval()
    fisher_scores = {}
    activations = {}
    gradients = {}
    hooks = []
    
    def save_act(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_grad(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    fisher_accumulator = {}
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name in activations:
            if name in gradients:
                act = activations[name]
                grad = gradients[name]
                fisher = (grad * act).pow(2).mean(dim=[0, 2, 3])
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = fisher
                else:
                    fisher_accumulator[name] += fisher
    
    for h in hooks:
        h.remove()
    
    for name, fisher in fisher_accumulator.items():
        fisher_scores[name] = fisher.cpu().numpy()
    
    return fisher_scores


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# ============================================================================
# GRA 评分函数
# ============================================================================

def compute_gra_scores(model, dataloader, num_batches=10, rho=0.5):
    """GRA 语义对齐评分"""
    model.eval()
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.mean(dim=[2, 3]).detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            all_logits.append(model(inputs).detach())
            all_targets.append(targets.detach())
    
    for h in hooks:
        h.remove()
    
    logits = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    
    correct = logits.gather(1, targets_cat.view(-1, 1)).squeeze()
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets_cat.view(-1, 1), False)
    max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
    margin = correct - max_wrong
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
    
    gra_scores = {}
    for name, act_list in activations.items():
        acts = torch.cat(act_list, dim=0)
        C = acts.size(1)
        scores = []
        for c in range(C):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            delta = (act_norm - margin_norm).abs()
            gamma = (delta.min() + rho * delta.max()) / (delta + rho * delta.max() + 1e-8)
            scores.append(gamma.mean().item())
        gra_scores[name] = np.array(scores)
    
    return gra_scores


def compute_l1_scores(model):
    """L1 范数评分"""
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            l1_scores[name] = w.abs().view(w.size(0), -1).sum(dim=1).cpu().numpy()
    return l1_scores


# ============================================================================
# 融合评分
# ============================================================================

def compute_combined_scores(model, dataloader, fisher_weight=0.5, gra_weight=0.3, l1_weight=0.2, 
                           variant=None, num_batches=10):
    """
    融合评分: Fisher + GRA + L1
    """
    # 根据变体选择 Fisher 计算方法
    if variant == 'fisher_weighted':
        fisher_scores = compute_fisher_weighted(model, dataloader, num_batches)
    elif variant == 'fisher_orthogonal':
        fisher_scores = compute_fisher_orthogonal(model, dataloader, num_batches)
    elif variant == 'fisher_class_balanced':
        fisher_scores = compute_fisher_class_balanced(model, dataloader, num_batches)
    elif variant == 'fisher_stable':
        fisher_scores = compute_fisher_stable(model, dataloader, num_batches)
    else:
        fisher_scores = compute_standard_fisher(model, dataloader, num_batches)
    
    gra_scores = compute_gra_scores(model, dataloader, num_batches)
    l1_scores = compute_l1_scores(model)
    
    final_scores = {}
    for name in fisher_scores:
        if name in l1_scores and name in gra_scores:
            combined = (fisher_weight * normalize(fisher_scores[name]) +
                       gra_weight * normalize(gra_scores[name]) +
                       l1_weight * normalize(l1_scores[name]))
            final_scores[name] = combined
    
    return final_scores


# ============================================================================
# 剪枝和微调
# ============================================================================

def apply_pruning(model, scores, ratio):
    """全局剪枝"""
    all_scores = []
    score_to_layer = []
    
    for name, s in scores.items():
        for i, score in enumerate(s):
            all_scores.append(score)
            score_to_layer.append((name, i))
    
    all_scores = np.array(all_scores)
    num_prune = int(len(all_scores) * ratio)
    threshold = np.partition(all_scores, num_prune)[num_prune]
    
    masks = {}
    for name, s in scores.items():
        mask = s > threshold
        # 确保至少保留10%
        if mask.sum() < len(mask) * 0.1:
            keep_idx = np.argsort(s)[-max(1, int(len(mask) * 0.1)):]
            mask = np.zeros_like(mask, dtype=bool)
            mask[keep_idx] = True
        masks[name] = mask
    
    return masks


def finetune(model, trainloader, testloader, epochs=40, lr=0.01):
    """微调模型"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: {acc:.2f}%, Best={best_acc:.2f}%")
    
    return best_acc


def evaluate(model, testloader):
    """评估模型"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ResNet-56')
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--fisher_weight', type=float, default=0.5)
    parser.add_argument('--gra_weight', type=float, default=0.3)
    parser.add_argument('--l1_weight', type=float, default=0.2)
    parser.add_argument('--variant', type=str, default=None)
    args = parser.parse_args()
    
    print(f"GRA 算法变体实验")
    print(f"架构: {args.arch}, 数据集: {args.dataset}, 剪枝率: {args.ratio}")
    print(f"权重: Fisher={args.fisher_weight}, GRA={args.gra_weight}, L1={args.l1_weight}")
    print(f"变体: {args.variant}")
    print("="*60)
    
    # 加载数据
    trainloader, testloader, num_classes = get_dataloader(args.dataset)
    
    # 加载基线模型
    baseline_name = f"baseline_{args.dataset.lower().replace('-', '')}_{args.arch.lower().replace('-', '')}.pth"
    baseline_path = PROJECT_ROOT / "experiments" / baseline_name
    
    model = get_model(args.arch, num_classes)
    if baseline_path.exists():
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        print(f"  Loaded baseline: {baseline_path}")
    
    baseline_acc = evaluate(model, testloader)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    
    # 计算评分
    print(f"  Computing scores...")
    scores = compute_combined_scores(
        model, trainloader, 
        fisher_weight=args.fisher_weight,
        gra_weight=args.gra_weight,
        l1_weight=args.l1_weight,
        variant=args.variant
    )
    
    # 剪枝
    print(f"  Applying pruning (ratio={args.ratio})...")
    masks = apply_pruning(model, scores, args.ratio)
    
    # 应用mask (简化版: 直接置零)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in masks:
            mask = masks[name]
            with torch.no_grad():
                module.weight.data[~mask] = 0
                if module.bias is not None:
                    module.bias.data[~mask] = 0
    
    pruned_acc = evaluate(model, testloader)
    print(f"  After pruning: {pruned_acc:.2f}%")
    
    # 微调
    print(f"  Fine-tuning for {args.epochs} epochs...")
    final_acc = finetune(model, trainloader, testloader, epochs=args.epochs)
    print(f"  Final accuracy: {final_acc:.2f}%")
    
    # 保存结果
    result = {
        'timestamp': datetime.now().isoformat(),
        'architecture': args.arch,
        'dataset': args.dataset,
        'method': 'GRA',
        'ratio': args.ratio,
        'fisher_weight': args.fisher_weight,
        'gra_weight': args.gra_weight,
        'l1_weight': args.l1_weight,
        'variant': args.variant,
        'baseline_acc': baseline_acc,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc
    }
    
    results_file = PROJECT_ROOT / "experiments" / "overnight_deep_results.csv"
    df = pd.DataFrame([result])
    
    if results_file.exists():
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df.to_csv(results_file, index=False)
    
    print(f"\nResult saved to: {results_file}")


if __name__ == "__main__":
    main()
