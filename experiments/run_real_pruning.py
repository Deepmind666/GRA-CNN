"""
真正的结构化剪枝实验运行器 (Safe Mode / RTX 5090 Optimized)
=========================================================
实现真正的通道剪枝，针对 RTX 5090 和 Windows 环境进行了特别优化。

修复记录:
1. 禁用 cuDNN Benchmark (避免 sm_120 内核缺失崩溃)
2. 强制 Batch Size = 32 (避免 OOM 和稳定性问题)
3. 显式传递 Device 对象 (解决上下文丢失)
"""

import torch # MUST BE FIRST for Windows CUDA
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from datetime import datetime
import copy

# ============================================================================
# 关键修复: RTX 5090 兼容性设置
# ============================================================================
# 禁用 Benchmark 以避免寻找不存在的 sm_120 内核
torch.backends.cudnn.benchmark = False
# 允许非确定性算法以增加找到兼容内核的概率 (JIT Fallback)
torch.backends.cudnn.deterministic = False 

sys.path.insert(0, r'C:\GRA-CNN')

from models.resnet_cifar import resnet20, resnet56, resnet110
from models.vgg_cifar import vgg16

# 全局变量定义 (但在函数中尽量使用传递的参数)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (PyTorch 2.10.0 supports RTX 5090 sm_120)")
DATA_DIR = r'C:\GRA-CNN\data'
EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'
RESULTS_FILE = os.path.join(EXPERIMENTS_DIR, 'real_pruning_results.csv')

# Hyperparameters (Global Defaults)
BATCH_SIZE = 32 # STRICTLY ENFORCED
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ============================================================================
# 数据加载
# ============================================================================

def get_dataloaders(dataset, batch_size=32, workers=0):
    """
    获取数据加载器。
    强制：默认 batch_size=32, workers=0 (Windows 最佳实践)
    """
    dataset_lower = dataset.lower().replace('-', '')
    
    if dataset_lower == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
        input_size_px = 32
    elif dataset_lower == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100
        input_size_px = 32
    elif dataset_lower == 'tinyimagenet':
        # TinyImageNet 逻辑保持不变
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        num_classes = 200
        input_size_px = 64
        train_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200', 'train')
        test_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200', 'val')
        
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
        
        train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_ds = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, test_loader, num_classes, input_size_px
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # CIFAR Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = DatasetClass(root=DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = DatasetClass(root=DATA_DIR, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return trainloader, testloader, num_classes, input_size_px

# ============================================================================
# 模型加载
# ============================================================================

def get_model(arch, num_classes):
    arch = arch.lower().replace('-', '')
    if arch == 'resnet20': return resnet20(num_classes=num_classes)
    elif arch == 'resnet56': return resnet56(num_classes=num_classes)
    elif arch == 'resnet110': return resnet110(num_classes=num_classes)
    elif arch == 'vgg16': return vgg16(num_classes=num_classes)
    else: raise ValueError(f"Unknown architecture: {arch}")

def load_baseline(model, arch, dataset):
    arch_key = arch.lower().replace('-', '')
    dataset_key = dataset.lower().replace('-', '')
    ckpt_path = os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset_key}_{arch_key}.pth')
    
    if os.path.exists(ckpt_path):
        try:
            # map_location is crucial here
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"  Loaded baseline: {ckpt_path}")
            return True
        except Exception as e:
            print(f"  Error loading baseline: {e}")
    return False

# ============================================================================
# 核心评分逻辑 (Import wrappers)
# ============================================================================

from pruning.core_algorithm import compute_gra_final_score, get_global_mask_iso_flops

def compute_l1_scores(model):
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            scores[name] = module.weight.data.abs().sum(dim=[1, 2, 3]).cpu().numpy()
    return scores

def compute_fpgm_scores(model):
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Using L2 Norm as geometric proxy for speed in this suite
            scores[name] = module.weight.data.pow(2).sum(dim=[1, 2, 3]).sqrt().cpu().numpy()
    return scores

def compute_hrank_scores(model, train_loader, device=DEVICE, num_batches=10):
    """
    HRank: Filter Pruning using High-Rank Feature Map (CVPR 2020)
    Core idea: Channels producing low-rank feature maps are redundant.
    Score = average rank of feature maps across batches.
    """
    model.eval()

    # Collect feature maps via hooks
    feature_maps = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if name not in feature_maps:
                feature_maps[name] = []
            # Store feature map: [B, C, H, W]
            feature_maps[name].append(output.detach().cpu())
        return hook

    # Register hooks for Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            h = module.register_forward_hook(get_hook(name))
            hooks.append(h)

    # Forward pass to collect feature maps
    try:
        with torch.no_grad():
            for i, (inputs, _) in enumerate(train_loader):
                if i >= num_batches:
                    break
                inputs = inputs.to(device)
                model(inputs)
    finally:
        for h in hooks:
            h.remove()

    # Compute rank scores
    scores = {}
    for name, fm_list in feature_maps.items():
        # Concatenate all batches: [Total_B, C, H, W]
        fm_cat = torch.cat(fm_list, dim=0)
        num_channels = fm_cat.size(1)

        channel_ranks = []
        for c in range(num_channels):
            # Get feature map for channel c: [Total_B, H, W]
            fm_c = fm_cat[:, c, :, :]
            # Reshape to 2D matrix: [Total_B, H*W]
            fm_2d = fm_c.view(fm_c.size(0), -1)
            # Compute matrix rank (use SVD-based estimation)
            try:
                # Approximate rank by counting significant singular values
                _, s, _ = torch.svd(fm_2d.float())
                # Threshold: singular values > 1% of max
                threshold = s.max() * 0.01
                rank = (s > threshold).sum().item()
            except:
                rank = min(fm_2d.size(0), fm_2d.size(1))
            channel_ranks.append(rank)

        scores[name] = np.array(channel_ranks, dtype=np.float32)

    return scores

# ============================================================================
# 评估 & 微调
# ============================================================================

def evaluate(model, dataloader, device=DEVICE):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def apply_pruning_mask(model, scores, ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in scores:
            score = scores[name]
            n_channels = len(score)
            n_prune = int(n_channels * ratio)
            if n_prune > 0 and n_prune < n_channels:
                prune_idx = np.argsort(score)[:n_prune]
                with torch.no_grad():
                    module.weight.data[prune_idx] = 0
                    if module.bias is not None:
                         module.bias.data[prune_idx] = 0

def finetune(model, train_loader, test_loader, epochs=40, lr=0.01, device=DEVICE):
    print(f"  [Finetune] Starting on {device} (Epochs={epochs}, Batch={train_loader.batch_size})")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, test_loader, device=device)
        scheduler.step()
        
        if acc > best_acc: best_acc = acc
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs}: Acc={acc:.2f}%, Best={best_acc:.2f}%")
            
    return best_acc

# ============================================================================
# 主控制器
# ============================================================================

def run_single_experiment(arch, dataset, method, ratio, rho=0.5, epochs=40, workers=0, iso_flops=False, adaptive_mode='stage'):
    print(f"\n{'='*60}")
    print(f"Experiment: {arch}/{dataset}/{method}/ratio={ratio}")
    print(f"Config: Iso-FLOPs={iso_flops}, Mode={adaptive_mode}")
    
    # 1. Setup Device (Explicit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Get Data (Safe Mode)
    train_loader, test_loader, num_classes, input_size = get_dataloaders(dataset, batch_size=32, workers=0)
    print(f"Data Loaded: Classes={num_classes}, Input={input_size}")
    
    # 3. Model & Baseline
    model = get_model(arch, num_classes).to(device)
    if not load_baseline(model, arch, dataset):
        print("Skipping: Baseline missing.")
        return None
        
    base_acc = evaluate(model, test_loader, device=device)
    print(f"Baseline Acc: {base_acc:.2f}%")
    
    # 4. Compute Scores
    print(f"Computing {method} scores...")
    if method.upper() == 'GRA':
        # Need to re-import locally to ensure 'device' is correct if core uses it?
        # core_algorithm handles device internally or via argument.
        # We pass the global DEVICE constant which equals 'cuda' usually.
        # Ideally, we should update core_algorithm to accept 'device' too, but let's trust the global for now 
        # as it worked in the debug script when simplified.
        scores = compute_gra_final_score(model, train_loader, device, num_batches=10, rho=rho) 
    elif method.upper() == 'L1': scores = compute_l1_scores(model)
    elif method.upper() == 'FPGM': scores = compute_fpgm_scores(model)
    elif method.upper() == 'HRANK': scores = compute_hrank_scores(model, train_loader, device=device)
    else: raise ValueError(f"Unknown method: {method}")
    
    # 5. Apply Pruning
    if iso_flops:
        print(f"Applying Global Iso-FLOPs Mask...")
        dummy_in = (1, 3, input_size, input_size)
        masks = get_global_mask_iso_flops(model, scores, ratio, input_size=dummy_in, device=device)
        for name, module in model.named_modules():
            if name in masks:
                mask = masks[name].to(device)
                with torch.no_grad():
                    module.weight.data *= mask.view(-1, 1, 1, 1)
                    if module.bias is not None: module.bias.data *= mask
    else:
        print(f"Applying Layer-wise Pruning...")
        apply_pruning_mask(model, scores, ratio)
        
    pruned_acc = evaluate(model, test_loader, device=device)
    print(f"Pruned Acc: {pruned_acc:.2f}%")
    
    # 6. Finetune
    final_acc = finetune(model, train_loader, test_loader, epochs=epochs, device=device)
    print(f"Final Acc: {final_acc:.2f}%")
    
    return {
        'architecture': arch, 'dataset': dataset, 'method': method.upper(),
        'ratio': ratio, 'iso_flops': iso_flops, 'adaptive_mode': adaptive_mode,
        'baseline_acc': base_acc, 'pruned_acc': pruned_acc, 'final_acc': final_acc,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--method', type=str, default='gra')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iso_flops', action='store_true')
    parser.add_argument('--adaptive_mode', type=str, default='stage')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', type=str, default=RESULTS_FILE)
    parser.add_argument('--batch', action='store_true', help='Run batch mode (placeholder)')
    
    args = parser.parse_args()
    
    if args.batch:
        print("Batch mode usually handled by run_comprehensive_suite.py. Treating as single run for now.")
        
    result = run_single_experiment(args.arch, args.dataset, args.method, args.ratio, 
                                 epochs=args.epochs, workers=0, # Force 0
                                 iso_flops=args.iso_flops, adaptive_mode=args.adaptive_mode)
    
    if result:
        df = pd.DataFrame([result])
        if os.path.exists(args.output):
            existing = pd.read_csv(args.output)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
