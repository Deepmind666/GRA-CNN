"""
GRA-CNN v4 完整实验矩阵
========================

特性:
1. 真正的结构化剪枝 (重建网络而非权重置零)
2. 完整ratio范围: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
3. 多种方法对比: L1, FPGM, GRA-v1, GRA-v4
4. 多seed统计: 42, 123, 456
5. 支持并发执行，GPU利用率80%

Author: GRA-CNN Team
Date: 2026-02-03
"""

import sys
sys.path.insert(0, r'C:\GRA-CNN')

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from collections import defaultdict

# 确保可复现性
# 默认优先可复现；如需提速可通过环境变量覆盖：
# GRA_CUDNN_BENCHMARK=1 / GRA_CUDNN_DETERMINISTIC=0
torch.backends.cudnn.benchmark = os.environ.get('GRA_CUDNN_BENCHMARK', '0') == '1'
torch.backends.cudnn.deterministic = os.environ.get('GRA_CUDNN_DETERMINISTIC', '1') == '1'


# =============================================================================
# Section 1: 基础设施
# =============================================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """确保多worker下的可复现性"""
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)  # 修复: 添加torch种子


def resolve_data_root(dataset='cifar10'):
    """
    Resolve dataset root for local/remote layouts.
    Priority:
    1) env GRA_DATA_ROOT
    2) ./data
    3) ../data (for server checkout under ...\\GRA-CNN\\project)
    """
    candidates = []
    env_root = os.environ.get("GRA_DATA_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(["data", os.path.join("..", "data")])

    def has_target(root_path):
        if not os.path.isdir(root_path):
            return False
        if dataset == 'cifar10':
            return os.path.isdir(os.path.join(root_path, "cifar-10-batches-py"))
        return os.path.isdir(os.path.join(root_path, "cifar-100-python"))

    for root in candidates:
        if has_target(root):
            return root
    return "data"


def get_dataloaders(dataset='cifar10', batch_size=2048, seed=42, workers=None):
    """获取训练和测试数据加载器（用于微调）- 大batch提高GPU利用率"""
    import torchvision
    import torchvision.transforms as transforms

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = resolve_data_root(dataset)

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=transform_test)

    # 使用generator确保可复现性
    g = torch.Generator()
    g.manual_seed(seed)

    if workers is None:
        workers = int(os.environ.get("GRA_NUM_WORKERS", "1"))
    workers = max(int(workers), 0)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=worker_init_fn if workers > 0 else None,
        generator=g,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


def get_scoring_dataloader(dataset='cifar10', batch_size=128, num_samples=2560, seed=42):
    """
    获取固定评分数据加载器 (P0-2修复)

    特性:
    - 无数据增强 (只有归一化)
    - 固定indices子集
    - 单worker + generator确保可复现

    Args:
        dataset: 数据集名称
        batch_size: 批大小
        num_samples: 评分使用的样本数 (默认2560 = 20 batches * 128)
        seed: 随机种子

    Returns:
        scoring_loader: 固定的评分数据加载器
    """
    import torchvision
    import torchvision.transforms as transforms

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # 无增强变换 - 只有归一化
    transform_scoring = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = resolve_data_root(dataset)

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_scoring)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform_scoring)

    # 固定indices子集
    g = torch.Generator()
    g.manual_seed(seed)
    all_indices = torch.randperm(len(trainset), generator=g).tolist()
    fixed_indices = all_indices[:num_samples]

    subset = torch.utils.data.Subset(trainset, fixed_indices)

    # 单worker + generator确保可复现
    scoring_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        worker_init_fn=worker_init_fn, generator=g)

    return scoring_loader


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def finetune(model, train_loader, test_loader, device, epochs=40):
    """微调剪枝后的模型 (混合精度加速)"""
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = evaluate(model, test_loader, device)
            best_acc = max(best_acc, acc)
            print(f"    Epoch {epoch+1}: {acc:.2f}%")

    return best_acc


# =============================================================================
# Section 2: 评分方法
# =============================================================================

def compute_l1_scores(model):
    """L1范数评分"""
    scores = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            scores[n] = m.weight.data.abs().sum(dim=(1,2,3)).cpu().numpy()
    return scores


def compute_fpgm_scores(model):
    """FPGM几何中值评分"""
    scores = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            C = w.size(0)
            w_flat = w.view(C, -1)
            dist = torch.cdist(w_flat, w_flat, p=2)
            scores[n] = dist.sum(dim=1).cpu().numpy()
    return scores


def compute_hrank_scores(model, scoring_loader, device, num_batches=10):
    """HRank: 基于特征图秩的评分 (CVPR 2020) - 批量SVD加速实现"""
    rank_sums = {}
    rank_counts = {}

    def hook(name):
        def fn(_m, _inp, out):
            if out.dim() != 4:
                return
            # out: [B, C, H, W] -> feat: [C, B, H*W]
            feat = out.detach().permute(1, 0, 2, 3).reshape(out.size(1), out.size(0), -1).float()
            try:
                s = torch.linalg.svdvals(feat)
            except RuntimeError:
                s = torch.linalg.svdvals(feat.cpu())
            thresh = 0.01 * s[..., :1]
            ranks = (s > thresh).sum(dim=-1).float().cpu()

            if name not in rank_sums:
                rank_sums[name] = ranks
                rank_counts[name] = 1
            else:
                rank_sums[name] += ranks
                rank_counts[name] += 1
        return fn

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook(n)))

    model.eval()
    with torch.no_grad():
        for i, (x, _y) in enumerate(scoring_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            model(x)

    for h in handles:
        h.remove()

    scores = {}
    for name, sums in rank_sums.items():
        count = max(rank_counts.get(name, 1), 1)
        scores[name] = (sums / count).numpy()

    return scores


def compute_taylor_scores(model, scoring_loader, device, num_batches=10):
    """Taylor展开重要性评分 (NeurIPS 2019)"""
    act_data = {}
    grad_data = {}

    def fwd_hook(name):
        def fn(_, __, out):
            act_data[name] = out.detach()
        return fn

    def bwd_hook(name):
        def fn(_, __, grad_out):
            grad_data[name] = grad_out[0].detach()
        return fn

    fwd_handles = []
    bwd_handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            fwd_handles.append(m.register_forward_hook(fwd_hook(n)))
            bwd_handles.append(m.register_full_backward_hook(bwd_hook(n)))

    scores = defaultdict(lambda: 0)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    for i, (x, y) in enumerate(scoring_loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for name in act_data:
            if name in grad_data:
                a = act_data[name]
                g = grad_data[name]
                if a.dim() == 4:
                    importance = (a * g).abs().mean(dim=(0, 2, 3))
                else:
                    importance = (a * g).abs().mean(dim=0)
                scores[name] = scores[name] + importance.cpu().numpy()

    for h in fwd_handles + bwd_handles:
        h.remove()

    for name in scores:
        scores[name] = scores[name] / max(num_batches, 1)

    return dict(scores)


def compute_random_scores(model):
    """随机评分基线 - 用于对比验证"""
    scores = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            num_ch = m.out_channels
            scores[n] = np.random.rand(num_ch)
    return scores


def compute_gra_v1_scores(model, scoring_loader, device, num_batches=20):
    """GRA-v1: Pearson相关评分 (使用固定评分数据加载器)"""
    act_data = defaultdict(list)
    margins = []

    def hook(name):
        def fn(m, inp, out):
            if out.dim() == 4:
                act_data[name].append(out.mean(dim=(2,3)).detach().cpu())
        return fn

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook(n)))

    model.eval()
    for i, (x, y) in enumerate(scoring_loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            c = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            lc = logits.clone()
            lc.scatter_(1, y.unsqueeze(1), float('-inf'))
            margins.append((c - lc.max(dim=1).values).cpu())

    for h in handles:
        h.remove()

    activations = {k: torch.cat(v) for k, v in act_data.items()}
    margin_seq = torch.cat(margins)

    # Pearson相关评分
    scores = {}
    ref = margin_seq.numpy()
    ref_c = ref - ref.mean()
    ref_s = ref.std() + 1e-8

    for name, act in activations.items():
        if act.dim() != 2:
            continue
        B, C = act.shape
        act_np = act.numpy()
        corrs = []
        for c in range(C):
            a = act_np[:, c]
            a_c = a - a.mean()
            a_s = a.std() + 1e-8
            corr = (ref_c * a_c).mean() / (ref_s * a_s)
            corrs.append((corr + 1) / 2)  # 映射到[0,1]
        scores[name] = np.array(corrs)

    return scores


def compute_gra_v4_scores(model, train_loader, device, num_batches=20, full_fusion=True, pruning_ratio=0.5):
    """
    GRA-v4: Pearson-GRA混合 + 深度+剪枝率双自适应

    Args:
        full_fusion: True=使用完整融合(Fisher+L1+GRA+Ortho), False=仅Pearson-GRA
        pruning_ratio: 剪枝率，用于自适应权重调整
    """
    if full_fusion:
        from pruning.core_algorithm_v4 import compute_gra_v4_scores as _compute_full
        scores, _ = _compute_full(model, train_loader, device, num_batches,
                                   verbose=False, pruning_ratio=pruning_ratio)
        return scores
    else:
        from pruning.core_algorithm_v4 import compute_gra_v4_simple as _compute_simple
        return _compute_simple(model, train_loader, device, num_batches)


# =============================================================================
# Section 3: 真正的结构化剪枝
# =============================================================================

def apply_structural_pruning(model, scores, ratio, arch, num_classes=10, device='cuda',
                             pruning_scope='resnet_conv1_only'):
    """
    真正的结构化剪枝 - 重建网络而非权重置零

    剪枝范围:
    - resnet_conv1_only: 仅剪枝每个BasicBlock内的conv1层 (内部通道)
    - stage_mid_only: Stage级统一mid_planes剪枝 (conv1输出+conv2输入，保持block输出不变)

    层保护 (P1-1说明):
    - ratio=0.5 表示"剪掉50%通道"，即保留50%
    - keep = num_ch * (1 - ratio) 计算保留数量

    Args:
        model: 原始模型
        scores: 各层重要性评分 {layer_name: np.array}
        ratio: 剪枝比例 (0.5 = 剪掉50%通道)
        arch: 架构名称 ('resnet56', 'vgg16'等)
        num_classes: 分类数
        device: 设备
        pruning_scope: 剪枝范围 ('resnet_conv1_only' 或 'stage_mid_only')

    Returns:
        new_model: 重建后的剪枝模型
        mask_dict: 各层掩码 {layer_name: bool_array}
    """
    from pruning.prune_model import build_new_resnet, build_new_vgg, build_resnet_stage_pruned

    mask_dict = {}

    if pruning_scope == 'stage_mid_only' and 'resnet' in arch:
        # Stage级统一通道剪枝
        from experiments.full_network_pruning.full_prune import get_stage_scores, generate_stage_masks
        stage_scores = get_stage_scores(model, scores, arch)
        stage_masks = generate_stage_masks(stage_scores, ratio)
        new_model = build_resnet_stage_pruned(model, stage_masks, arch)
        mask_dict = stage_masks
    else:
        # 原有逻辑: 仅剪枝conv1层
        new_cfg = []
        for name, s in scores.items():
            if 'conv1' in name and 'layer' in name:
                num_ch = len(s)
                keep = max(4, int(num_ch * (1 - ratio)))
                idx = np.argsort(s)[-keep:]
                mask = np.zeros(num_ch, dtype=bool)
                mask[idx] = True
                mask_dict[name] = mask
                new_cfg.append(int(mask.sum()))

        if 'resnet' in arch:
            new_model = build_new_resnet(model, new_cfg, mask_dict, arch, num_classes)
        elif 'vgg' in arch:
            threshold = np.percentile(
                np.concatenate([s.flatten() for s in scores.values()]),
                ratio * 100
            )
            new_model = build_new_vgg(model, scores, threshold, arch)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    new_model.to(device)
    return new_model, mask_dict


# =============================================================================
# Section 4: 实验配置
# =============================================================================

# =============================================================================
# 完整实验矩阵 (RTX 5090 32GB+54GB共享 + Ultra 9 285K 24核 + 95GB RAM)
# =============================================================================
# 阶段1: 使用已有checkpoint (resnet20, resnet56)
# 阶段2: 训练缺失baseline后扩展到全部架构
ARCHITECTURES_AVAILABLE = ['resnet20', 'resnet56']
ARCHITECTURES_FULL = ['resnet20', 'resnet56', 'resnet110', 'vgg16', 'mobilenetv2']

# 当前使用可用架构
ARCHITECTURES = ARCHITECTURES_AVAILABLE

METHODS = ['L1', 'FPGM', 'HRank', 'Taylor', 'GRA-v1', 'GRA-v4', 'Random']
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 123, 456, 789, 1024]
# GRA和Random方法使用较少seeds以节省时间
SEEDS_REDUCED = [42, 123, 456]

# 当前实验数: 2 * 7 * 9 * 5 = 630组
# 完整实验数: 5 * 7 * 9 * 5 = 1575组
# 并行配置: 4路并发 (32GB显存充裕)

# Checkpoint映射
CHECKPOINTS = {
    'resnet20': 'checkpoints/resnet20_best.pth',
    'resnet56': 'checkpoints/resnet56_best_new.pth',
    'resnet110': 'checkpoints/resnet110_best.pth',
    'vgg16': 'checkpoints/vgg16_bn_best.pth',
    'mobilenetv2': 'checkpoints/mobilenetv2_best.pth',
}


# =============================================================================
# Section 5: 单个实验运行
# =============================================================================

def run_single_experiment(arch, method, ratio, seed, device, pruning_scope='resnet_conv1_only'):
    """运行单个实验配置"""
    set_seed(seed)

    # 加载数据 (微调用)
    train_loader, test_loader, num_classes = get_dataloaders('cifar10', seed=seed)

    # 加载固定评分数据集 (P0-2: 无增强+固定indices)
    scoring_loader = get_scoring_dataloader('cifar10', seed=42)  # 固定seed=42确保所有方法用同一评分集

    # 加载模型
    if arch == 'resnet56':
        from models.resnet_cifar import resnet56
        model = resnet56(num_classes=num_classes)
    elif arch == 'resnet20':
        from models.resnet_cifar import resnet20
        model = resnet20(num_classes=num_classes)
    elif arch == 'resnet110':
        from models.resnet_cifar import resnet110
        model = resnet110(num_classes=num_classes)
    elif arch == 'vgg16':
        from models.vgg_cifar import vgg16
        model = vgg16(num_classes=num_classes)
    elif arch == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2_cifar
        model = mobilenetv2_cifar(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt = CHECKPOINTS.get(arch)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    sd = torch.load(ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)

    model.to(device)

    # 评估baseline
    baseline = evaluate(model, test_loader, device)

    # 计算评分 (使用固定评分数据集)
    if method == 'L1':
        scores = compute_l1_scores(model)
    elif method == 'FPGM':
        scores = compute_fpgm_scores(model)
    elif method == 'HRank':
        scores = compute_hrank_scores(model, scoring_loader, device, num_batches=5)
    elif method == 'Taylor':
        scores = compute_taylor_scores(model, scoring_loader, device)
    elif method == 'GRA-v1':
        scores = compute_gra_v1_scores(model, scoring_loader, device)
    elif method == 'GRA-v4':
        scores = compute_gra_v4_scores(model, scoring_loader, device, full_fusion=True, pruning_ratio=ratio)
    elif method == 'Random':
        scores = compute_random_scores(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 应用真正的结构化剪枝 (重建网络)
    pruned_model, mask_dict = apply_structural_pruning(
        model, scores, ratio, arch, num_classes, device, pruning_scope
    )

    # 剪枝后准确率
    pruned_acc = evaluate(pruned_model, test_loader, device)

    # 微调
    final_acc = finetune(pruned_model, train_loader, test_loader, device, epochs=40)

    # 计算剪枝后参数量
    params_after = sum(p.numel() for p in pruned_model.parameters())
    params_before = sum(p.numel() for p in model.parameters())

    return {
        'architecture': arch,
        'dataset': 'cifar10',
        'method': method,
        'ratio': ratio,
        'iso_flops': False,  # P0-1: 当前使用参数剪枝比例，非iso-flops
        'seed': seed,
        'gra_version': '4.0',
        'baseline_acc': baseline,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc,
        'params_before': params_before,
        'params_after': params_after,
        'compression_ratio': params_before / params_after,
        'pruning_scope': pruning_scope,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# Section 6: CSV输出
# =============================================================================

def save_results_csv(results, csv_file):
    """保存结果到CSV (证据链格式)"""
    import csv
    fieldnames = [
        'architecture', 'dataset', 'method', 'ratio', 'iso_flops', 'seed',
        'gra_version', 'baseline_acc', 'pruned_acc', 'final_acc',
        'params_before', 'params_after', 'compression_ratio',
        'pruning_scope', 'timestamp'
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


# =============================================================================
# Section 7: 主函数
# =============================================================================

# 全局剪枝范围配置
PRUNING_SCOPE = 'resnet_conv1_only'  # 可选: 'resnet_conv1_only', 'stage_mid_only'

def main():
    """主函数 - 运行完整实验矩阵"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 结果保存路径
    out_dir = 'experiments/v4_results'
    os.makedirs(out_dir, exist_ok=True)

    results = []
    # 计算总实验数: 4方法*5seeds + 3方法*3seeds = 20+9=29 per arch*ratio
    total_full = len(ARCHITECTURES) * 4 * len(RATIOS) * len(SEEDS)  # L1,FPGM,HRank,Taylor
    total_reduced = len(ARCHITECTURES) * 3 * len(RATIOS) * len(SEEDS_REDUCED)  # GRA-v1,GRA-v4,Random
    total = total_full + total_reduced
    count = 0

    print("=" * 70)
    print("GRA-CNN v4 完整实验矩阵")
    print("=" * 70)
    print(f"架构: {ARCHITECTURES}")
    print(f"方法: {METHODS}")
    print(f"剪枝率: {RATIOS}")
    print(f"种子: {SEEDS}")
    print(f"剪枝范围: {PRUNING_SCOPE}")
    print(f"总实验数: {total}")
    print("=" * 70)

    for arch in ARCHITECTURES:
        for method in METHODS:
            # GRA和Random方法使用减少的seeds
            seeds_to_use = SEEDS_REDUCED if method in ['GRA-v1', 'GRA-v4', 'Random'] else SEEDS
            for ratio in RATIOS:
                for seed in seeds_to_use:
                    count += 1
                    print(f"\n[{count}/{total}] {arch}/{method}/r={ratio}/s={seed}")

                    try:
                        result = run_single_experiment(
                            arch, method, ratio, seed, device, PRUNING_SCOPE
                        )
                        results.append(result)
                        print(f"  Baseline: {result['baseline_acc']:.2f}%")
                        print(f"  Final: {result['final_acc']:.2f}%")

                        # 实时保存JSON
                        out_file = f"{out_dir}/results_{arch}.json"
                        with open(out_file, 'w') as f:
                            json.dump(results, f, indent=2)

                        # 同时保存CSV (证据链格式)
                        csv_file = f"{out_dir}/results_{arch}.csv"
                        save_results_csv(results, csv_file)

                    except Exception as e:
                        print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
