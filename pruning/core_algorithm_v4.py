"""
GRA-CNN v4.0: Pearson-GRA Hybrid with Depth-Adaptive Fusion
============================================================

v4.0 核心创新点:
1. Pearson-GRA混合评分: 结合相关性分析与灰色关联分析
2. 深度自适应权重: 浅层偏重L1稳定性，深层偏重语义对齐
3. 真正的结构化剪枝: 重建网络而非权重置零
4. 完整的实验矩阵支持: ratio从0.1到0.9

理论基础:
- 信息瓶颈理论: 浅层保留更多信息，深层压缩到任务相关特征
- 灰色关联分析: 衡量序列趋势一致性
- Pearson相关: 线性相关性度量

Author: GRA-CNN Team
Version: 4.0
Date: 2026-02-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Algorithm version
GRA_VERSION = "4.0"


# =============================================================================
# Section 1: Architecture Detection
# =============================================================================

def detect_architecture(model):
    """
    检测网络架构类型和深度信息

    Returns:
        arch_type: 'resnet', 'vgg', 'mobilenet', 'unknown'
        max_stage: 最大stage数 (ResNet) 或 conv层数 (VGG)
    """
    layer_names = [name for name, _ in model.named_modules()]

    # ResNet检测
    has_layer4 = any('layer4' in n for n in layer_names)
    has_layer3 = any('layer3' in n for n in layer_names)
    has_layer1 = any('layer1' in n for n in layer_names)

    if has_layer1:
        if has_layer4:
            return 'resnet', 4  # ResNet-50/101/152
        elif has_layer3:
            return 'resnet', 3  # ResNet-20/56/110 (CIFAR)
        else:
            return 'resnet', 2

    # VGG检测
    if any('features' in n for n in layer_names):
        conv_count = sum(1 for n, m in model.named_modules()
                        if isinstance(m, nn.Conv2d) and 'features' in n)
        return 'vgg', conv_count

    # MobileNet检测
    if any('features' in n and 'conv' in n for n in layer_names):
        return 'mobilenet', 17

    return 'unknown', 4


def get_layer_depth_ratio(layer_name, model):
    """
    计算层的相对深度 (0.0 ~ 1.0)

    用于深度自适应权重分配
    """
    arch_type, max_stage = detect_architecture(model)

    if arch_type == 'resnet':
        # ResNet: 按stage划分
        if 'conv1' in layer_name and 'layer' not in layer_name:
            return 0.0
        elif 'layer1' in layer_name:
            return 1.0 / max_stage
        elif 'layer2' in layer_name:
            return 2.0 / max_stage
        elif 'layer3' in layer_name:
            return 3.0 / max_stage
        elif 'layer4' in layer_name:
            return 4.0 / max_stage
        return 0.5

    elif arch_type == 'vgg':
        # VGG: 按conv层顺序
        conv_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'features' in name:
                if name == layer_name:
                    return conv_idx / max(max_stage - 1, 1)
                conv_idx += 1
        return 0.5

    return 0.5


# =============================================================================
# Section 2: Pearson-GRA Hybrid Score (v4.0 核心创新)
# =============================================================================

def compute_pearson_correlation(act_seq, ref_seq, eps=1e-8):
    """
    计算Pearson相关系数

    Args:
        act_seq: [B] 通道激活序列
        ref_seq: [B] 参考序列 (margin)

    Returns:
        corr: 相关系数 [-1, 1]
    """
    act_centered = act_seq - act_seq.mean()
    ref_centered = ref_seq - ref_seq.mean()

    numerator = (act_centered * ref_centered).sum()
    denominator = (act_centered.pow(2).sum().sqrt() + eps) * \
                  (ref_centered.pow(2).sum().sqrt() + eps)

    return numerator / denominator


def compute_gra_coefficient(act_seq, ref_seq, rho=0.5, eps=1e-8):
    """
    计算灰色关联系数

    Args:
        act_seq: [B] 归一化后的通道激活序列
        ref_seq: [B] 归一化后的参考序列
        rho: 分辨系数 (0.5为标准值)

    Returns:
        gamma: 灰色关联度 [0, 1]
    """
    delta = (act_seq - ref_seq).abs()
    d_min, d_max = delta.min(), delta.max()

    # 修复: delta全0时返回1 (完全一致)
    if d_max < eps:
        return torch.tensor(1.0)

    gamma = (d_min + rho * d_max) / (delta + rho * d_max + eps)
    return gamma.mean()


def compute_pearson_gra_hybrid(act_seq, margin_seq, rho=0.5, alpha=0.6, eps=1e-8):
    """
    v4.0 Pearson-GRA混合评分

    核心思想:
    - Pearson: 捕捉线性趋势一致性 (当margin高时，重要通道激活也高)
    - GRA: 捕捉序列形状相似性 (对非线性关系更鲁棒)
    - 混合: score = alpha * pearson_score + (1-alpha) * gra_score

    Args:
        act_seq: [B, C] 通道激活矩阵
        margin_seq: [B] 分类margin序列
        rho: GRA分辨系数
        alpha: Pearson权重 (默认0.6，偏重线性相关)

    Returns:
        scores: [C] 每个通道的混合评分
    """
    B, C = act_seq.shape

    # 归一化margin
    margin_norm = (margin_seq - margin_seq.min()) / \
                  (margin_seq.max() - margin_seq.min() + eps)

    scores = torch.zeros(C)

    for c in range(C):
        act_c = act_seq[:, c]

        # 归一化激活
        act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + eps)

        # 1. Pearson相关系数 [-1, 1] -> [0, 1]
        pearson = compute_pearson_correlation(act_c, margin_seq, eps)
        pearson_score = (pearson + 1) / 2

        # 2. GRA系数 [0, 1]
        gra_score = compute_gra_coefficient(act_norm, margin_norm, rho, eps)

        # 3. 混合评分
        scores[c] = alpha * pearson_score + (1 - alpha) * gra_score

    return scores


# =============================================================================
# Section 3: Component Scores (Fisher, L1, Orthogonality)
# =============================================================================

def compute_fisher_scores(model, dataloader, device, num_batches=10, eps=1e-8):
    """
    计算Fisher信息评分

    Fisher信息衡量参数对损失的敏感度
    高Fisher = 参数变化对损失影响大 = 重要
    """
    original_training = model.training

    fisher_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fisher_scores[name] = torch.zeros(module.out_channels, device=device)

    if not fisher_scores:
        return {}

    model.eval()
    num_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)
        num_samples += images.size(0)

        model.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                grad = module.weight.grad
                fisher_per_ch = (grad ** 2).view(grad.size(0), -1).sum(dim=1)
                fisher_scores[name] += fisher_per_ch

    # 归一化
    for name in fisher_scores:
        fisher_scores[name] = (fisher_scores[name] / max(num_samples, 1)).cpu().numpy()

    model.train(original_training)
    return fisher_scores


def compute_l1_scores(model):
    """
    计算L1范数评分

    简单但有效: 权重绝对值大的滤波器更重要
    """
    l1_scores = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            l1_per_ch = weight.abs().view(weight.size(0), -1).sum(dim=1)
            l1_scores[name] = l1_per_ch.cpu().numpy()

    return l1_scores


def compute_orthogonality_scores(model, eps=1e-8):
    """
    计算正交性评分

    与其他滤波器正交的滤波器捕获独特信息
    高正交性 = 独特特征 = 重要
    """
    ortho_scores = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            out_c = weight.size(0)

            if out_c < 2:
                ortho_scores[name] = np.ones(out_c)
                continue

            # 展平每个滤波器
            W = weight.view(out_c, -1)
            W_norm = W / (W.norm(dim=1, keepdim=True) + eps)

            # 余弦相似度矩阵
            sim_matrix = torch.mm(W_norm, W_norm.t())
            sim_matrix.fill_diagonal_(-1)

            # 最大相似度
            max_sim = sim_matrix.max(dim=1).values

            # 正交性 = 1 - 最大相似度
            ortho = (1 - max_sim).clamp(0, 1).cpu().numpy()
            ortho_scores[name] = ortho

    return ortho_scores


# =============================================================================
# Section 4: Depth-Adaptive Weight Allocation (v4.0 核心)
# =============================================================================

def get_adaptive_weights_v4(layer_name, model, pruning_ratio=0.5):
    """
    v4.0 深度+剪枝率 双自适应权重分配

    深度自适应 (信息瓶颈理论):
    - 浅层: 高L1权重(稳定), 低GRA权重(梯度噪声大)
    - 深层: 高GRA权重(语义对齐), 低L1权重

    剪枝率自适应:
    - 低剪枝率(r<0.5): 偏L1/FPGM简单指标，降低Fisher/Ortho噪声
    - 高剪枝率(r>=0.5): 保持多维融合，GRA权重提升

    Returns:
        weights: dict with Fisher, L1, GRA, Ortho weights
        rho: 自适应分辨系数
    """
    depth_ratio = get_layer_depth_ratio(layer_name, model)

    # 基础权重
    base = {'Fisher': 0.30, 'L1': 0.35, 'GRA': 0.25, 'Ortho': 0.10}

    # 深度自适应调整
    depth_adj = 0.10 * depth_ratio

    # 剪枝率自适应调整
    # ratio_factor: 0.0 at r=0.3, 1.0 at r=0.9
    ratio_factor = max(0.0, min(1.0, (pruning_ratio - 0.3) / 0.6))
    # 低剪枝率: 降低Fisher/Ortho噪声，增大L1
    # 高剪枝率: 增大GRA/Fisher，降低L1
    ratio_adj = 0.08 * (ratio_factor - 0.5)  # [-0.04, +0.04]

    w_gra = base['GRA'] + depth_adj + ratio_adj
    w_l1 = max(base['L1'] - depth_adj - ratio_adj, 0.10)
    w_fisher = base['Fisher'] + ratio_adj * 0.5
    w_ortho = max(base['Ortho'] - abs(ratio_adj) * 0.5, 0.02)

    # 归一化
    total = w_gra + w_l1 + w_fisher + w_ortho
    weights = {
        'Fisher': w_fisher / total,
        'L1': w_l1 / total,
        'GRA': w_gra / total,
        'Ortho': w_ortho / total
    }

    # 自适应rho: 浅层0.3(敏感), 深层0.7(稳定)
    rho = 0.3 + 0.4 * depth_ratio

    return weights, rho


# =============================================================================
# Section 5: Data Collection
# =============================================================================

def collect_activations_and_margins(model, dataloader, device, num_batches=10):
    """
    收集通道激活序列和margin序列

    margin = correct_logit - max_wrong_logit (分类置信度)
    用于GRA语义对齐评分

    Returns:
        activations: dict[layer_name] -> tensor [N, C]
        margins: tensor [N]
    """
    original_training = model.training
    act_data = defaultdict(list)
    all_margins = []

    def make_hook(name):
        def hook(_module, _inp, out):
            if out.dim() == 4:
                act = out.mean(dim=[2, 3])  # 空间平均
            else:
                act = out
            act_data[name].append(act.detach().cpu())
        return hook

    # 注册hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                logits = model(images)

                # 计算margin
                correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze()
                logits_copy = logits.clone()
                logits_copy.scatter_(1, labels.unsqueeze(1), float('-inf'))
                max_wrong = logits_copy.max(dim=1).values

                margins = correct_logits - max_wrong
                all_margins.append(margins.cpu())

    finally:
        for h in handles:
            h.remove()
        model.train(original_training)

    # 合并结果
    activations = {name: torch.cat(acts, dim=0) for name, acts in act_data.items()}
    margins = torch.cat(all_margins, dim=0)

    return activations, margins


# =============================================================================
# Section 6: Main Scoring Function
# =============================================================================

def normalize_scores(scores, eps=1e-8):
    """Min-max归一化到[0, 1]"""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < eps:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min + eps)


def compute_gra_v4_scores(model, dataloader, device, num_batches=10, verbose=True, pruning_ratio=0.5):
    """
    v4.0 主评分函数

    整合所有组件:
    1. Fisher信息 (梯度敏感度)
    2. L1范数 (权重幅度)
    3. Pearson-GRA混合 (语义对齐)
    4. 正交性 (特征独特性)

    Returns:
        scores: dict[layer_name] -> np.array
        metadata: 额外信息
    """
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Computing importance scores...")

    # Step 1: 收集激活和margin
    if verbose:
        print("  Step 1: Collecting activations and margins...")
    activations, margins = collect_activations_and_margins(
        model, dataloader, device, num_batches
    )

    # Step 2: 计算各组件评分
    if verbose:
        print("  Step 2: Computing Fisher scores...")
    fisher_scores = compute_fisher_scores(model, dataloader, device, num_batches)

    if verbose:
        print("  Step 3: Computing L1 scores...")
    l1_scores = compute_l1_scores(model)

    if verbose:
        print("  Step 4: Computing orthogonality scores...")
    ortho_scores = compute_orthogonality_scores(model)

    # Step 3: 融合评分
    if verbose:
        print("  Step 5: Fusing scores with depth-adaptive weights...")

    final_scores = {}
    metadata = {'weights': {}, 'version': GRA_VERSION}

    for layer_name in activations.keys():
        act = activations[layer_name]
        num_channels = act.size(1)

        # 获取深度自适应权重
        weights, rho = get_adaptive_weights_v4(layer_name, model, pruning_ratio)
        metadata['weights'][layer_name] = weights

        # 计算Pearson-GRA混合评分
        gra = compute_pearson_gra_hybrid(act, margins, rho=rho, alpha=0.6)
        gra_np = gra.numpy() if isinstance(gra, torch.Tensor) else gra

        # 获取其他评分
        fisher = fisher_scores.get(layer_name, np.ones(num_channels))
        l1 = l1_scores.get(layer_name, np.ones(num_channels))
        ortho = ortho_scores.get(layer_name, np.ones(num_channels))

        # 归一化
        fisher_norm = normalize_scores(fisher)
        l1_norm = normalize_scores(l1)
        gra_norm = normalize_scores(gra_np)
        ortho_norm = normalize_scores(ortho)

        # 加权融合
        combined = (weights['Fisher'] * fisher_norm +
                   weights['L1'] * l1_norm +
                   weights['GRA'] * gra_norm +
                   weights['Ortho'] * ortho_norm)

        final_scores[layer_name] = combined

    if verbose:
        print(f"  Done! Computed scores for {len(final_scores)} layers.")

    return final_scores, metadata


# =============================================================================
# Section 7: Layer Protection
# =============================================================================

def get_layer_protection_ratio(layer_name, model, num_channels):
    """
    自适应层保护比例

    原则:
    - 首尾层: 最多剪枝60% (关键层)
    - 小通道层(<64): 最多剪枝50%
    - 中间层: 最多剪枝85%

    Returns:
        max_prune_ratio: 该层允许的最大剪枝比例
    """
    depth_ratio = get_layer_depth_ratio(layer_name, model)

    # 首尾层保护
    if depth_ratio < 0.15 or depth_ratio > 0.85:
        max_prune = 0.60
    # 小通道保护
    elif num_channels < 64:
        max_prune = 0.50
    # 中间层可以激进剪枝
    else:
        max_prune = 0.85

    return max_prune


# =============================================================================
# Section 8: Pruning Mask Generation
# =============================================================================

def generate_pruning_masks(scores, prune_ratio, model=None, min_channels=4):
    """
    生成剪枝掩码

    Args:
        scores: dict[layer_name] -> importance scores
        prune_ratio: 目标剪枝比例 (0.5 = 剪掉50%通道)
        model: 用于层保护
        min_channels: 每层最少保留通道数

    Returns:
        masks: dict[layer_name] -> binary mask (1=keep, 0=prune)
    """
    masks = {}

    for layer_name, layer_scores in scores.items():
        num_channels = len(layer_scores)

        # 获取层保护比例
        if model is not None:
            max_prune = get_layer_protection_ratio(layer_name, model, num_channels)
        else:
            max_prune = 0.85

        # 实际剪枝比例
        actual_prune = min(prune_ratio, max_prune)
        keep_count = max(min_channels, int(num_channels * (1 - actual_prune)))

        # 按分数排序，保留top-k
        sorted_indices = np.argsort(layer_scores)[::-1]  # 降序
        mask = np.zeros(num_channels)
        mask[sorted_indices[:keep_count]] = 1
        masks[layer_name] = mask

    return masks


# =============================================================================
# Section 9: Iso-FLOPs Pruning
# =============================================================================

def estimate_layer_flops(module, spatial_size, out_channels):
    """估算单层FLOPs"""
    if not isinstance(module, nn.Conv2d):
        return 0

    k = module.kernel_size[0]
    c_in = module.in_channels
    stride = module.stride[0]
    groups = module.groups

    h_out = spatial_size // stride
    c_in_per_group = c_in // groups

    return 2 * k * k * c_in_per_group * out_channels * h_out * h_out


def get_layer_flops_info(model, input_size=32, device='cuda'):
    """
    获取每层FLOPs信息

    修复: 使用dummy forward + hook获取真实空间尺寸
    """
    layer_info = {}
    output_sizes = {}

    # 使用hook获取真实输出尺寸
    def make_hook(name):
        def hook(module, inp, out):
            if out.dim() == 4:
                output_sizes[name] = (out.size(2), out.size(3))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    # Dummy forward
    try:
        model.eval()
        dummy = torch.zeros(1, 3, input_size, input_size).to(device)
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()

    # 计算FLOPs
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            c_out = module.out_channels
            h, w = output_sizes.get(name, (input_size, input_size))

            # FLOPs per channel
            k = module.kernel_size[0]
            c_in = module.in_channels
            groups = module.groups
            c_in_per_group = c_in // groups

            flops_per_ch = 2 * k * k * c_in_per_group * h * w

            layer_info[name] = {
                'module': module,
                'channels': c_out,
                'flops_per_channel': flops_per_ch,
                'total_flops': flops_per_ch * c_out,
                'spatial_size': (h, w)
            }

    return layer_info


def generate_iso_flops_masks(model, scores, target_flops_ratio,
                              input_size=32, min_channels=4, verbose=False):
    """
    Iso-FLOPs剪枝掩码生成

    确保剪枝后FLOPs精确达到目标比例

    Args:
        model: 模型
        scores: 重要性评分
        target_flops_ratio: 目标保留FLOPs比例 (0.5 = 保留50%)
        input_size: 输入空间尺寸
        min_channels: 每层最少保留通道数

    Returns:
        masks: 剪枝掩码
        actual_ratio: 实际FLOPs比例
    """
    if verbose:
        print(f"[Iso-FLOPs] Target ratio: {target_flops_ratio:.1%}")

    flops_info = get_layer_flops_info(model, input_size)
    total_flops = sum(info['total_flops'] for info in flops_info.values())

    # 二分搜索找到合适的阈值
    low, high = 0.0, 1.0
    best_masks = {}
    best_ratio = 1.0

    for _ in range(20):
        mid = (low + high) / 2
        masks = {}
        current_flops = 0

        for name, layer_scores in scores.items():
            if name not in flops_info:
                continue

            info = flops_info[name]
            num_ch = len(layer_scores)

            # 层保护
            max_prune = get_layer_protection_ratio(name, model, num_ch)
            keep_ratio = max(1 - mid, 1 - max_prune)
            keep_count = max(min_channels, int(num_ch * keep_ratio))

            # 生成掩码
            sorted_idx = np.argsort(layer_scores)[::-1]
            mask = np.zeros(num_ch)
            mask[sorted_idx[:keep_count]] = 1
            masks[name] = mask

            current_flops += info['flops_per_channel'] * keep_count

        current_ratio = current_flops / total_flops

        if abs(current_ratio - target_flops_ratio) < abs(best_ratio - target_flops_ratio):
            best_masks = masks.copy()
            best_ratio = current_ratio

        if current_ratio > target_flops_ratio:
            low = mid
        else:
            high = mid

    if verbose:
        print(f"  Achieved ratio: {best_ratio:.1%}")

    return best_masks, best_ratio


# =============================================================================
# Section 10: Main API
# =============================================================================

def prune_model_v4(model, dataloader, device, prune_ratio=0.5,
                   use_iso_flops=True, input_size=32, verbose=True):
    """
    v4.0 主剪枝API

    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        device: 设备
        prune_ratio: 目标剪枝比例
        use_iso_flops: 是否使用Iso-FLOPs约束
        input_size: 输入空间尺寸
        verbose: 是否打印详情

    Returns:
        masks: 剪枝掩码
        scores: 重要性评分
        metadata: 元数据
    """
    if verbose:
        print("=" * 60)
        print(f"GRA-CNN v{GRA_VERSION} Pruning")
        print("=" * 60)
        print(f"  Target prune ratio: {prune_ratio:.0%}")
        print(f"  Iso-FLOPs: {use_iso_flops}")
        print("-" * 60)

    # Step 1: 计算重要性评分
    scores, metadata = compute_gra_v4_scores(
        model, dataloader, device, num_batches=10, verbose=verbose,
        pruning_ratio=prune_ratio
    )

    # Step 2: 生成剪枝掩码
    if use_iso_flops:
        target_ratio = 1.0 - prune_ratio
        masks, actual_ratio = generate_iso_flops_masks(
            model, scores, target_ratio, input_size, verbose=verbose
        )
        metadata['actual_flops_ratio'] = actual_ratio
    else:
        masks = generate_pruning_masks(scores, prune_ratio, model)

    if verbose:
        print("-" * 60)
        print("Pruning complete!")
        print("=" * 60)

    return masks, scores, metadata


# =============================================================================
# Section 11: Simplified Scoring for Quick Experiments
# =============================================================================

def compute_gra_v4_simple(model, dataloader, device, num_batches=20):
    """
    简化版GRA v4评分 - 用于快速实验

    只计算Pearson-GRA混合评分，不包含Fisher和正交性
    速度更快，适合大规模实验矩阵

    Returns:
        scores: dict[layer_name] -> np.array
    """
    activations, margins = collect_activations_and_margins(
        model, dataloader, device, num_batches
    )

    scores = {}
    for layer_name, act in activations.items():
        depth_ratio = get_layer_depth_ratio(layer_name, model)
        rho = 0.3 + 0.4 * depth_ratio

        gra = compute_pearson_gra_hybrid(act, margins, rho=rho, alpha=0.6)
        scores[layer_name] = gra.numpy()

    return scores


# =============================================================================
# Section 12: Backward Compatibility Aliases
# =============================================================================

# 向后兼容别名
compute_gra_final_score = compute_gra_v4_scores
get_global_mask_iso_flops = generate_iso_flops_masks
get_pruning_mask_simple = generate_pruning_masks
