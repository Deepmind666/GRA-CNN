"""
GRA v7.0: Margin-Attribution Semantic Pruning (MASP)

Key innovations:
1. Margin Attribution: grad x act w.r.t. margin (causal, not correlational)
2. Difficulty Weighting: hard-sample emphasis via sigmoid gating
3. Class-Coverage Breadth: entropy over per-class contributions (class-aware)
4. Regime-Adaptive Normalization: ratio-aware value/rank mixing per metric
5. Quality-Gated Fusion: multi-split reliability + contrast + uncertainty
6. Mid-Ratio Stabilization: disagreement shrinkage at r~0.5
"""

# ===========================================================================
# SECTION 1: Imports and constants
# ===========================================================================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

GRA_VERSION = "7.0"
_EPS = 1e-8


@dataclass
class FusionParams:
    """Fusion weight parameters for GRA v7.0 scoring."""
    fisher: float = 0.30
    l1: float = 0.36
    semantic: float = 0.24
    ortho: float = 0.10
    reliability: float = 0.0
    uncertainty: float = 0.0
    quality: float = 0.0


# ===========================================================================
# SECTION 2: Architecture detection (reuse from v5.2)
# ===========================================================================
def detect_architecture(model: nn.Module) -> Tuple[str, int]:
    """Detect model architecture type and stage count.

    Uses module names (layer1/layer2/features) for robust detection,
    matching the convention used across all GRA algorithm versions.

    Returns:
        Tuple of (architecture_type, max_stage_or_conv_count)
    """
    names = [name for name, _ in model.named_modules()]
    if any("layer1" in n for n in names):
        if any("layer4" in n for n in names):
            return "resnet", 4
        if any("layer3" in n for n in names):
            return "resnet", 3
        return "resnet", 2
    if any("features" in n for n in names):
        conv_count = sum(
            1 for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and "features" in n
        )
        if conv_count > 0:
            return "vgg", conv_count
    if any("features" in n and "conv" in n for n in names):
        return "mobilenet", 17
    return "unknown", 4


def get_layer_depth_ratio(layer_name: str, model: nn.Module) -> float:
    """Get the depth ratio (0.0 to 1.0) of a named layer within the model.

    Uses architecture-aware depth mapping matching v5.2/v5.51 conventions.

    Args:
        layer_name: Name of the layer (e.g. 'layer2.0.conv1').
        model: The neural network model.

    Returns:
        Float in [0, 1] indicating relative depth.
    """
    arch_type, max_stage = detect_architecture(model)
    if arch_type == "resnet":
        if "conv1" in layer_name and "layer" not in layer_name:
            return 0.0
        for i in range(1, max_stage + 1):
            if f"layer{i}" in layer_name:
                return float(i) / max_stage
        return 0.5
    if arch_type == "vgg":
        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "features" in name:
                if name == layer_name:
                    return idx / max(max_stage - 1, 1)
                idx += 1
        return 0.5
    return 0.5


# ===========================================================================
# SECTION 3: Numerical utilities
# ===========================================================================
def _safe_standardize(x: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Standardize array to zero mean, unit variance with clipping."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sigma = x.std()
    if sigma < _EPS:
        return np.zeros_like(x)
    z = (x - mu) / sigma
    return np.clip(z, -clip, clip)


def _safe_minmax01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Per-channel when ndim==2 (from v5.52).

    For [N, C] matrices, normalizes each channel (column) independently.
    Constant channels get value 0.5.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        # Per-channel: normalize along axis=0 (samples)
        mins = x.min(axis=0, keepdims=True)
        maxs = x.max(axis=0, keepdims=True)
        span = maxs - mins
        out = np.where(span > _EPS, (x - mins) / (span + _EPS), 0.5)
        return out
    lo = x.min()
    hi = x.max()
    rng = hi - lo
    if rng < _EPS:
        return np.full_like(x, 0.5)
    return (x - lo) / rng


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Rank data with average tie-breaking. Returns ranks in [1, n]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = np.arange(1, n + 1, dtype=np.float64)

    sorted_x = x[sorter]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[sorter[k]] = avg_rank
        i = j
    return ranks


def _spearman_corr01(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation mapped to [0, 1].

    Uses Pearson-on-ranks (handles ties correctly, unlike the simplified
    1 - 6*sum(d^2)/(n*(n^2-1)) formula).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 3 or len(b) < 3:
        return 0.5
    ra = _rankdata_average_ties(a)
    rb = _rankdata_average_ties(b)
    # Pearson correlation on ranks
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = np.linalg.norm(ra_c) * np.linalg.norm(rb_c)
    if denom < _EPS:
        return 0.5
    rho = float(np.dot(ra_c, rb_c) / denom)
    return float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))


def _normalize_quantile(scores: np.ndarray, q_low: float = 0.05,
                         q_high: float = 0.95) -> np.ndarray:
    """Quantile-based normalization to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    lo = np.quantile(scores, q_low)
    hi = np.quantile(scores, q_high)
    rng = hi - lo
    if rng < _EPS:
        return _safe_minmax01(scores)
    normed = (scores - lo) / rng
    return np.clip(normed, 0.0, 1.0)


def _normalize_rank(scores: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n <= 1:
        return np.ones_like(scores) * 0.5
    ranks = _rankdata_average_ties(scores)
    return (ranks - 1.0) / max(n - 1, 1)


def _mid_ratio_factor(r: float) -> float:
    """Compute mid-ratio factor: exp(-((r-0.50)/0.10)^2)."""
    return math.exp(-((r - 0.50) / 0.10) ** 2)


def _score_flatness(v: np.ndarray) -> float:
    """Detect flat distributions. Returns value in [0, 1].

    A flat distribution (all values similar) returns close to 1.0.
    A highly varied distribution returns close to 0.0.
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2:
        return 1.0
    std = v.std()
    rng = v.max() - v.min()
    if rng < _EPS:
        return 1.0
    cv = std / (np.abs(v.mean()) + _EPS)
    flatness = math.exp(-2.0 * cv)
    return float(np.clip(flatness, 0.0, 1.0))


# ===========================================================================
# SECTION 4: Regime-Adaptive Normalization
# ===========================================================================
_LAMBDA_BASES = {
    "fisher": 0.28,
    "l1": 0.22,
    "semantic": 0.25,
    "ortho": 0.22,
}


def normalize_regime_adaptive(scores: np.ndarray, pruning_ratio: float,
                               metric: str) -> Tuple[np.ndarray, float]:
    """Regime-adaptive normalization mixing value and rank normalization.

    lambda = lambda_base * (1 - 0.70*mid) * (1 - flat)

    Args:
        scores: Raw scores array.
        pruning_ratio: Current pruning ratio.
        metric: One of fisher, l1, semantic, ortho.

    Returns:
        Tuple of (normalized_scores, lambda_value).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return scores, 0.0

    mid = _mid_ratio_factor(pruning_ratio)
    flat = _score_flatness(scores)
    lambda_base = _LAMBDA_BASES.get(metric, 0.25)
    lam = lambda_base * (1.0 - 0.70 * mid) * (1.0 - flat)
    lam = float(np.clip(lam, 0.0, 1.0))

    val_norm = _normalize_quantile(scores)
    rank_norm = _normalize_rank(scores)

    blended = (1.0 - lam) * val_norm + lam * rank_norm
    return blended, lam


# ===========================================================================
# SECTION 5: Data Collection
# ===========================================================================
def collect_activations_and_margins(model: nn.Module, dataloader,
                                     device: torch.device,
                                     num_batches: int = 12
                                     ) -> Tuple[Dict[str, list], torch.Tensor, torch.Tensor]:
    """Collect per-layer activations, classification margins, and labels.

    Hooks on Conv2d layers capture activations (mean-pooled from 4D to 2D).
    Margins are computed as z_correct - z_max_wrong for each sample.

    Args:
        model: The neural network model.
        dataloader: DataLoader providing (inputs, labels) batches.
        device: Torch device.
        num_batches: Number of batches to collect.

    Returns:
        Tuple of (activations_dict, margins_tensor, labels_tensor)
        activations_dict maps layer_name -> list of [N, C] tensors
    """
    model.eval()
    activations = defaultdict(list)
    hooks = []
    all_margins = []
    all_labels = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            # Mean pool 4D -> 2D: [N, C, H, W] -> [N, C]
            if out.dim() == 4:
                pooled = out.mean(dim=[2, 3])
            else:
                pooled = out
            activations[name].append(pooled.detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)

            # Compute margin: z_correct - z_max_wrong
            batch_size = logits.size(0)
            num_classes = logits.size(1)
            z_correct = logits[torch.arange(batch_size, device=device), labels]
            # Mask out correct class to find max wrong
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size, device=device), labels] = False
            z_wrong = logits[mask].view(batch_size, num_classes - 1)
            z_max_wrong = z_wrong.max(dim=1)[0]
            margins = z_correct - z_max_wrong

            all_margins.append(margins.cpu())
            all_labels.append(labels.cpu())

    for h in hooks:
        h.remove()

    margins_tensor = torch.cat(all_margins, dim=0) if all_margins else torch.tensor([])
    labels_tensor = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([], dtype=torch.long)

    return dict(activations), margins_tensor, labels_tensor


# ===========================================================================
# SECTION 6: Margin Attribution Scoring (Innovation #1)
# ===========================================================================
def compute_margin_attribution_scores(model: nn.Module, dataloader,
                                       device: torch.device,
                                       num_batches: int = 12
                                       ) -> Tuple[Dict[str, np.ndarray], Dict[str, list]]:
    """Compute margin attribution scores via grad x act w.r.t. margin.

    For each batch:
      - Forward pass, compute margin = z_correct - z_max_wrong
      - Backward margin.mean() to get gradients on conv output activations
      - Per channel: |activation * gradient| averaged over spatial dims and samples
      - Weight by difficulty: w_i = sigmoid(gamma * (tau - margin_i))

    Args:
        model: The neural network model.
        dataloader: DataLoader providing (inputs, labels) batches.
        device: Torch device.
        num_batches: Number of batches to process.

    Returns:
        Tuple of (attribution_scores_dict, per_class_attribution_dict, per_sample_dict)
        attribution_scores_dict maps layer_name -> ndarray of shape [C]
        per_class_attribution_dict maps layer_name -> list of per-sample [C] arrays
        per_sample_dict maps layer_name -> ndarray of shape [N_total, C] (weighted attribution per sample)
    """
    model.eval()
    # We need gradients for margin attribution, so no torch.no_grad()
    conv_outputs = {}  # name -> tensor (with grad)
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if out.requires_grad:
                out.retain_grad()
            conv_outputs[name] = out
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Accumulators
    layer_scores_accum = defaultdict(list)  # name -> list of [C] arrays
    per_class_accum = defaultdict(list)  # name -> list of (label, [C]) tuples
    per_sample_accum = defaultdict(list)  # name -> list of [N, C] arrays
    all_margins_for_tau = []

    # First pass: collect all margins to compute tau
    margin_list = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            bs = logits.size(0)
            nc = logits.size(1)
            z_c = logits[torch.arange(bs, device=device), labels]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(bs, device=device), labels] = False
            z_w = logits[mask].view(bs, nc - 1).max(dim=1)[0]
            margin_list.append((z_c - z_w).cpu())

    if margin_list:
        all_margins_cat = torch.cat(margin_list, dim=0)
        tau = float(all_margins_cat.median())
    else:
        tau = 0.0
    gamma = 1.5

    # Second pass: compute attribution with gradients
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device).requires_grad_(True)
        labels = labels.to(device)
        conv_outputs.clear()
        model.zero_grad()

        logits = model(inputs)
        bs = logits.size(0)
        nc = logits.size(1)
        z_c = logits[torch.arange(bs, device=device), labels]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(bs, device=device), labels] = False
        z_w = logits[mask].view(bs, nc - 1).max(dim=1)[0]
        margins = z_c - z_w  # [N]

        # Difficulty weights: w_i = sigmoid(gamma * (tau - margin_i))
        diff_weights = torch.sigmoid(gamma * (tau - margins))  # [N]

        # Backward pass on margin mean
        loss = margins.mean()
        loss.backward()

        # Compute per-channel attribution for each layer
        for name, act in conv_outputs.items():
            if act.grad is None:
                continue
            grad = act.grad  # [N, C, H, W]
            # |activation * gradient| averaged over spatial dims -> [N, C]
            if act.dim() == 4:
                attr = (act * grad).abs().mean(dim=[2, 3])  # [N, C]
            else:
                attr = (act * grad).abs()  # [N, C]

            # Apply difficulty weighting
            w = diff_weights.unsqueeze(1).to(attr.device)  # [N, 1]
            weighted_attr = (attr * w).detach().cpu().numpy()  # [N, C]
            # Mean over samples
            layer_scores_accum[name].append(weighted_attr.mean(axis=0))  # [C]
            # Per-sample accumulation for reliability estimation
            per_sample_accum[name].append(weighted_attr)  # [N, C]

            # Per-class accumulation
            labels_np = labels.cpu().numpy()
            attr_np = attr.detach().cpu().numpy()  # [N, C]
            for i in range(bs):
                per_class_accum[name].append((int(labels_np[i]), attr_np[i]))

    for h in hooks:
        h.remove()

    # Aggregate
    attribution_scores = {}
    for name, score_list in layer_scores_accum.items():
        attribution_scores[name] = np.mean(np.stack(score_list, axis=0), axis=0)

    # Per-sample matrices for reliability estimation
    per_sample_dict = {}
    for name, sample_list in per_sample_accum.items():
        per_sample_dict[name] = np.concatenate(sample_list, axis=0)  # [N_total, C]

    return attribution_scores, dict(per_class_accum), per_sample_dict


# ===========================================================================
# SECTION 7: Class-Coverage Breadth (Innovation #3)
# ===========================================================================
def compute_class_breadth(per_class_data: list, num_channels: int) -> np.ndarray:
    """Compute class-coverage breadth via entropy over per-class contributions.

    For each channel, compute entropy over per-class attribution distribution.
    Higher entropy = channel serves more classes = more important to keep.

    Args:
        per_class_data: List of (label, attribution_array) tuples.
        num_channels: Number of channels in the layer.

    Returns:
        ndarray of shape [C] with breadth scores.
    """
    if not per_class_data or num_channels == 0:
        return np.ones(max(num_channels, 1), dtype=np.float64) * 0.5

    # Group by class
    class_attrs = defaultdict(list)
    for label, attr in per_class_data:
        class_attrs[label].append(attr)

    num_classes = len(class_attrs)
    if num_classes <= 1:
        return np.ones(num_channels, dtype=np.float64) * 0.5

    # For each class k, compute u_k = mean(|a*g|) for samples of class k -> [C]
    class_means = np.zeros((num_classes, num_channels), dtype=np.float64)
    for idx, (cls, attr_list) in enumerate(sorted(class_attrs.items())):
        stacked = np.stack(attr_list, axis=0)  # [N_k, C]
        class_means[idx] = np.abs(stacked).mean(axis=0)  # [C]

    # For each channel, compute entropy of normalized class distribution
    breadth = np.zeros(num_channels, dtype=np.float64)
    max_entropy = math.log(num_classes) if num_classes > 1 else 1.0

    for c in range(num_channels):
        u = class_means[:, c]  # [K]
        u_sum = u.sum()
        if u_sum < _EPS:
            breadth[c] = 0.0
            continue
        p = u / u_sum  # Normalize to probability distribution
        p = p + _EPS  # Avoid log(0)
        p = p / p.sum()  # Re-normalize after epsilon
        entropy = -np.sum(p * np.log(p))
        breadth[c] = entropy / max_entropy if max_entropy > _EPS else 0.0

    return breadth


# ===========================================================================
# SECTION 8: Standard Component Scores (from v5.2)
# ===========================================================================
def compute_fisher_scores(model: nn.Module, dataloader,
                           device: torch.device,
                           num_batches: int = 12) -> Dict[str, np.ndarray]:
    """Compute activation-weighted Fisher information scores.

    Fisher score per channel: (grad * weight)^2 aggregated per output channel.

    Args:
        model: The neural network model.
        dataloader: DataLoader providing (inputs, labels) batches.
        device: Torch device.
        num_batches: Number of batches to process.

    Returns:
        Dict mapping layer_name -> ndarray of shape [C] with Fisher scores.
    """
    model.eval()
    fisher_accum = defaultdict(lambda: None)
    count = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()

        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                # (grad * weight)^2 summed over (C_in, kH, kW) -> [C_out]
                gw = (module.weight.grad * module.weight).detach()
                score = (gw ** 2).sum(dim=[1, 2, 3]).cpu().numpy()
                if fisher_accum[name] is None:
                    fisher_accum[name] = np.zeros_like(score)
                fisher_accum[name] += score
        count += 1

    fisher_scores = {}
    for name, accum in fisher_accum.items():
        if accum is not None and count > 0:
            fisher_scores[name] = accum / count

    return fisher_scores


def compute_l1_scores(model: nn.Module) -> Dict[str, np.ndarray]:
    """Compute L1-norm scores per output channel for each Conv2d layer.

    Args:
        model: The neural network model.

    Returns:
        Dict mapping layer_name -> ndarray of shape [C] with L1 scores.
    """
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().numpy()
            # L1 norm per output channel: sum |w| over (C_in, kH, kW)
            score = np.abs(w).sum(axis=(1, 2, 3))
            l1_scores[name] = score.astype(np.float64)
    return l1_scores


def compute_orthogonality_scores(model: nn.Module) -> Dict[str, np.ndarray]:
    """Compute orthogonality-based importance scores per channel.

    Channels whose filters are more orthogonal to others are more important
    (they capture unique information).

    Args:
        model: The neural network model.

    Returns:
        Dict mapping layer_name -> ndarray of shape [C] with ortho scores.
    """
    ortho_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().numpy()
            c_out = w.shape[0]
            if c_out < 2:
                ortho_scores[name] = np.ones(c_out, dtype=np.float64)
                continue
            # Flatten each filter: [C_out, C_in*kH*kW]
            flat = w.reshape(c_out, -1).astype(np.float64)
            # Normalize rows
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms = np.where(norms < _EPS, 1.0, norms)
            flat_normed = flat / norms
            # Cosine similarity matrix
            cos_sim = flat_normed @ flat_normed.T  # [C_out, C_out]
            # For each channel, max cosine similarity with any other channel
            np.fill_diagonal(cos_sim, -1.0)
            max_sim = cos_sim.max(axis=1)
            # Higher orthogonality (lower max similarity) = higher importance
            ortho_scores[name] = np.clip(1.0 - max_sim, 0.0, 1.0).astype(np.float64)
    return ortho_scores


# ===========================================================================
# SECTION 9: Reliability Estimation
# ===========================================================================
def estimate_reliability_multisplit(scores: np.ndarray, num_splits: int = 6,
                                    seed: int = 42
                                    ) -> Tuple[float, float]:
    """Estimate reliability via multi-split Spearman correlation.

    K random half-splits of the score array, compute Spearman between halves.
    reliability = median(correlations), uncertainty = clip(IQR/0.25, 0, 1)

    Args:
        scores: 1D or 2D score array. If 2D, treated as [samples, channels].
        num_splits: Number of random half-splits.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reliability, uncertainty) both in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        # Cannot split a 1D score vector meaningfully; return moderate values
        return 0.5, 0.5

    n_samples = scores.shape[0]
    if n_samples < 4:
        return 0.5, 0.5

    rng = np.random.RandomState(seed)
    correlations = []

    for _ in range(num_splits):
        perm = rng.permutation(n_samples)
        half = n_samples // 2
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]
        mean_a = scores[idx_a].mean(axis=0)  # [C]
        mean_b = scores[idx_b].mean(axis=0)  # [C]
        corr = _spearman_corr01(mean_a, mean_b)
        correlations.append(corr)

    correlations = np.array(correlations)
    reliability = float(np.median(correlations))
    q75 = np.percentile(correlations, 75)
    q25 = np.percentile(correlations, 25)
    iqr = q75 - q25
    uncertainty = float(np.clip(iqr / 0.25, 0.0, 1.0))

    return reliability, uncertainty


# ===========================================================================
# SECTION 10: Semantic Quality Assessment
# ===========================================================================
def estimate_semantic_contrast(scores: np.ndarray) -> float:
    """Estimate semantic contrast based on score distribution spread.

    Based on std and q90-q10 spread.

    Args:
        scores: 1D array of scores.

    Returns:
        Contrast value in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) < 3:
        return 0.5
    std = scores.std()
    q90 = np.percentile(scores, 90)
    q10 = np.percentile(scores, 10)
    spread = q90 - q10
    mean_abs = np.abs(scores.mean()) + _EPS
    # Combine normalized std and spread
    cv = std / mean_abs
    spread_norm = spread / mean_abs
    # Map to [0, 1] via sigmoid-like transform
    raw = 0.5 * cv + 0.5 * spread_norm
    contrast = 1.0 - math.exp(-1.5 * raw)
    return float(np.clip(contrast, 0.0, 1.0))


def compute_semantic_quality(reliability: float, uncertainty: float,
                              contrast: float) -> float:
    """Compute overall semantic quality score.

    Q = 0.45*R + 0.25*C + 0.30*(1-U)

    Args:
        reliability: Reliability score in [0, 1].
        uncertainty: Uncertainty score in [0, 1].
        contrast: Contrast score in [0, 1].

    Returns:
        Quality score in [0, 1].
    """
    q = 0.45 * reliability + 0.25 * contrast + 0.30 * (1.0 - uncertainty)
    return float(np.clip(q, 0.0, 1.0))


# ===========================================================================
# SECTION 11: Adaptive Fusion Weights
# ===========================================================================
def get_fusion_params(layer_name: str, model: nn.Module,
                      pruning_ratio: float, quality: float,
                      reliability: float, uncertainty: float
                      ) -> FusionParams:
    """Compute adaptive fusion weights for a given layer.

    Base weights by ratio regime:
        r<=0.40: fisher=0.28, l1=0.42, semantic=0.20, ortho=0.10
        r<=0.65: fisher=0.30, l1=0.34, semantic=0.26, ortho=0.10
        r>0.65:  fisher=0.32, l1=0.36, semantic=0.22, ortho=0.10

    Depth shift: semantic gets +0.06*depth, l1 loses same.
    Quality gate: semantic *= (0.30 + 0.70*quality), backoff to l1/fisher.
    Normalize to sum=1.

    Args:
        layer_name: Name of the layer.
        model: The neural network model.
        pruning_ratio: Current pruning ratio.
        quality: Semantic quality score.
        reliability: Reliability score.
        uncertainty: Uncertainty score.

    Returns:
        FusionParams with computed weights.
    """
    r = pruning_ratio

    # Base weights by ratio regime
    if r <= 0.40:
        w_f, w_l, w_s, w_o = 0.28, 0.42, 0.20, 0.10
    elif r <= 0.65:
        w_f, w_l, w_s, w_o = 0.30, 0.34, 0.26, 0.10
    else:
        w_f, w_l, w_s, w_o = 0.32, 0.36, 0.22, 0.10

    # Depth shift: semantic gets +0.06*depth, l1 loses same
    depth = get_layer_depth_ratio(layer_name, model)
    depth_shift = 0.06 * depth
    w_s += depth_shift
    w_l -= depth_shift

    # Quality gate: semantic *= (0.30 + 0.70*quality), backoff to l1/fisher
    quality_gate = 0.30 + 0.70 * quality
    semantic_original = w_s
    w_s *= quality_gate
    backoff = semantic_original - w_s
    # Distribute backoff to l1 and fisher proportionally
    w_l += backoff * 0.60
    w_f += backoff * 0.40

    # Normalize to sum=1
    total = w_f + w_l + w_s + w_o
    if total > _EPS:
        w_f /= total
        w_l /= total
        w_s /= total
        w_o /= total

    return FusionParams(
        fisher=w_f,
        l1=w_l,
        semantic=w_s,
        ortho=w_o,
        reliability=reliability,
        uncertainty=uncertainty,
        quality=quality,
    )


# ===========================================================================
# SECTION 12: Main Scoring Function
# ===========================================================================
def _cache_batches(dataloader, device: torch.device,
                    num_batches: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Cache dataloader batches to CPU for deterministic multi-pass reuse."""
    cached = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        cached.append((inputs.cpu(), labels.cpu()))
    return cached


def compute_gra_v7_scores(model: nn.Module, dataloader,
                           device: torch.device,
                           num_batches: int = 12,
                           pruning_ratio: float = 0.5,
                           verbose: bool = True
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Compute GRA v7.0 MASP fused importance scores for all Conv2d layers.

    Steps:
      0. Cache dataloader batches (all steps share identical samples)
      1. Collect activations, margins, labels
      2. Compute margin attribution scores (with difficulty weighting)
      3. Compute class breadth scores
      4. Combine: semantic = normalize(attribution) * (1 + 0.15*normalize(breadth))
      5. Compute Fisher, L1, Ortho scores
      6. For each layer: reliability, quality, fusion, fallback, shrinkage
      7. Return scores and rich metadata per layer

    Args:
        model: The neural network model.
        dataloader: DataLoader providing (inputs, labels) batches.
        device: Torch device.
        num_batches: Number of batches to process.
        pruning_ratio: Target pruning ratio.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (scores_dict, metadata_dict)
        scores_dict maps layer_name -> ndarray of fused importance scores [C]
        metadata_dict maps layer_name -> dict with quality, reliability, etc.
    """
    r = pruning_ratio
    mid = _mid_ratio_factor(r)

    if verbose:
        print(f"[GRA v{GRA_VERSION}] Starting MASP scoring with pruning_ratio={r:.2f}")
        print(f"[GRA v{GRA_VERSION}] Mid-ratio factor: {mid:.4f}")

    # ------------------------------------------------------------------
    # Step 0: Cache batches so all steps use identical samples
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 0: Caching {num_batches} batches...")
    cached_batches = _cache_batches(dataloader, device, num_batches)
    if verbose:
        total_samples = sum(x.size(0) for x, _ in cached_batches)
        print(f"[GRA v{GRA_VERSION}]   Cached {len(cached_batches)} batches, {total_samples} samples")

    # ------------------------------------------------------------------
    # Step 1: Compute margin attribution scores (with difficulty weighting)
    #         Also collects per-class and per-sample data for breadth & reliability
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 1: Computing margin attribution scores...")
    attribution_scores, per_class_accum, per_sample_semantic = compute_margin_attribution_scores(
        model, cached_batches, device, num_batches=len(cached_batches)
    )

    # ------------------------------------------------------------------
    # Step 2: Compute class breadth scores
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 2: Computing class-coverage breadth...")
    breadth_scores = {}
    for name in attribution_scores:
        num_ch = len(attribution_scores[name])
        pc_data = per_class_accum.get(name, [])
        breadth_scores[name] = compute_class_breadth(pc_data, num_ch)

    # ------------------------------------------------------------------
    # Step 3: Combine semantic = normalize(attribution) * (1 + 0.15*normalize(breadth))
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 3: Combining attribution and breadth...")
    semantic_raw = {}
    for name in attribution_scores:
        attr_n = _safe_minmax01(attribution_scores[name])
        brd_n = _safe_minmax01(breadth_scores[name])
        semantic_raw[name] = attr_n * (1.0 + 0.15 * brd_n)

    # ------------------------------------------------------------------
    # Step 4: Compute Fisher, L1, Ortho scores
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 4: Computing Fisher, L1, Ortho scores...")
    fisher_scores = compute_fisher_scores(model, cached_batches, device, num_batches=len(cached_batches))
    l1_scores = compute_l1_scores(model)
    ortho_scores = compute_orthogonality_scores(model)

    # ------------------------------------------------------------------
    # Step 5: Per-layer fusion with quality gating and fallback
    # ------------------------------------------------------------------
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 5: Per-layer adaptive fusion...")

    # Get all layer names from L1 (always available)
    all_layer_names = list(l1_scores.keys())
    final_scores = {}
    metadata = {}

    # Collect per-layer quality values for global threshold
    layer_qualities = {}

    # First pass: compute quality for each layer to get global threshold
    for name in all_layer_names:
        sem = semantic_raw.get(name, None)
        # Use per-sample semantic attribution for reliability (not raw activations)
        sem_samples = per_sample_semantic.get(name, None)
        if sem is not None and sem_samples is not None and sem_samples.shape[0] >= 4:
            rel, unc = estimate_reliability_multisplit(
                sem_samples, num_splits=6, seed=42
            )
            contrast = estimate_semantic_contrast(sem)
            quality = compute_semantic_quality(rel, unc, contrast)
        else:
            rel, unc, contrast, quality = 0.5, 0.5, 0.5, 0.5

        layer_qualities[name] = {
            "quality": quality,
            "reliability": rel,
            "uncertainty": unc,
            "contrast": contrast,
        }

    # Compute global quality threshold
    all_q_values = [v["quality"] for v in layer_qualities.values()]
    if all_q_values:
        q_global = float(np.quantile(all_q_values, 0.20))
    else:
        q_global = 0.5

    # Second pass: per-layer fusion
    for name in all_layer_names:
        lq = layer_qualities[name]
        quality = lq["quality"]
        rel = lq["reliability"]
        unc = lq["uncertainty"]
        contrast = lq["contrast"]
        depth = get_layer_depth_ratio(name, model)

        # 6a. Get fusion params
        fp = get_fusion_params(name, model, r, quality, rel, unc)

        # Get raw scores for this layer
        s_fisher = fisher_scores.get(name, None)
        s_l1 = l1_scores.get(name, None)
        s_semantic = semantic_raw.get(name, None)
        s_ortho = ortho_scores.get(name, None)

        if s_l1 is None:
            continue

        num_ch = len(s_l1)

        # Fill missing scores with zeros
        if s_fisher is None:
            s_fisher = np.zeros(num_ch, dtype=np.float64)
        if s_semantic is None:
            s_semantic = np.zeros(num_ch, dtype=np.float64)
        if s_ortho is None:
            s_ortho = np.ones(num_ch, dtype=np.float64) * 0.5

        # 6d. Normalize all 4 components with regime-adaptive normalization
        fisher_n, lam_fisher = normalize_regime_adaptive(s_fisher, r, "fisher")
        l1_n, lam_l1 = normalize_regime_adaptive(s_l1, r, "l1")
        semantic_n, lam_semantic = normalize_regime_adaptive(s_semantic, r, "semantic")
        ortho_n, lam_ortho = normalize_regime_adaptive(s_ortho, r, "ortho")

        # 6e. Compute conservative baseline: B = 0.65*L1_n + 0.35*Fisher_n
        baseline = 0.65 * l1_n + 0.35 * fisher_n

        # 6f. Quality-gate semantic: S_adj = g*S_n + (1-g)*B
        #     where g = clip(0.20 + 0.80*Q, 0.20, 1.0)
        g = float(np.clip(0.20 + 0.80 * quality, 0.20, 1.0))
        semantic_adj = g * semantic_n + (1.0 - g) * baseline

        # 6g. Fuse: fused = w_f*F_n + w_l*L_n + w_s*S_adj + w_o*O_n
        fused = (
            fp.fisher * fisher_n
            + fp.l1 * l1_n
            + fp.semantic * semantic_adj
            + fp.ortho * ortho_n
        )

        # 6h. Dynamic fallback: if Q < dynamic_threshold, drop semantic
        #     mid = exp(-((r-0.50)/0.10)^2)
        #     tau_ratio = 0.48 + 0.08*mid
        #     tau = clip(tau_ratio + 0.05*U + 0.03*(0.5-depth), 0.40, 0.70)
        #     q_global = quantile(all_Q_values, 0.20)
        #     tau_eff = max(tau, q_global - 0.02)
        #     fallback = (Q < tau_eff)
        tau_ratio = 0.48 + 0.08 * mid
        tau = float(np.clip(
            tau_ratio + 0.05 * unc + 0.03 * (0.5 - depth),
            0.40, 0.70
        ))
        tau_eff = max(tau, q_global - 0.02)
        fallback = quality < tau_eff

        if fallback:
            # Drop semantic entirely, use conservative baseline
            fused = baseline
            if verbose:
                print(f"[GRA v{GRA_VERSION}]   {name}: FALLBACK "
                      f"(Q={quality:.3f} < tau_eff={tau_eff:.3f})")

        # 6i. Mid-ratio disagreement shrinkage
        #     agree = spearman(S_adj, B)
        #     dis = clip((0.55-agree)/0.55, 0, 1)
        #     weak = clip((0.60-Q)/0.60, 0, 1)
        #     gamma = 0.30 * mid_factor * dis * weak
        #     fused = (1-gamma)*fused + gamma*B
        agreement = _spearman_corr01(semantic_adj, baseline)
        dis = float(np.clip((0.55 - agreement) / 0.55, 0.0, 1.0))
        weak = float(np.clip((0.60 - quality) / 0.60, 0.0, 1.0))
        gamma_shrink = 0.30 * mid * dis * weak
        fused = (1.0 - gamma_shrink) * fused + gamma_shrink * baseline

        final_scores[name] = fused

        # Metadata per layer
        metadata[name] = {
            "quality": quality,
            "reliability": rel,
            "uncertainty": unc,
            "contrast": contrast,
            "fallback": fallback,
            "weights": {
                "fisher": fp.fisher,
                "l1": fp.l1,
                "semantic": fp.semantic,
                "ortho": fp.ortho,
            },
            "mid_factor": mid,
            "gamma": gamma_shrink,
            "agreement": agreement,
            "norm_lambda": {
                "fisher": lam_fisher,
                "l1": lam_l1,
                "semantic": lam_semantic,
                "ortho": lam_ortho,
            },
            "depth": depth,
            "quality_gate_g": g,
            "tau_eff": tau_eff,
        }

    if verbose:
        n_fallback = sum(1 for v in metadata.values() if v["fallback"])
        n_total = len(metadata)
        print(f"[GRA v{GRA_VERSION}] Done. {n_total} layers scored, "
              f"{n_fallback} fallbacks.")

    return final_scores, metadata


# Backward compatibility alias
compute_gra_v7_final_score = compute_gra_v7_scores
