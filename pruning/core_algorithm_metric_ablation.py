"""
Metric ablation: replace GRA semantic branch with cosine/correlation-based redundancy.

Purpose: controlled experiment to test whether the GRA relational coefficient is
uniquely valuable, or whether any similarity metric (cosine, Pearson correlation)
achieves comparable results when plugged into the same fusion pipeline.

Design: Steps 0, 4, 5 of the v8 pipeline are kept IDENTICAL.
Only Steps 1-3 (semantic branch) are replaced.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any, Dict, Tuple

from pruning.core_algorithm_v7 import (
    get_layer_depth_ratio,
    _safe_minmax01, _mid_ratio_factor,
    normalize_regime_adaptive,
    compute_fisher_scores, compute_l1_scores, compute_orthogonality_scores,
    _cache_batches, _EPS,
)
from pruning.core_algorithm_v7 import estimate_semantic_contrast
from pruning.core_algorithm_v8 import (
    estimate_topk_reliability, compute_critic_weights,
)

VERSION = "ablation-metric-1.0"


# ===================================================================
# Replacement semantic branch: class-profile inter-channel similarity
# ===================================================================

def _cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for rows of mat [C, K]."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + _EPS
    normed = mat / norms
    return normed @ normed.T


def _correlation_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation for rows of mat [C, K]."""
    C = mat.shape[0]
    if C < 2:
        return np.ones((C, C))
    centered = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + _EPS
    normed = centered / norms
    corr = normed @ normed.T
    return np.clip(corr, -1.0, 1.0)


def compute_class_profile_semantic(
    model: nn.Module, cached_batches: list, device: torch.device,
    num_batches: int = 12, metric: str = "cosine",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute per-channel semantic importance via inter-channel class-profile similarity.

    For each Conv2d layer:
      1. Collect mean channel activations per class across calibration batches
      2. Build class profile vector per channel: v_c = [mean_act_class0, ..., mean_act_classK]
      3. Compute pairwise similarity between all channel pairs (cosine or correlation)
      4. importance_c = 1 - mean_similarity_c  (less redundant = more important)

    Returns:
        semantic_scores: Dict[layer_name, np.ndarray[C]] per-channel importance
        per_sample_scores: Dict[layer_name, np.ndarray[N, C]] for reliability estimation
    """
    model.eval()

    # Register hooks on Conv2d layers
    hooks = []
    activations = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = out.detach()
        return hook_fn

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            hooks.append(module.register_forward_hook(make_hook(name)))
            conv_layers.append(name)

    # Collect per-class activations
    # class_accum[layer][class_label] = list of per-channel mean activations
    class_accum = defaultdict(lambda: defaultdict(list))
    # per_sample[layer] = list of [C] arrays (for reliability estimation)
    sample_accum = defaultdict(list)

    with torch.no_grad():
        for bi, (inp, lab) in enumerate(cached_batches):
            if bi >= num_batches:
                break
            inp, lab = inp.to(device), lab.to(device)
            model(inp)

            for name in conv_layers:
                act = activations.get(name)
                if act is None:
                    continue
                # act shape: [B, C, H, W] -> per-channel mean: [B, C]
                ch_mean = act.mean(dim=(2, 3)).cpu().numpy()
                sample_accum[name].append(ch_mean)

                labels_np = lab.cpu().numpy()
                for i in range(ch_mean.shape[0]):
                    class_accum[name][int(labels_np[i])].append(ch_mean[i])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute semantic scores per layer
    sim_fn = _cosine_similarity_matrix if metric == "cosine" else _correlation_matrix

    semantic_scores = {}
    per_sample_out = {}

    for name in conv_layers:
        # Build class profile matrix: [C, num_classes]
        class_dict = class_accum[name]
        if not class_dict:
            continue
        classes = sorted(class_dict.keys())
        num_channels = len(class_dict[classes[0]][0]) if class_dict[classes[0]] else 0
        if num_channels == 0:
            continue

        # Per-class mean activation per channel
        profile = np.zeros((num_channels, len(classes)), dtype=np.float64)
        for ci, cls in enumerate(classes):
            arrs = np.array(class_dict[cls])  # [N_cls, C]
            profile[:, ci] = arrs.mean(axis=0)

        # Pairwise similarity and derive importance
        sim_mat = sim_fn(profile)  # [C, C]
        np.fill_diagonal(sim_mat, 0.0)  # exclude self-similarity
        if num_channels > 1:
            mean_sim = sim_mat.sum(axis=1) / (num_channels - 1)
        else:
            mean_sim = np.zeros(num_channels)

        # importance = 1 - mean_similarity (less redundant = more important)
        # For correlation, values can be negative; use abs to measure redundancy
        if metric == "correlation":
            mean_sim = np.abs(mean_sim)

        importance = 1.0 - np.clip(mean_sim, 0.0, 1.0)
        semantic_scores[name] = importance

        # Per-sample scores for reliability estimation
        arrs = sample_accum.get(name)
        if arrs:
            per_sample_out[name] = np.concatenate(arrs, axis=0)

    return semantic_scores, per_sample_out


# ===================================================================
# Main scoring function (same interface as compute_gra_v8_scores)
# ===================================================================

def compute_metric_ablation_scores(
    model: nn.Module, dataloader, device: torch.device,
    num_batches: int = 12, pruning_ratio: float = 0.5, verbose: bool = True,
    metric: str = "cosine",
    gate_kappa: float = 10.0, gate_tau0: float = 0.55, gate_beta: float = 0.20,
    w_semantic_bounds: Tuple[float, float] = (0.10, 0.45),
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Metric ablation scoring. Same interface and fusion as GRA v8, different semantic branch."""
    r = pruning_ratio
    mid = _mid_ratio_factor(r)
    score_start = time.time()

    if verbose:
        print(f"[MetricAblation v{VERSION}] {metric} scoring (r={r:.2f})")

    # Step 0: Cache batches (same as v8)
    cached = _cache_batches(dataloader, device, num_batches)
    if verbose:
        ns = sum(x.size(0) for x, _ in cached)
        print(f"[MetricAblation]   Cached {len(cached)} batches, {ns} samples")

    # Steps 1-3 REPLACED: class-profile semantic scoring
    if verbose:
        print(f"[MetricAblation] Step 1-3: {metric} class-profile semantic...")
    semantic_raw, per_sample = compute_class_profile_semantic(
        model, cached, device, len(cached), metric=metric)

    # Step 4: Fisher, L1, Ortho (IDENTICAL to v8)
    if verbose:
        print(f"[MetricAblation] Step 4: Fisher, L1, Ortho...")
    fisher_s = compute_fisher_scores(model, cached, device, len(cached))
    l1_s = compute_l1_scores(model)
    ortho_s = compute_orthogonality_scores(model)

    # Step 5: Per-layer fusion (IDENTICAL to v8: CRITIC + soft gate)
    if verbose:
        print(f"[MetricAblation] Step 5: CRITIC fusion + soft gate...")

    all_layers = list(l1_s.keys())
    final_scores = {}
    metadata = {}
    k_ratio = float(np.clip(r, 0.2, 0.6))

    # Quality estimation per layer
    layer_q = {}
    for name in all_layers:
        ss = per_sample.get(name)
        sem = semantic_raw.get(name)
        if sem is not None and ss is not None and ss.shape[0] >= 4:
            rel, unc = estimate_topk_reliability(ss, k_ratio=k_ratio)
            con = estimate_semantic_contrast(sem)
            q = float(np.clip(0.45 * rel + 0.25 * con + 0.30 * (1.0 - unc), 0, 1))
        else:
            rel, unc, con, q = 0.5, 0.5, 0.5, 0.5
        layer_q[name] = dict(quality=q, reliability=rel,
                             uncertainty=unc, contrast=con)

    # Soft gate threshold (same as v8)
    tau_eff = gate_tau0 + gate_beta * (r - 0.5)
    gate_vals = []

    for name in all_layers:
        lq = layer_q[name]
        q = lq["quality"]
        depth = get_layer_depth_ratio(name, model)

        sf = fisher_s.get(name)
        sl = l1_s.get(name)
        ss = semantic_raw.get(name)
        so = ortho_s.get(name)
        if sl is None:
            continue
        nc = len(sl)
        if sf is None:
            sf = np.zeros(nc, dtype=np.float64)
        if ss is None:
            ss = np.zeros(nc, dtype=np.float64)
        if so is None:
            so = np.ones(nc, dtype=np.float64) * 0.5

        fn, _ = normalize_regime_adaptive(sf, r, "fisher")
        ln, _ = normalize_regime_adaptive(sl, r, "l1")
        sn, _ = normalize_regime_adaptive(ss, r, "semantic")
        on, _ = normalize_regime_adaptive(so, r, "ortho")

        baseline = 0.65 * ln + 0.35 * fn

        # CRITIC weights (same as v8)
        mat = np.stack([fn, ln, sn, on], axis=1)
        cw = compute_critic_weights(mat, semantic_idx=2,
                                    w_semantic_bounds=w_semantic_bounds,
                                    quality=q)
        full = cw[0] * fn + cw[1] * ln + cw[2] * sn + cw[3] * on

        # Soft gate (same as v8)
        g = 1.0 / (1.0 + math.exp(-gate_kappa * (q - tau_eff)))
        fused = (1.0 - g) * baseline + g * full
        gate_vals.append(g)

        final_scores[name] = fused
        metadata[name] = {
            "quality": q, "reliability": lq["reliability"],
            "uncertainty": lq["uncertainty"], "contrast": lq["contrast"],
            "fallback": g < 0.1, "gate_g": g, "depth": depth,
            "tau_eff": tau_eff,
            "weights": {"fisher": float(cw[0]), "l1": float(cw[1]),
                        "semantic": float(cw[2]), "ortho": float(cw[3])},
        }

    score_time = time.time() - score_start
    n_total = len(metadata)
    gate_mean = float(np.mean(gate_vals)) if gate_vals else 0.0
    fb_count = sum(1 for v in metadata.values() if v.get("fallback"))

    if verbose:
        print(f"[MetricAblation] Done. {n_total} layers, "
              f"gate_mean={gate_mean:.3f}, near_fallback={fb_count}, "
              f"score_time={score_time:.1f}s")

    metadata["__global__"] = {
        "version": VERSION, "metric": metric,
        "score_time_s": score_time,
        "gate_kappa": gate_kappa, "gate_mean": gate_mean,
        "fallback_rate": fb_count / max(n_total, 1),
        "quality_gate_mean": float(np.mean([
            v["quality"] for k, v in metadata.items() if k != "__global__"
        ])) if n_total > 0 else 0.0,
        "semantic_weight_mean": float(np.mean([
            v["weights"]["semantic"]
            for k, v in metadata.items() if k != "__global__"
        ])) if n_total > 0 else 0.0,
    }
    return final_scores, metadata
