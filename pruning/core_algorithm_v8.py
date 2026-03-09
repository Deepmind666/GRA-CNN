"""
GRA v8.0: Soft-Margin Attribution with CRITIC Fusion (SMAC)

Four improvements over v7.0 MASP:
  M1: Soft-Margin (logsumexp replaces hard max, MAD-adaptive difficulty weights)
  M2: SmoothGrad denoising (K noise samples for attribution averaging)
  M3: Soft gate (sigmoid continuous gate replaces hard fallback)
  M4: CRITIC data-driven weighting (replaces hand-tuned fusion weights)
"""

import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pruning.core_algorithm_v7 import (
    detect_architecture, get_layer_depth_ratio,
    _safe_minmax01, _rankdata_average_ties, _spearman_corr01,
    _mid_ratio_factor, _score_flatness,
    normalize_regime_adaptive,
    compute_fisher_scores, compute_l1_scores, compute_orthogonality_scores,
    compute_class_breadth, estimate_semantic_contrast, _cache_batches, _EPS,
)

GRA_VERSION = "8.0"


# ===================================================================
# M1+M2: Soft-Margin Attribution with SmoothGrad
# ===================================================================
def compute_margin_attribution_v8(
    model: nn.Module, cached_batches: list, device: torch.device,
    num_batches: int = 12, soft_margin_temp: float = 1.0,
    gamma0: float = 1.5, denoise_K: int = 4, sigma_x: float = 0.02,
) -> Tuple[Dict[str, np.ndarray], Dict[str, list], Dict[str, np.ndarray]]:
    """Soft-margin attribution with SmoothGrad denoising."""
    model.eval()
    Tm = soft_margin_temp

    # First pass: collect margins for tau + MAD
    margin_list = []
    with torch.no_grad():
        for bi, (inp, lab) in enumerate(cached_batches):
            if bi >= num_batches:
                break
            inp, lab = inp.to(device), lab.to(device)
            logits = model(inp)
            bs, nc = logits.shape
            z_c = logits[torch.arange(bs, device=device), lab]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(bs, device=device), lab] = False
            z_wrong = logits[mask].view(bs, nc - 1)
            soft_max_wrong = Tm * torch.logsumexp(z_wrong / Tm, dim=1)
            margin_list.append((z_c - soft_max_wrong).cpu())

    if margin_list:
        all_m = torch.cat(margin_list)
        tau = float(all_m.median())
        mad = float((all_m - all_m.median()).abs().median()) + 1e-8
    else:
        tau, mad = 0.0, 1.0

    # Setup hooks
    conv_outputs = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def _hook(n):
                def fn(m, i, o):
                    if o.requires_grad:
                        o.retain_grad()
                    conv_outputs[n] = o
                return fn
            hooks.append(mod.register_forward_hook(_hook(name)))

    # K-pass attribution
    layer_accum = defaultdict(lambda: None)
    per_class_accum = defaultdict(list)
    per_sample_accum = defaultdict(list)

    for k in range(denoise_K):
        for bi, (inp, lab) in enumerate(cached_batches):
            if bi >= num_batches:
                break
            inp, lab = inp.to(device), lab.to(device)
            if k > 0 and sigma_x > 0:
                inp = inp + torch.randn_like(inp) * sigma_x
            inp.requires_grad_(True)
            conv_outputs.clear()
            model.zero_grad()

            logits = model(inp)
            bs, nc = logits.shape
            z_c = logits[torch.arange(bs, device=device), lab]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(bs, device=device), lab] = False
            z_wrong = logits[mask].view(bs, nc - 1)
            margins = z_c - Tm * torch.logsumexp(z_wrong / Tm, dim=1)

            diff_w = torch.sigmoid(gamma0 * (tau - margins.detach()) / mad)
            margins.mean().backward()

            for name, act in conv_outputs.items():
                if act.grad is None:
                    continue
                attr = (act * act.grad).abs()
                if attr.dim() == 4:
                    attr = attr.mean(dim=[2, 3])
                weighted = (attr * diff_w.unsqueeze(1)).detach().cpu().numpy()
                batch_mean = weighted.mean(axis=0)
                if layer_accum[name] is None:
                    layer_accum[name] = np.zeros_like(batch_mean)
                layer_accum[name] += batch_mean
                if k == 0:
                    per_sample_accum[name].append(weighted)
                    for i in range(len(lab)):
                        per_class_accum[name].append(
                            (int(lab[i].item()), weighted[i]))

    for h in hooks:
        h.remove()

    total_passes = denoise_K * min(num_batches, len(cached_batches))
    attr_scores = {}
    for name, acc in layer_accum.items():
        if acc is not None and total_passes > 0:
            attr_scores[name] = acc / total_passes

    per_sample_out = {}
    for name, arrs in per_sample_accum.items():
        if arrs:
            per_sample_out[name] = np.concatenate(arrs, axis=0)

    return attr_scores, dict(per_class_accum), per_sample_out


# ===================================================================
# M3: Top-k Jaccard reliability (replaces v7 Spearman full-rank)
# ===================================================================
def estimate_topk_reliability(
    scores: np.ndarray, k_ratio: float = 0.3, n_splits: int = 8, seed: int = 42
) -> Tuple[float, float]:
    """How stable is the bottom-k pruning set across random half-splits."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1 or scores.shape[0] < 4:
        return 0.5, 0.5
    n_samples, n_ch = scores.shape
    k = max(2, int(n_ch * k_ratio))
    rng = np.random.RandomState(seed)
    overlaps = []
    for _ in range(n_splits):
        perm = rng.permutation(n_samples)
        half = n_samples // 2
        ma = scores[perm[:half]].mean(axis=0)
        mb = scores[perm[half:2 * half]].mean(axis=0)
        ba = set(np.argsort(ma)[:k])
        bb = set(np.argsort(mb)[:k])
        overlaps.append(len(ba & bb) / k)
    overlaps = np.array(overlaps)
    rel = float(np.median(overlaps))
    iqr = float(np.percentile(overlaps, 75) - np.percentile(overlaps, 25))
    unc = float(np.clip(iqr / 0.25, 0.0, 1.0))
    return rel, unc


# ===================================================================
# M4: CRITIC data-driven weighting
# ===================================================================
def compute_critic_weights(
    score_matrix: np.ndarray, semantic_idx: int = 2,
    w_semantic_bounds: Tuple[float, float] = (0.10, 0.45),
    quality: float = 1.0,
) -> np.ndarray:
    """CRITIC objective weighting: sigma * conflict."""
    C, M = score_matrix.shape
    if C < 3 or M < 2:
        return np.ones(M) / M

    sigma = score_matrix.std(axis=0) + _EPS
    ranked = np.zeros_like(score_matrix)
    for j in range(M):
        ranked[:, j] = _rankdata_average_ties(score_matrix[:, j])
    corr = np.corrcoef(ranked.T)
    if corr.shape != (M, M):
        return np.ones(M) / M

    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    conflict = (1.0 - np.abs(corr)).sum(axis=1)
    w = sigma * conflict
    w = w / (w.sum() + _EPS)

    # Constrain semantic + quality decay
    w[semantic_idx] = np.clip(w[semantic_idx],
                              w_semantic_bounds[0], w_semantic_bounds[1])
    w[semantic_idx] *= (0.3 + 0.7 * quality)
    w = w / (w.sum() + _EPS)
    return w


# ===================================================================
# Main scoring function
# ===================================================================
def compute_gra_v8_scores(
    model: nn.Module, dataloader, device: torch.device,
    num_batches: int = 12, pruning_ratio: float = 0.5, verbose: bool = True,
    soft_margin_temp: float = 1.0, gamma0: float = 1.5,
    denoise_K: int = 4, sigma_x: float = 0.02,
    gate_kappa: float = 10.0, gate_tau0: float = 0.55, gate_beta: float = 0.20,
    w_semantic_bounds: Tuple[float, float] = (0.10, 0.45),
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """GRA v8.0 SMAC fused importance scores. Same interface as v7."""
    r = pruning_ratio
    mid = _mid_ratio_factor(r)
    score_start = time.time()

    if verbose:
        print(f"[GRA v{GRA_VERSION}] SMAC scoring (r={r:.2f}, "
              f"Tm={soft_margin_temp}, K={denoise_K}, kappa={gate_kappa})")

    # Step 0: Cache batches
    cached = _cache_batches(dataloader, device, num_batches)
    if verbose:
        ns = sum(x.size(0) for x, _ in cached)
        print(f"[GRA v{GRA_VERSION}]   Cached {len(cached)} batches, {ns} samples")

    # Step 1: M1+M2 soft-margin attribution + SmoothGrad
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 1: Soft-margin attribution...")
    attr_scores, pc_data, per_sample = compute_margin_attribution_v8(
        model, cached, device, len(cached),
        soft_margin_temp, gamma0, denoise_K, sigma_x)

    # Step 2: Class breadth
    breadth = {}
    for name in attr_scores:
        breadth[name] = compute_class_breadth(
            pc_data.get(name, []), len(attr_scores[name]))

    # Step 3: Combine semantic
    semantic_raw = {}
    for name in attr_scores:
        an = _safe_minmax01(attr_scores[name])
        bn = _safe_minmax01(breadth[name])
        semantic_raw[name] = an * (1.0 + 0.15 * bn)

    # Step 4: Fisher, L1, Ortho
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 4: Fisher, L1, Ortho...")
    fisher_s = compute_fisher_scores(model, cached, device, len(cached))
    l1_s = compute_l1_scores(model)
    ortho_s = compute_orthogonality_scores(model)

    # Step 5: Per-layer fusion (M3 soft gate + M4 CRITIC)
    if verbose:
        print(f"[GRA v{GRA_VERSION}] Step 5: CRITIC fusion + soft gate...")

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
            q = float(np.clip(0.45*rel + 0.25*con + 0.30*(1.0-unc), 0, 1))
        else:
            rel, unc, con, q = 0.5, 0.5, 0.5, 0.5
        layer_q[name] = dict(quality=q, reliability=rel,
                             uncertainty=unc, contrast=con)

    # M3: tau_eff
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
        if sf is None: sf = np.zeros(nc, dtype=np.float64)
        if ss is None: ss = np.zeros(nc, dtype=np.float64)
        if so is None: so = np.ones(nc, dtype=np.float64) * 0.5

        fn, _ = normalize_regime_adaptive(sf, r, "fisher")
        ln, _ = normalize_regime_adaptive(sl, r, "l1")
        sn, _ = normalize_regime_adaptive(ss, r, "semantic")
        on, _ = normalize_regime_adaptive(so, r, "ortho")

        baseline = 0.65 * ln + 0.35 * fn

        # M4: CRITIC weights
        mat = np.stack([fn, ln, sn, on], axis=1)
        cw = compute_critic_weights(mat, semantic_idx=2,
                                    w_semantic_bounds=w_semantic_bounds,
                                    quality=q)
        full = cw[0]*fn + cw[1]*ln + cw[2]*sn + cw[3]*on

        # M3: soft gate
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
        print(f"[GRA v{GRA_VERSION}] Done. {n_total} layers, "
              f"gate_mean={gate_mean:.3f}, near_fallback={fb_count}, "
              f"score_time={score_time:.1f}s")

    metadata["__global__"] = {
        "version": GRA_VERSION, "score_time_s": score_time,
        "soft_margin_temp": soft_margin_temp, "denoise_K": denoise_K,
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


# Backward compat alias
compute_gra_v8_final_score = compute_gra_v8_scores
