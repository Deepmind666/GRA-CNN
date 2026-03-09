"""
GRA-CHIP: Gray Relational Analysis enhanced Channel Independence Pruning
========================================================================

Three innovations over CHIP (NeurIPS 2021):

1. Functional Redundancy Detection (FRD):
   GRA detects nonlinear activation-pattern similarity that CHIP's
   nuclear-norm (linear independence) misses. Channels that are
   linearly independent but functionally redundant get penalized.

2. Class-Aware GRA (CA-GRA):
   Per-class GRA redundancy -- a channel is truly redundant only if
   it is redundant across ALL classes. This prevents pruning channels
   that serve minority classes.

3. Confidence-Adaptive Fusion (CAF):
   When CHIP and GRA agree, trust the joint signal. When they
   disagree, be conservative (keep the channel). The fusion weight
   adapts per-channel based on agreement level.

Ablation flags: use_frd, use_ca, use_caf (all default True).
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


def _collect_feature_maps(model, dataloader, device, num_batches=5):
    """Collect feature maps and labels from all conv layers."""
    features = defaultdict(list)
    labels_list = []
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            features[name].append(out.detach().cpu())
        return hook_fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(x.to(device))
            labels_list.append(y)

    for h in hooks:
        h.remove()

    feats = {k: torch.cat(v, dim=0) for k, v in features.items()}
    labels = torch.cat(labels_list, dim=0).numpy()
    return feats, labels


def compute_chip_ci(feat):
    """
    CHIP original: Channel Independence via nuclear norm.
    feat: [N, C, H, W] tensor
    Returns: [C] array, higher = more important
    """
    N, C, H, W = feat.shape
    mat = feat.reshape(N, C, -1)  # [N, C, HW]
    ci = np.zeros(C)

    for n in range(min(N, 50)):  # subsample for speed
        orig_norm = torch.linalg.norm(mat[n], ord='nuc').item()
        for j in range(C):
            reduced = mat[n].clone()
            reduced[j] = 0
            ci[j] += orig_norm - torch.linalg.norm(reduced, ord='nuc').item()

    ci /= min(N, 50)
    return ci


def compute_gra_redundancy(feat, rho=0.5):
    """
    GRA inter-channel redundancy: how correlated each channel is with others.
    feat: [N, C, H, W] tensor
    Returns: [C] array, higher = more redundant (should be pruned)
    """
    N, C, H, W = feat.shape
    # Channel activation profile: mean activation per sample -> [N, C]
    profile = feat.mean(dim=(2, 3)).numpy()  # [N, C]

    # Normalize each channel's profile to [0,1]
    p_min = profile.min(axis=0, keepdims=True)
    p_max = profile.max(axis=0, keepdims=True)
    profile = (profile - p_min) / (p_max - p_min + 1e-8)

    # GRA correlation matrix: gra(i,j) for all channel pairs
    redundancy = np.zeros(C)
    for i in range(C):
        delta = np.abs(profile[:, i:i+1] - profile)  # [N, C]
        d_min = delta.min()
        d_max = delta.max()
        grc = (d_min + rho * d_max) / (delta + rho * d_max + 1e-8)  # [N, C]
        # Mean GRA coefficient with all other channels
        grc_mean = grc.mean(axis=0)  # [C]
        grc_mean[i] = 0  # exclude self
        redundancy[i] = grc_mean.sum() / max(C - 1, 1)

    return redundancy


def compute_ca_gra_redundancy(feat, labels, rho=0.5):
    """
    Innovation 2: Class-Aware GRA (CA-GRA).
    A channel is redundant only if redundant across ALL classes.
    Returns: [C] array, higher = more redundant.
    """
    C = feat.shape[1]
    profile = feat.mean(dim=(2, 3)).numpy()  # [N, C]

    classes = np.unique(labels)
    # Per-class redundancy: take the MIN across classes (conservative)
    per_class_red = []

    for cls in classes:
        mask = labels == cls
        if mask.sum() < 2:
            continue
        p = profile[mask]  # [Nc, C]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        p = (p - p_min) / (p_max - p_min + 1e-8)

        red = np.zeros(C)
        for i in range(C):
            delta = np.abs(p[:, i:i+1] - p)
            d_min, d_max = delta.min(), delta.max()
            grc = (d_min + rho * d_max) / (delta + rho * d_max + 1e-8)
            grc_mean = grc.mean(axis=0)
            grc_mean[i] = 0
            red[i] = grc_mean.sum() / max(C - 1, 1)
        per_class_red.append(red)

    if not per_class_red:
        return compute_gra_redundancy(feat, rho)

    # Channel is redundant only if redundant in ALL classes (min)
    return np.min(per_class_red, axis=0)


def _minmax(x):
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 1e-8 else np.ones_like(x) * 0.5


def compute_gra_chip_scores(model, dataloader, device,
                            num_batches=5, alpha=0.5, rho=0.5):
    """GRA-CHIP baseline: simple linear fusion (for ablation)."""
    feats, labels = _collect_feature_maps(model, dataloader, device, num_batches)
    scores = {}
    for name, feat in feats.items():
        C = feat.shape[1]
        if C < 2:
            scores[name] = np.ones(C)
            continue
        ci_n = _minmax(compute_chip_ci(feat))
        gra_n = _minmax(compute_gra_redundancy(feat, rho=rho))
        scores[name] = alpha * ci_n + (1 - alpha) * (1 - gra_n)
    return scores


def compute_gra_chip_v2_scores(
    model, dataloader, device, num_batches=5, alpha=0.5, rho=0.5,
    use_frd=True, use_ca=True, use_caf=True,
):
    """
    GRA-CHIP v2: full method with three innovations.

    Args:
        use_frd: Innovation 1 - Functional Redundancy Detection
        use_ca:  Innovation 2 - Class-Aware GRA
        use_caf: Innovation 3 - Confidence-Adaptive Fusion
    """
    feats, labels = _collect_feature_maps(model, dataloader, device, num_batches)
    scores = {}

    for name, feat in feats.items():
        C = feat.shape[1]
        if C < 2:
            scores[name] = np.ones(C)
            continue

        # CHIP structural importance
        ci = _minmax(compute_chip_ci(feat))

        # GRA functional redundancy
        if use_ca:
            gra_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho))
        else:
            gra_red = _minmax(compute_gra_redundancy(feat, rho))
        gra_imp = 1.0 - gra_red  # invert: low redundancy = important

        # Innovation 1: FRD penalty
        # Channels CHIP thinks important but GRA finds redundant
        if use_frd:
            disagreement = np.clip(ci - gra_imp, 0, None)  # positive where CHIP overestimates
            frd_penalty = 0.3 * disagreement
            ci = ci - frd_penalty

        # Innovation 3: Confidence-Adaptive Fusion
        if use_caf:
            # Agreement: both signals point same direction
            agreement = 1.0 - np.abs(ci - gra_imp)  # [0,1], 1=perfect agreement
            # When they agree, trust GRA more; when they disagree, trust CHIP (conservative)
            w_gra = alpha * agreement  # GRA weight scales with agreement
            w_ci = 1.0 - w_gra
            scores[name] = w_ci * ci + w_gra * gra_imp
        else:
            scores[name] = alpha * ci + (1 - alpha) * gra_imp

    return scores


def _estimate_quality_from_signals(anchor, sem_imp):
    """Estimate semantic quality from anchor/semantic agreement."""
    anchor_n = _minmax(anchor)
    sem_n = _minmax(sem_imp)
    # R: agreement (1 = high agreement)
    r_val = float(np.clip(1.0 - np.mean(np.abs(anchor_n - sem_n)), 0.0, 1.0))
    # U: uncertainty (higher std on semantic means lower uncertainty)
    u_val = float(np.clip(1.0 - np.std(sem_n), 0.0, 1.0))
    # C: semantic contrast (spread between p90 and p10)
    c_val = float(np.clip(np.percentile(sem_n, 90) - np.percentile(sem_n, 10), 0.0, 1.0))
    q_val = float(np.clip(0.45 * r_val + 0.35 * (1.0 - u_val) + 0.20 * c_val, 0.0, 1.0))
    return r_val, u_val, c_val, q_val


def _collect_ccr_risk_scores(
    model,
    dataloader,
    device,
    target_layers,
    num_batches=8,
):
    """Collect class-conditional risk scores from |act * grad| statistics."""
    target_set = set(target_layers or [])
    if not target_set:
        return {}

    if hasattr(dataloader.dataset, "classes"):
        num_classes = len(dataloader.dataset.classes)
    else:
        num_classes = 100

    layer_stats = {}
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = out
            if out.requires_grad:
                out.retain_grad()
        return hook_fn

    for name, module in model.named_modules():
        if name in target_set and isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad(set_to_none=True)
        activations.clear()

        logits = model(x)
        picked = logits.gather(1, y.unsqueeze(1)).sum()
        picked.backward()

        y_np = y.detach().cpu().numpy()
        uniq = np.unique(y_np)
        for name, act in activations.items():
            if act.grad is None:
                continue
            contrib = torch.abs(act * act.grad).mean(dim=(2, 3)).detach().cpu().numpy()
            c_num = contrib.shape[1]
            if name not in layer_stats:
                layer_stats[name] = {
                    "sum": np.zeros((num_classes, c_num), dtype=np.float64),
                    "sum_sq": np.zeros((num_classes, c_num), dtype=np.float64),
                    "count": np.zeros((num_classes,), dtype=np.float64),
                }
            stat = layer_stats[name]
            for cls in uniq:
                cls = int(cls)
                mask = y_np == cls
                if not np.any(mask):
                    continue
                cls_contrib = contrib[mask]
                stat["sum"][cls] += cls_contrib.sum(axis=0)
                stat["sum_sq"][cls] += np.square(cls_contrib).sum(axis=0)
                stat["count"][cls] += float(mask.sum())

    for h in hooks:
        h.remove()
    model.train(was_training)
    model.zero_grad(set_to_none=True)

    risk_scores = {}
    eps = 1e-8
    for name, stat in layer_stats.items():
        counts = stat["count"]
        valid = counts > 0
        c_num = stat["sum"].shape[1]
        if valid.sum() == 0:
            risk_scores[name] = np.ones(c_num, dtype=np.float64) * 0.5
            continue

        safe_counts = np.maximum(counts[:, None], 1.0)
        mu = stat["sum"] / safe_counts
        var = stat["sum_sq"] / safe_counts - np.square(mu)
        sigma = np.sqrt(np.maximum(var, 1e-10))

        valid_idx = np.where(valid)[0]
        risk = np.zeros(c_num, dtype=np.float64)
        for cls in valid_idx:
            others = [j for j in valid_idx if j != cls]
            if others:
                other_mean = mu[others].mean(axis=0)
            else:
                other_mean = mu[cls]

            u_val = mu[cls] / (other_mean + eps)
            v_val = mu[cls] / (sigma[cls] + eps)
            logits = 1.2 * (u_val - 1.0) + 0.8 * (v_val - 1.0)
            risk_cls = 1.0 / (1.0 + np.exp(-logits))
            risk = np.maximum(risk, risk_cls)

        risk_scores[name] = np.clip(risk, 0.0, 1.0)

    return risk_scores


def compute_gra_chip_v3_scores(
    model,
    dataloader,
    device,
    num_batches=8,
    pruning_ratio=0.7,
    target_layers=None,
    taylor_scores=None,
    fisher_scores=None,
    quality_scores=None,
    rho=0.5,
    use_ca=True,
    enable_boundary=True,
    enable_quality_gate=True,
    tau_base=0.58,
    gate_quality_percentile=70.0,
    min_gate_on_rate=0.40,
    gate_floor_percentile=55.0,
    boundary_low=0.35,
    boundary_high=0.65,
    boundary_low_high_r=0.40,
    boundary_high_high_r=0.60,
    max_swap_ratio=0.10,
    sem_margin=0.08,
    gap_tol=0.06,
    min_keep_channels=4,
    min_keep_ratio=0.10,
):
    """
    GRA-CHIP v3:
      - Anchor: 0.60*CHIP + 0.25*Taylor + 0.15*Fisher
      - Semantic only refines boundary decisions (swap), no global fusion
      - Quality gate disables semantic path on low-quality layers

    Returns:
      scores: layer -> [C] importance
      meta:   global + per-layer diagnostics
    """
    feats, labels = _collect_feature_maps(model, dataloader, device, num_batches)
    if target_layers is None:
        target_layers = list(feats.keys())

    r = float(pruning_ratio)
    if r >= 0.7:
        low_q = float(boundary_low_high_r)
        high_q = float(boundary_high_high_r)
    else:
        low_q = float(boundary_low)
        high_q = float(boundary_high)

    layer_pack = {}
    for name in target_layers:
        feat = feats.get(name)
        if feat is None:
            continue
        c_num = feat.shape[1]
        if c_num < 2:
            layer_pack[name] = {
                "anchor": np.ones(c_num, dtype=np.float64),
                "semantic": np.ones(c_num, dtype=np.float64),
                "quality": 1.0,
                "R": 1.0,
                "U": 0.0,
                "C": 1.0,
            }
            continue

        chip_n = _minmax(compute_chip_ci(feat))
        if use_ca:
            sem_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho=rho))
        else:
            sem_red = _minmax(compute_gra_redundancy(feat, rho=rho))
        sem_imp = 1.0 - sem_red

        if taylor_scores is not None and name in taylor_scores:
            taylor_n = _minmax(taylor_scores[name])
        else:
            taylor_n = chip_n

        if fisher_scores is not None and name in fisher_scores:
            fisher_n = _minmax(fisher_scores[name])
        else:
            fisher_n = chip_n

        anchor = _minmax(0.60 * chip_n + 0.25 * taylor_n + 0.15 * fisher_n)

        if quality_scores is not None and name in quality_scores:
            q_val = float(np.clip(quality_scores[name], 0.0, 1.0))
            r_val, u_val, c_val = 0.5, 0.5, 0.5
        else:
            r_val, u_val, c_val, q_val = _estimate_quality_from_signals(anchor, sem_imp)

        layer_pack[name] = {
            "anchor": anchor,
            "semantic": sem_imp,
            "quality": q_val,
            "R": r_val,
            "U": u_val,
            "C": c_val,
        }

    q_values = [v["quality"] for v in layer_pack.values()]
    if q_values:
        tau_candidate = max(float(tau_base), float(np.percentile(q_values, gate_quality_percentile)))
        gate_on_candidate = float(np.mean(np.asarray(q_values) >= tau_candidate))
        if gate_on_candidate < float(min_gate_on_rate):
            tau_dyn = float(np.percentile(q_values, gate_floor_percentile))
        else:
            tau_dyn = tau_candidate
    else:
        tau_candidate = float(tau_base)
        tau_dyn = float(tau_base)
        gate_on_candidate = 0.0

    scores = {}
    meta = {}
    gate_on_layers = 0
    semantic_layers = 0
    total_swaps = 0

    for name, pack in layer_pack.items():
        base = np.asarray(pack["anchor"], dtype=np.float64)
        sem = np.asarray(pack["semantic"], dtype=np.float64)
        quality = float(pack["quality"])

        c_num = len(base)
        keep_k = int(np.clip(round((1.0 - r) * c_num), 1, c_num))
        keep_floor = max(int(min_keep_channels), int(np.ceil(min_keep_ratio * c_num)))
        keep_k = max(keep_k, min(keep_floor, c_num))

        order = np.argsort(base)[::-1]
        keep_mask = np.zeros(c_num, dtype=bool)
        keep_mask[order[:keep_k]] = True

        gate_on = True
        if enable_quality_gate:
            gate_on = quality >= tau_dyn

        swaps = 0
        if gate_on:
            gate_on_layers += 1
            if enable_boundary:
                # Boundary zone centered at keep/prune threshold.
                band_width = max(2, int(round((high_q - low_q) * c_num)))
                lo = max(0, keep_k - band_width // 2)
                hi = min(c_num, keep_k + band_width // 2 + 1)
                if hi <= lo:
                    hi = min(c_num, lo + 2)
                candidate = order[lo:hi]
            else:
                candidate = order

            kept_candidate = [idx for idx in candidate if keep_mask[idx]]
            pruned_candidate = [idx for idx in candidate if not keep_mask[idx]]
            if kept_candidate and pruned_candidate:
                semantic_layers += 1
                kept_candidate.sort(key=lambda idx: (sem[idx], base[idx]))
                pruned_candidate.sort(key=lambda idx: (-sem[idx], -base[idx]))

                q_conf = float(np.clip((quality - tau_dyn) / max(1.0 - tau_dyn, 1e-8), 0.0, 1.0))
                raw_swaps = int(round(max_swap_ratio * c_num * (1.0 + q_conf)))
                max_swaps = max(2, raw_swaps)
                max_swaps = min(max_swaps, len(kept_candidate), len(pruned_candidate))

                for i in range(max_swaps):
                    out_idx = kept_candidate[i]
                    in_idx = pruned_candidate[i]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if sem_gain >= sem_margin and anchor_gap <= gap_tol:
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps += 1
                # Conservative fallback to avoid semantic path fully inactive.
                if swaps == 0 and max_swaps > 0:
                    out_idx = kept_candidate[0]
                    in_idx = pruned_candidate[0]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if sem_gain >= min(0.03, sem_margin) and anchor_gap <= (gap_tol + 0.02):
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps = 1
        else:
            q_conf = 0.0

        total_swaps += swaps
        out_score = np.array(base, copy=True)
        out_score[keep_mask] += 1.0
        scores[name] = out_score

        meta[name] = {
            "quality": quality,
            "R": float(pack["R"]),
            "U": float(pack["U"]),
            "C": float(pack["C"]),
            "gate_on": bool(gate_on),
            "swaps": int(swaps),
            "q_conf": q_conf,
            "keep_k": int(keep_k),
        }

    n_layers = max(len(scores), 1)
    meta["__global__"] = {
        "version": "GRA-CHIP-v3",
        "method": "GRA-CHIP-v3",
        "anchor_weights": {"chip": 0.60, "taylor": 0.25, "fisher": 0.15},
        "tau_base": float(tau_base),
        "tau_candidate": float(tau_candidate),
        "tau_dyn": float(tau_dyn),
        "gate_quality_percentile": float(gate_quality_percentile),
        "gate_on_rate_candidate": float(gate_on_candidate),
        "min_gate_on_rate": float(min_gate_on_rate),
        "boundary_quantiles": [float(low_q), float(high_q)],
        "gate_on_rate": float(gate_on_layers / n_layers),
        "semantic_applied_rate": float(semantic_layers / n_layers),
        "total_swaps": int(total_swaps),
        "avg_swaps_per_layer": float(total_swaps / n_layers),
        "quality_mean": float(np.mean(q_values)) if q_values else 0.0,
        "use_ca": bool(use_ca),
        "enable_boundary": bool(enable_boundary),
        "enable_quality_gate": bool(enable_quality_gate),
    }
    return scores, meta


def compute_gra_chip_v31_scores(
    model,
    dataloader,
    device,
    architecture="resnet56",
    num_batches=8,
    pruning_ratio=0.7,
    target_layers=None,
    taylor_scores=None,
    fisher_scores=None,
    rho=0.5,
    use_ca=True,
    enable_boundary=True,
    enable_quality_gate=True,
    enable_consensus=True,
    sem_margin=0.05,
    consensus_margin=0.02,
    gap_tol=0.05,
    min_keep_channels=4,
    min_keep_ratio=0.10,
):
    """
    GRA-CHIP v3.1:
      - Same anchor as v3: 0.60*CHIP + 0.25*Taylor + 0.15*Fisher
      - Architecture-aware gate (ResNet/VGG use different thresholds)
      - Boundary-only swap with consensus constraint
    """
    feats, labels = _collect_feature_maps(model, dataloader, device, num_batches)
    if target_layers is None:
        target_layers = list(feats.keys())

    r = float(pruning_ratio)
    low_q, high_q = (0.40, 0.60) if r >= 0.7 else (0.35, 0.65)
    is_vgg = str(architecture).lower().startswith("vgg")

    layer_pack = {}
    for name in target_layers:
        feat = feats.get(name)
        if feat is None:
            continue
        c_num = feat.shape[1]
        if c_num < 2:
            layer_pack[name] = {
                "anchor": np.ones(c_num, dtype=np.float64),
                "semantic": np.ones(c_num, dtype=np.float64),
                "chip": np.ones(c_num, dtype=np.float64),
                "taylor": np.ones(c_num, dtype=np.float64),
                "quality": 1.0,
                "R": 1.0,
                "U": 0.0,
                "C": 1.0,
            }
            continue

        chip_n = _minmax(compute_chip_ci(feat))
        if use_ca:
            sem_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho=rho))
        else:
            sem_red = _minmax(compute_gra_redundancy(feat, rho=rho))
        sem_imp = 1.0 - sem_red

        if taylor_scores is not None and name in taylor_scores:
            taylor_n = _minmax(taylor_scores[name])
        else:
            taylor_n = chip_n
        if fisher_scores is not None and name in fisher_scores:
            fisher_n = _minmax(fisher_scores[name])
        else:
            fisher_n = chip_n

        anchor = _minmax(0.60 * chip_n + 0.25 * taylor_n + 0.15 * fisher_n)
        r_val, u_val, c_val, q_val = _estimate_quality_from_signals(anchor, sem_imp)

        layer_pack[name] = {
            "anchor": anchor,
            "semantic": sem_imp,
            "chip": chip_n,
            "taylor": taylor_n,
            "quality": float(q_val),
            "R": float(r_val),
            "U": float(u_val),
            "C": float(c_val),
        }

    q_values = [v["quality"] for v in layer_pack.values()]
    if q_values:
        if is_vgg:
            tau_candidate = max(0.55, float(np.percentile(q_values, 65.0)))
            gate_on_candidate = float(np.mean(np.asarray(q_values) >= tau_candidate))
            tau_dyn = float(np.percentile(q_values, 50.0)) if gate_on_candidate < 0.45 else tau_candidate
        else:
            tau_candidate = max(0.58, float(np.percentile(q_values, 70.0)))
            gate_on_candidate = float(np.mean(np.asarray(q_values) >= tau_candidate))
            tau_dyn = float(np.percentile(q_values, 55.0)) if gate_on_candidate < 0.40 else tau_candidate
    else:
        tau_candidate = 0.55 if is_vgg else 0.58
        tau_dyn = tau_candidate
        gate_on_candidate = 0.0

    scores = {}
    meta = {}
    gate_on_layers = 0
    semantic_layers = 0
    total_swaps = 0

    for name, pack in layer_pack.items():
        base = np.asarray(pack["anchor"], dtype=np.float64)
        sem = np.asarray(pack["semantic"], dtype=np.float64)
        chip_n = np.asarray(pack["chip"], dtype=np.float64)
        taylor_n = np.asarray(pack["taylor"], dtype=np.float64)
        quality = float(pack["quality"])
        c_num = len(base)

        keep_k = int(np.clip(round((1.0 - r) * c_num), 1, c_num))
        keep_floor = max(int(min_keep_channels), int(np.ceil(min_keep_ratio * c_num)))
        keep_k = max(keep_k, min(keep_floor, c_num))

        order = np.argsort(base)[::-1]
        keep_mask = np.zeros(c_num, dtype=bool)
        keep_mask[order[:keep_k]] = True

        gate_on = True
        if enable_quality_gate:
            gate_on = quality >= tau_dyn

        swaps = 0
        q_conf = 0.0
        if gate_on:
            gate_on_layers += 1
            candidate = order
            if enable_boundary:
                band_width = max(2, int(round((high_q - low_q) * c_num)))
                lo = max(0, keep_k - band_width // 2)
                hi = min(c_num, keep_k + band_width // 2 + 1)
                if hi <= lo:
                    hi = min(c_num, lo + 2)
                candidate = order[lo:hi]

            kept_candidate = [idx for idx in candidate if keep_mask[idx]]
            pruned_candidate = [idx for idx in candidate if not keep_mask[idx]]
            if kept_candidate and pruned_candidate:
                semantic_layers += 1
                kept_candidate.sort(key=lambda idx: (sem[idx], base[idx]))
                pruned_candidate.sort(key=lambda idx: (-sem[idx], -base[idx]))

                q_conf = float(np.clip((quality - tau_dyn) / max(1.0 - tau_dyn, 1e-8), 0.0, 1.0))
                max_swaps = int(round(0.10 * c_num * (1.0 + q_conf)))
                max_swaps = max(2, max_swaps)
                if is_vgg:
                    if abs(r - 0.7) < 1e-6:
                        max_swaps = min(max_swaps, 4)
                    else:
                        max_swaps = min(max_swaps, 6)
                else:
                    max_swaps = min(max_swaps, 10)
                max_swaps = min(max_swaps, len(kept_candidate), len(pruned_candidate))

                for i in range(max_swaps):
                    out_idx = kept_candidate[i]
                    in_idx = pruned_candidate[i]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if enable_consensus:
                        consensus = 0.5 * float(taylor_n[in_idx] - taylor_n[out_idx]) + 0.5 * float(chip_n[in_idx] - chip_n[out_idx])
                    else:
                        consensus = 1.0
                    if sem_gain >= sem_margin and consensus >= consensus_margin and anchor_gap <= gap_tol:
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps += 1

        total_swaps += swaps
        out_score = np.array(base, copy=True)
        out_score[keep_mask] += 1.0
        scores[name] = out_score
        meta[name] = {
            "quality": quality,
            "R": float(pack["R"]),
            "U": float(pack["U"]),
            "C": float(pack["C"]),
            "gate_on": bool(gate_on),
            "swaps": int(swaps),
            "q_conf": float(q_conf),
            "keep_k": int(keep_k),
        }

    n_layers = max(len(scores), 1)
    meta["__global__"] = {
        "version": "GRA-CHIP-v3.1",
        "method": "GRA-CHIP-v3.1",
        "architecture_profile": "vgg" if is_vgg else "resnet",
        "anchor_weights": {"chip": 0.60, "taylor": 0.25, "fisher": 0.15},
        "tau_candidate": float(tau_candidate),
        "tau_dyn": float(tau_dyn),
        "gate_on_rate_candidate": float(gate_on_candidate),
        "boundary_quantiles": [float(low_q), float(high_q)],
        "gate_on_rate": float(gate_on_layers / n_layers),
        "semantic_applied_rate": float(semantic_layers / n_layers),
        "total_swaps": int(total_swaps),
        "avg_swaps_per_layer": float(total_swaps / n_layers),
        "quality_mean": float(np.mean(q_values)) if q_values else 0.0,
        "use_ca": bool(use_ca),
        "enable_boundary": bool(enable_boundary),
        "enable_quality_gate": bool(enable_quality_gate),
        "enable_consensus": bool(enable_consensus),
        "sem_margin": float(sem_margin),
        "consensus_margin": float(consensus_margin),
        "gap_tol": float(gap_tol),
    }
    return scores, meta


def compute_gra_chip_v32_scores(
    model,
    dataloader,
    device,
    architecture="resnet56",
    num_batches=8,
    pruning_ratio=0.7,
    target_layers=None,
    taylor_scores=None,
    fisher_scores=None,
    rho=0.5,
    use_ca=True,
    enable_boundary=True,
    enable_quality_gate=True,
    enable_consensus=True,
    enable_risk=True,
    sem_margin=0.05,
    consensus_margin=0.02,
    gap_tol=0.05,
    min_keep_channels=4,
    min_keep_ratio=0.10,
):
    """
    GRA-CHIP v3.2:
      - Keep v3 anchor (CHIP/Taylor/Fisher)
      - Add class-conditional risk constraints for boundary swaps
      - Protect top-risk channels from being swapped out
    """
    feats, labels = _collect_feature_maps(model, dataloader, device, num_batches)
    if target_layers is None:
        target_layers = list(feats.keys())

    r = float(pruning_ratio)
    low_q, high_q = (0.40, 0.60) if r >= 0.7 else (0.35, 0.65)
    is_vgg = str(architecture).lower().startswith("vgg")

    if is_vgg:
        tau_base = 0.55
        gate_quality_percentile = 65.0
        min_gate_on_rate = 0.45
        gate_floor_percentile = 50.0
        risk_tol = 0.01
        risk_protect_ratio = 0.10
    else:
        tau_base = 0.58
        gate_quality_percentile = 70.0
        min_gate_on_rate = 0.40
        gate_floor_percentile = 55.0
        risk_tol = 0.03
        risk_protect_ratio = 0.05

    risk_scores = {}
    if enable_risk:
        risk_scores = _collect_ccr_risk_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            target_layers=target_layers,
            num_batches=num_batches,
        )

    layer_pack = {}
    for name in target_layers:
        feat = feats.get(name)
        if feat is None:
            continue
        c_num = feat.shape[1]
        if c_num < 2:
            layer_pack[name] = {
                "anchor": np.ones(c_num, dtype=np.float64),
                "semantic": np.ones(c_num, dtype=np.float64),
                "chip": np.ones(c_num, dtype=np.float64),
                "taylor": np.ones(c_num, dtype=np.float64),
                "risk": np.ones(c_num, dtype=np.float64) * 0.5,
                "quality": 1.0,
                "R": 1.0,
                "U": 0.0,
                "C": 1.0,
            }
            continue

        chip_n = _minmax(compute_chip_ci(feat))
        if use_ca:
            sem_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho=rho))
        else:
            sem_red = _minmax(compute_gra_redundancy(feat, rho=rho))
        sem_imp = 1.0 - sem_red

        if taylor_scores is not None and name in taylor_scores:
            taylor_n = _minmax(taylor_scores[name])
        else:
            taylor_n = chip_n
        if fisher_scores is not None and name in fisher_scores:
            fisher_n = _minmax(fisher_scores[name])
        else:
            fisher_n = chip_n
        anchor = _minmax(0.60 * chip_n + 0.25 * taylor_n + 0.15 * fisher_n)
        risk = np.asarray(risk_scores.get(name, np.ones(c_num) * 0.5), dtype=np.float64)

        r_val, u_val, c_val, q_val = _estimate_quality_from_signals(anchor, sem_imp)
        layer_pack[name] = {
            "anchor": anchor,
            "semantic": sem_imp,
            "chip": chip_n,
            "taylor": taylor_n,
            "risk": np.clip(risk, 0.0, 1.0),
            "quality": float(q_val),
            "R": float(r_val),
            "U": float(u_val),
            "C": float(c_val),
        }

    q_values = [v["quality"] for v in layer_pack.values()]
    if q_values:
        tau_candidate = max(tau_base, float(np.percentile(q_values, gate_quality_percentile)))
        gate_on_candidate = float(np.mean(np.asarray(q_values) >= tau_candidate))
        tau_dyn = (
            float(np.percentile(q_values, gate_floor_percentile))
            if gate_on_candidate < min_gate_on_rate
            else tau_candidate
        )
    else:
        tau_candidate = float(tau_base)
        tau_dyn = float(tau_base)
        gate_on_candidate = 0.0

    scores = {}
    meta = {}
    gate_on_layers = 0
    semantic_layers = 0
    total_swaps = 0
    risk_checks = 0
    risk_rejects = 0
    risk_reject_values = []

    for name, pack in layer_pack.items():
        base = np.asarray(pack["anchor"], dtype=np.float64)
        sem = np.asarray(pack["semantic"], dtype=np.float64)
        chip_n = np.asarray(pack["chip"], dtype=np.float64)
        taylor_n = np.asarray(pack["taylor"], dtype=np.float64)
        risk = np.asarray(pack["risk"], dtype=np.float64)
        quality = float(pack["quality"])
        c_num = len(base)

        keep_k = int(np.clip(round((1.0 - r) * c_num), 1, c_num))
        keep_floor = max(int(min_keep_channels), int(np.ceil(min_keep_ratio * c_num)))
        keep_k = max(keep_k, min(keep_floor, c_num))

        order = np.argsort(base)[::-1]
        keep_mask = np.zeros(c_num, dtype=bool)
        keep_mask[order[:keep_k]] = True

        gate_on = True
        if enable_quality_gate:
            gate_on = quality >= tau_dyn

        swaps = 0
        q_conf = 0.0
        protected_out = set()
        if enable_risk:
            p_num = max(1, int(np.ceil(risk_protect_ratio * c_num)))
            protected_out = set(np.argsort(risk)[::-1][:p_num].tolist())

        if gate_on:
            gate_on_layers += 1
            candidate = order
            if enable_boundary:
                band_width = max(2, int(round((high_q - low_q) * c_num)))
                lo = max(0, keep_k - band_width // 2)
                hi = min(c_num, keep_k + band_width // 2 + 1)
                if hi <= lo:
                    hi = min(c_num, lo + 2)
                candidate = order[lo:hi]

            kept_candidate = [idx for idx in candidate if keep_mask[idx]]
            if enable_risk:
                kept_candidate = [idx for idx in kept_candidate if idx not in protected_out]
            pruned_candidate = [idx for idx in candidate if not keep_mask[idx]]

            if kept_candidate and pruned_candidate:
                semantic_layers += 1
                kept_candidate.sort(key=lambda idx: (sem[idx], base[idx]))
                pruned_candidate.sort(key=lambda idx: (-sem[idx], -base[idx]))

                q_conf = float(np.clip((quality - tau_dyn) / max(1.0 - tau_dyn, 1e-8), 0.0, 1.0))
                max_swaps = int(round(0.10 * c_num * (1.0 + q_conf)))
                max_swaps = max(2, max_swaps)
                if is_vgg:
                    max_swaps = min(max_swaps, 4 if abs(r - 0.7) < 1e-6 else 6)
                else:
                    max_swaps = min(max_swaps, 10)
                max_swaps = min(max_swaps, len(kept_candidate), len(pruned_candidate))

                for i in range(max_swaps):
                    out_idx = kept_candidate[i]
                    in_idx = pruned_candidate[i]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if enable_consensus:
                        consensus = 0.5 * float(taylor_n[in_idx] - taylor_n[out_idx]) + 0.5 * float(chip_n[in_idx] - chip_n[out_idx])
                    else:
                        consensus = 1.0

                    allow = sem_gain >= sem_margin and consensus >= consensus_margin and anchor_gap <= gap_tol
                    if enable_risk:
                        risk_checks += 1
                        risk_loss = float(risk[out_idx] - risk[in_idx])
                        if risk_loss > risk_tol:
                            allow = False
                            risk_rejects += 1
                            risk_reject_values.append(risk_loss)

                    if allow:
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps += 1

        total_swaps += swaps
        out_score = np.array(base, copy=True)
        out_score[keep_mask] += 1.0
        scores[name] = out_score
        meta[name] = {
            "quality": quality,
            "R": float(pack["R"]),
            "U": float(pack["U"]),
            "C": float(pack["C"]),
            "gate_on": bool(gate_on),
            "swaps": int(swaps),
            "q_conf": float(q_conf),
            "keep_k": int(keep_k),
            "risk_protected_channels": int(len(protected_out)),
        }

    n_layers = max(len(scores), 1)
    meta["__global__"] = {
        "version": "GRA-CHIP-v3.2",
        "method": "GRA-CHIP-v3.2",
        "architecture_profile": "vgg" if is_vgg else "resnet",
        "anchor_weights": {"chip": 0.60, "taylor": 0.25, "fisher": 0.15},
        "tau_base": float(tau_base),
        "tau_candidate": float(tau_candidate),
        "tau_dyn": float(tau_dyn),
        "gate_quality_percentile": float(gate_quality_percentile),
        "gate_on_rate_candidate": float(gate_on_candidate),
        "min_gate_on_rate": float(min_gate_on_rate),
        "boundary_quantiles": [float(low_q), float(high_q)],
        "gate_on_rate": float(gate_on_layers / n_layers),
        "semantic_applied_rate": float(semantic_layers / n_layers),
        "total_swaps": int(total_swaps),
        "avg_swaps_per_layer": float(total_swaps / n_layers),
        "quality_mean": float(np.mean(q_values)) if q_values else 0.0,
        "risk_guard_rate": float(risk_rejects / max(risk_checks, 1)),
        "risk_protect_ratio": float(risk_protect_ratio),
        "risk_tol": float(risk_tol),
        "avg_risk_loss_rejected": float(np.mean(risk_reject_values)) if risk_reject_values else 0.0,
        "risk_checks": int(risk_checks),
        "risk_rejects": int(risk_rejects),
        "use_ca": bool(use_ca),
        "enable_boundary": bool(enable_boundary),
        "enable_quality_gate": bool(enable_quality_gate),
        "enable_consensus": bool(enable_consensus),
        "enable_risk": bool(enable_risk),
        "sem_margin": float(sem_margin),
        "consensus_margin": float(consensus_margin),
        "gap_tol": float(gap_tol),
    }
    return scores, meta
