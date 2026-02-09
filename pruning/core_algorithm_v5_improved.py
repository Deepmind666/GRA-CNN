from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

GRA_VERSION = "5.0"
_EPS = 1e-8


@dataclass(frozen=True)
class FusionParams:
    fisher: float
    l1: float
    gra: float
    ortho: float
    rho: float
    alpha: float
    reliability: float


def detect_architecture(model: nn.Module) -> Tuple[str, int]:
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
    arch_type, max_stage = detect_architecture(model)
    if arch_type == "resnet":
        if "conv1" in layer_name and "layer" not in layer_name:
            return 0.0
        if "layer1" in layer_name:
            return 1.0 / max_stage
        if "layer2" in layer_name:
            return 2.0 / max_stage
        if "layer3" in layer_name:
            return 3.0 / max_stage
        if "layer4" in layer_name:
            return 4.0 / max_stage
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


def _safe_standardize(x: torch.Tensor, clip: float = 4.0) -> torch.Tensor:
    mu = x.mean()
    std = x.std(unbiased=False)
    z = (x - mu) / (std + _EPS)
    return torch.clamp(z, -clip, clip)


def _safe_minmax01(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin + _EPS)


def smooth_margin_reference(margins: torch.Tensor, clip_z: float = 4.0) -> torch.Tensor:
    z = _safe_standardize(margins, clip=clip_z)
    return torch.sigmoid(z)


def _pearson_scores(acts: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    ref_center = ref - ref.mean()
    acts_center = acts - acts.mean(dim=0, keepdim=True)

    numerator = (acts_center * ref_center.unsqueeze(1)).sum(dim=0)
    denom = torch.sqrt((acts_center.pow(2).sum(dim=0) + _EPS) * (ref_center.pow(2).sum() + _EPS))
    corr = numerator / denom
    return torch.clamp((corr + 1.0) * 0.5, 0.0, 1.0)


def _gra_scores(acts_norm: torch.Tensor, ref_norm: torch.Tensor, rho: float) -> torch.Tensor:
    delta = (acts_norm - ref_norm.unsqueeze(1)).abs()
    d_min = delta.min(dim=0).values
    d_max = delta.max(dim=0).values

    stable = d_max > _EPS
    out = torch.ones(d_max.shape, dtype=acts_norm.dtype, device=acts_norm.device)
    if stable.any():
        d_min_s = d_min[stable]
        d_max_s = d_max[stable]
        delta_s = delta[:, stable]
        numer = d_min_s + rho * d_max_s
        denom = delta_s + rho * d_max_s.unsqueeze(0) + _EPS
        out[stable] = (numer.unsqueeze(0) / denom).mean(dim=0)
    return torch.clamp(out, 0.0, 1.0)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or b.size < 4:
        return 0.5
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.5

    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    if den < _EPS:
        return 0.5
    corr = float(np.dot(ra, rb) / den)
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def compute_semantic_scores(
    activations: torch.Tensor,
    margins: torch.Tensor,
    rho: float,
    alpha: float,
) -> np.ndarray:
    # activations: [N, C], margins: [N]
    ref_raw = _safe_standardize(margins)
    pearson = _pearson_scores(activations, ref_raw)

    acts_norm = _safe_minmax01(activations)
    ref_norm = smooth_margin_reference(margins)
    gra = _gra_scores(acts_norm, ref_norm, rho=rho)

    score = alpha * pearson + (1.0 - alpha) * gra
    return score.detach().cpu().numpy()


def estimate_semantic_reliability(
    activations: torch.Tensor,
    margins: torch.Tensor,
    rho: float,
    alpha: float,
) -> float:
    n = activations.size(0)
    if n < 16:
        return 0.5

    idx_a = torch.arange(0, n, 2)
    idx_b = torch.arange(1, n, 2)
    if idx_a.numel() < 8 or idx_b.numel() < 8:
        return 0.5

    score_a = compute_semantic_scores(activations[idx_a], margins[idx_a], rho=rho, alpha=alpha)
    score_b = compute_semantic_scores(activations[idx_b], margins[idx_b], rho=rho, alpha=alpha)
    return _spearman_corr(score_a, score_b)


def compute_fisher_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 12,
) -> Dict[str, np.ndarray]:
    original_training = model.training
    fisher: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fisher[name] = torch.zeros(module.out_channels, device=device)

    if not fisher:
        return {}

    model.eval()
    num_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        num_samples += images.size(0)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                grad = module.weight.grad
                fisher[name] += (grad.pow(2).view(grad.size(0), -1).sum(dim=1))

    out = {name: (v / max(num_samples, 1)).detach().cpu().numpy() for name, v in fisher.items()}
    model.train(original_training)
    return out


def compute_l1_scores(model: nn.Module) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            score = module.weight.data.abs().view(module.out_channels, -1).sum(dim=1)
            out[name] = score.detach().cpu().numpy()
    return out


def compute_orthogonality_scores(model: nn.Module) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        w = module.weight.data
        c_out = w.size(0)
        if c_out < 2:
            out[name] = np.ones(c_out, dtype=np.float32)
            continue

        flat = w.view(c_out, -1)
        norm = flat / (flat.norm(dim=1, keepdim=True) + _EPS)
        sim = torch.mm(norm, norm.t())
        sim.fill_diagonal_(-1.0)
        max_sim = sim.max(dim=1).values
        ortho = (1.0 - max_sim).clamp(0.0, 1.0)
        out[name] = ortho.detach().cpu().numpy()
    return out


def collect_activations_and_margins(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 12,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    original_training = model.training
    act_data: Dict[str, list[torch.Tensor]] = defaultdict(list)
    all_margins: list[torch.Tensor] = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            if out.dim() == 4:
                pooled = out.mean(dim=(2, 3))
            else:
                pooled = out
            act_data[name].append(pooled.detach().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)

                correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                wrong = logits.clone()
                wrong.scatter_(1, labels.unsqueeze(1), float("-inf"))
                max_wrong = wrong.max(dim=1).values
                margins = correct - max_wrong
                all_margins.append(margins.detach().cpu())
    finally:
        for h in handles:
            h.remove()
        model.train(original_training)

    if not all_margins:
        raise RuntimeError("No samples were collected for activation/margin computation.")

    activations = {name: torch.cat(chunks, dim=0) for name, chunks in act_data.items() if chunks}
    margins = torch.cat(all_margins, dim=0)
    return activations, margins


def normalize_scores_quantile(
    scores: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    if scores.size == 0:
        return scores

    lo = float(np.quantile(scores, q_low))
    hi = float(np.quantile(scores, q_high))
    if hi - lo < _EPS:
        return np.full_like(scores, 0.5, dtype=np.float64)

    clipped = np.clip(scores, lo, hi)
    return (clipped - lo) / (hi - lo + _EPS)


def get_adaptive_fusion_params(
    layer_name: str,
    model: nn.Module,
    pruning_ratio: float,
    reliability: float,
) -> FusionParams:
    if pruning_ratio <= 0.40:
        w_fisher, w_l1, w_gra, w_ortho = 0.26, 0.44, 0.20, 0.10
    elif pruning_ratio <= 0.65:
        w_fisher, w_l1, w_gra, w_ortho = 0.30, 0.36, 0.24, 0.10
    else:
        w_fisher, w_l1, w_gra, w_ortho = 0.33, 0.30, 0.27, 0.10

    depth_ratio = get_layer_depth_ratio(layer_name, model)
    depth_shift = 0.08 * depth_ratio
    w_gra += depth_shift
    w_l1 = max(w_l1 - depth_shift, 0.12)

    # Reliability gate: if semantic score is unstable, gracefully fallback to L1/Fisher.
    gra_before = w_gra
    gra_scale = 0.55 + 0.45 * float(np.clip(reliability, 0.0, 1.0))
    w_gra = w_gra * gra_scale
    backoff = max(gra_before - w_gra, 0.0)
    w_l1 += backoff * 0.65
    w_fisher += backoff * 0.35

    total = max(w_fisher + w_l1 + w_gra + w_ortho, _EPS)
    w_fisher /= total
    w_l1 /= total
    w_gra /= total
    w_ortho /= total

    rho = float(np.clip(0.25 + 0.35 * depth_ratio + 0.15 * (1.0 - reliability), 0.20, 0.80))
    alpha = float(np.clip(0.55 + 0.10 * (pruning_ratio - 0.5), 0.45, 0.70))

    return FusionParams(
        fisher=w_fisher,
        l1=w_l1,
        gra=w_gra,
        ortho=w_ortho,
        rho=rho,
        alpha=alpha,
        reliability=float(np.clip(reliability, 0.0, 1.0)),
    )


def compute_gra_v5_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 12,
    pruning_ratio: float = 0.5,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if verbose:
        print(f"[GRA v{GRA_VERSION}] score collection starts")

    activations, margins = collect_activations_and_margins(
        model=model,
        dataloader=dataloader,
        device=device,
        num_batches=num_batches,
    )
    fisher_scores = compute_fisher_scores(model, dataloader, device, num_batches=num_batches)
    l1_scores = compute_l1_scores(model)
    ortho_scores = compute_orthogonality_scores(model)

    final_scores: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {
        "version": GRA_VERSION,
        "pruning_ratio": float(pruning_ratio),
        "layers": {},
    }

    for layer_name, act in activations.items():
        num_channels = act.size(1)
        if num_channels < 1:
            continue

        # First-pass params for reliability estimation.
        initial_rho = float(np.clip(0.30 + 0.30 * get_layer_depth_ratio(layer_name, model), 0.20, 0.70))
        initial_alpha = 0.55
        reliability = estimate_semantic_reliability(act, margins, initial_rho, initial_alpha)
        params = get_adaptive_fusion_params(layer_name, model, pruning_ratio, reliability)

        semantic = compute_semantic_scores(act, margins, rho=params.rho, alpha=params.alpha)
        fisher = fisher_scores.get(layer_name, np.ones(num_channels, dtype=np.float64))
        l1 = l1_scores.get(layer_name, np.ones(num_channels, dtype=np.float64))
        ortho = ortho_scores.get(layer_name, np.ones(num_channels, dtype=np.float64))

        fisher_n = normalize_scores_quantile(np.asarray(fisher, dtype=np.float64))
        l1_n = normalize_scores_quantile(np.asarray(l1, dtype=np.float64))
        semantic_n = normalize_scores_quantile(np.asarray(semantic, dtype=np.float64))
        ortho_n = normalize_scores_quantile(np.asarray(ortho, dtype=np.float64))

        fused = (
            params.fisher * fisher_n
            + params.l1 * l1_n
            + params.gra * semantic_n
            + params.ortho * ortho_n
        )
        final_scores[layer_name] = fused.astype(np.float32)

        metadata["layers"][layer_name] = {
            "weights": {
                "Fisher": params.fisher,
                "L1": params.l1,
                "GRA": params.gra,
                "Ortho": params.ortho,
            },
            "rho": params.rho,
            "alpha": params.alpha,
            "reliability": params.reliability,
            "channels": int(num_channels),
        }

    if verbose:
        print(f"[GRA v{GRA_VERSION}] done, layers={len(final_scores)}")

    return final_scores, metadata


# Backward-compat alias style for callers
compute_gra_v5_final_score = compute_gra_v5_scores
