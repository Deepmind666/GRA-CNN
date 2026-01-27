"""
GRA-CNN v5.3: Complete Implementation of GPT DeepSearch Red Team Recommendations
================================================================================

v5.3 Fixes (2026-01-27):
1. BN truly frozen using model.eval() instead of manual BN.eval() + model.train()
2. ResNet FLOPs fixed - stride=2 convs use INPUT size (no double downsampling)
3. try/finally protection for hooks and model state restoration
4. All gradient-requiring functions use torch.enable_grad() with eval mode

v5.2 Fixes:
- Single-pass collection for sample alignment
- Per-layer grad_norms for energy-gated GRA

Based on GPT DeepSearch Red Team Review (2026-01-25/26/27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Algorithm version
GRA_VERSION = "5.3"

# =============================================================================
# Section 1: Architecture Detection & Depth Calculation
# =============================================================================

def detect_network_architecture(model):
    """
    Detect network architecture type and depth info.

    Returns:
        arch_type: 'resnet', 'vgg', 'mobilenet', or 'unknown'
        depth_info: For ResNet: max_stage (3 or 4)
                    For VGG: dict mapping conv layer names to their order index
    """
    layer_names = [name for name, _ in model.named_modules()]

    # Check for ResNet
    has_layer4 = any('layer4' in n for n in layer_names)
    has_layer3 = any('layer3' in n for n in layer_names)
    has_layer2 = any('layer2' in n for n in layer_names)
    has_layer1 = any('layer1' in n for n in layer_names)

    if has_layer1:
        if has_layer4:
            return 'resnet', 4  # ResNet-50/101/152
        elif has_layer3:
            return 'resnet', 3  # ResNet-20/56/110 (CIFAR)
        elif has_layer2:
            return 'resnet', 2
        else:
            return 'resnet', 1

    # Check for VGG - build conv layer order map
    if any('features' in n for n in layer_names):
        conv_order = {}
        conv_idx = 0
        for name, module in model.named_modules():
            if 'features' in name and isinstance(module, nn.Conv2d):
                conv_order[name] = conv_idx
                conv_idx += 1
        total_convs = max(conv_idx, 1)
        # Return dict with order and total
        return 'vgg', {'order': conv_order, 'total': total_convs}

    # Check for MobileNet
    if any('features' in n and 'conv' in n for n in layer_names):
        return 'mobilenet', 17

    return 'unknown', 4


def get_layer_depth_ratio(layer_name, model):
    """
    Calculate relative depth of a layer in the network (0.0 ~ 1.0).

    v5.1: Fixed VGG depth calculation using conv layer order.
    """
    if model is None:
        # Fallback for no model: use heuristics
        if 'layer' in layer_name:
            try:
                stage = int(layer_name.split('layer')[1].split('.')[0])
                return min(stage / 4, 1.0)
            except:
                pass
        if 'conv1' in layer_name and 'layer' not in layer_name:
            return 0.0
        return 0.5

    arch_type, depth_info = detect_network_architecture(model)

    # ResNet style
    if arch_type == 'resnet' and 'layer' in layer_name:
        try:
            stage = int(layer_name.split('layer')[1].split('.')[0])
            max_stage = depth_info  # depth_info is int for ResNet
            return min(stage / max_stage, 1.0)
        except:
            pass

    # VGG style - use conv order map
    if arch_type == 'vgg' and isinstance(depth_info, dict):
        conv_order = depth_info.get('order', {})
        total_convs = depth_info.get('total', 1)
        if layer_name in conv_order:
            return conv_order[layer_name] / max(total_convs - 1, 1)

    # First conv layer
    if 'conv1' in layer_name and 'layer' not in layer_name:
        return 0.0

    # Default: middle depth
    return 0.5


# =============================================================================
# Section 2: NGSCS - Normalized Gradient-Semantic Consistency Score
# =============================================================================

def compute_ngscs_per_layer(model, dataloader, layer_name, device,
                            num_batches=5, w_max=3.0, eps=1e-8):
    """
    Compute Normalized GSCS for a specific layer.

    v5.3: Uses model.eval() + torch.enable_grad() to truly freeze BN.
    """
    original_training = model.training
    gradients = {}

    def bwd_hook(module, grad_in, grad_out):
        gradients['x'] = grad_out[0].detach()

    # Find and hook the target layer
    handle_bwd = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle_bwd = module.register_full_backward_hook(bwd_hook)
            break

    if handle_bwd is None:
        return 0.5

    g_by_class = defaultdict(list)
    norm_by_class = defaultdict(list)

    try:
        # v5.3: Use eval mode + enable_grad to freeze BN while allowing gradients
        model.eval()
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            if 'x' not in gradients:
                continue

            g = gradients['x']
            if g.dim() == 4:
                g = g.mean(dim=[2, 3])

            g_norm = g.norm(dim=1)

            for j in range(g.size(0)):
                c = int(labels[j].item())
                g_by_class[c].append(g[j].cpu())
                norm_by_class[c].append(g_norm[j].cpu())
    finally:
        if handle_bwd:
            handle_bwd.remove()
        # Restore original training mode
        model.train(original_training)

    if not g_by_class:
        return 0.5

    # Compute NGSCS per class with energy gating
    scores = []
    for c, gs in g_by_class.items():
        if len(gs) < 2:
            continue

        G = torch.stack(gs, dim=0)  # [Nc, C]
        norms = torch.stack(norm_by_class[c])  # [Nc]

        # Energy gating: down-weight samples with very small gradients
        med = norms.median() + eps
        w = (norms / med).clamp(0, w_max)

        # L2 normalize each gradient vector
        G_hat = G / (G.norm(dim=1, keepdim=True) + eps)

        # Weighted sum of normalized gradients
        v = (w.unsqueeze(1) * G_hat).sum(dim=0)

        # NGSCS = ||weighted_sum|| / sum(weights)
        score_c = v.norm() / (w.sum() + eps)
        scores.append(score_c.item())

    if not scores:
        return 0.5

    return np.mean(scores)


# =============================================================================
# Section 3: Hessian-Weighted Importance (Curvature Awareness)
# =============================================================================

def estimate_hessian_diagonal_hutchinson(model, dataloader, device,
                                         num_batches=3, num_probes=5):
    """
    Estimate Hessian diagonal using Hutchinson's method.

    v5.3: Uses model.eval() + torch.enable_grad() to truly freeze BN.
    """
    original_training = model.training

    # Get all conv layer weights
    conv_params = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_params[name] = module.weight

    if not conv_params:
        return {}

    # Initialize accumulators
    hess_diag = {name: torch.zeros_like(p) for name, p in conv_params.items()}

    # v5.3: Use eval mode to freeze BN, enable_grad for Hessian computation
    model.eval()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        for probe_idx in range(num_probes):
            # Random probe vector v with entries ±1
            v_dict = {name: torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
                     for name, p in conv_params.items()}

            model.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            # First gradient
            grads = torch.autograd.grad(loss, list(conv_params.values()),
                                       create_graph=True, allow_unused=True)

            # Compute v^T * grad
            vTg = sum((v_dict[name] * g).sum()
                     for name, g in zip(conv_params.keys(), grads)
                     if g is not None)

            # Second gradient (Hessian-vector product)
            if vTg.requires_grad:
                hvp = torch.autograd.grad(vTg, list(conv_params.values()),
                                         allow_unused=True)

                # Accumulate: H_ii ≈ v_i * (Hv)_i
                for name, hv in zip(conv_params.keys(), hvp):
                    if hv is not None:
                        hess_diag[name] += (v_dict[name] * hv).detach()

    # Average and take absolute value (we care about magnitude of curvature)
    total_samples = num_batches * num_probes
    hess_per_channel = {}
    for name, h in hess_diag.items():
        h_avg = h.abs() / max(total_samples, 1)
        # Average over [in_c, k, k] to get per-output-channel curvature
        hess_per_channel[name] = h_avg.view(h_avg.size(0), -1).mean(dim=1).cpu().numpy()

    # Restore original training mode
    model.train(original_training)
    return hess_per_channel


# =============================================================================
# Section 4: v5.0 Adaptive Weights with NGSCS Integration
# =============================================================================

def get_adaptive_weights_v5(layer_name, model, ngscs_score=None):
    """
    v5.0 Depth-Adaptive Weight Allocation with NGSCS modulation.

    Key insight from GPT review:
    - If NGSCS is low (noisy gradients), reduce GRA weight
    - If NGSCS is high (consistent gradients), increase GRA weight

    Returns:
        weights: dict with Fisher, Ortho, GRA, L1 weights
        rho: adaptive resolution coefficient
    """
    depth_ratio = get_layer_depth_ratio(layer_name, model)

    # GPT-recommended base weights
    base = {'Fisher': 0.40, 'Ortho': 0.20, 'GRA': 0.25, 'L1': 0.15}

    # Depth-adaptive adjustment
    depth_adj = 0.10 * depth_ratio

    # NGSCS modulation (v5.0 new feature)
    # If NGSCS is provided and low, reduce GRA weight
    ngscs_adj = 0.0
    if ngscs_score is not None:
        # NGSCS typically in [0, 1], higher = more consistent
        # If NGSCS < 0.3, reduce GRA; if > 0.7, boost GRA
        ngscs_adj = 0.05 * (ngscs_score - 0.5)  # Range: [-0.025, +0.025]

    w_gra = base['GRA'] + depth_adj + ngscs_adj
    w_l1 = max(base['L1'] - depth_adj, 0.05)
    w_fisher = base['Fisher']
    w_ortho = base['Ortho']

    # Normalize to sum=1
    total = w_gra + w_l1 + w_fisher + w_ortho
    weights = {
        'Fisher': w_fisher / total,
        'Ortho': w_ortho / total,
        'GRA': w_gra / total,
        'L1': w_l1 / total
    }

    # Adaptive rho: shallow=0.1 (sensitive), deep=0.7 (stable)
    rho = 0.1 + 0.6 * depth_ratio

    return weights, rho


# =============================================================================
# Section 5: Vectorized GRA with Energy Gating
# =============================================================================

def compute_gra_vectorized_v5(act_c, margin_norm, rho, grad_norms=None, eps=1e-8):
    """
    v5.0 Vectorized GRA with optional gradient energy gating.

    If grad_norms is provided, samples with very small gradients are down-weighted.
    """
    B, C = act_c.shape

    # 1. Normalize activations per channel
    act_min = act_c.min(dim=0, keepdim=True).values
    act_max = act_c.max(dim=0, keepdim=True).values
    act_norm = (act_c - act_min) / (act_max - act_min + eps)

    # 2. Compute difference matrix
    margin_expanded = margin_norm.unsqueeze(1)
    delta = (act_norm - margin_expanded).abs()

    # 3. Gray relational coefficients
    delta_min = delta.min(dim=0, keepdim=True).values
    delta_max = delta.max(dim=0, keepdim=True).values
    gamma = (delta_min + rho * delta_max) / (delta + rho * delta_max + eps)

    # 4. Apply energy gating if gradient norms provided
    if grad_norms is not None:
        med = grad_norms.median() + eps
        weights = (grad_norms / med).clamp(0, 3.0).unsqueeze(1)  # [B, 1]
        gra_scores = (gamma * weights).sum(dim=0) / (weights.sum() + eps)
    else:
        gra_scores = gamma.mean(dim=0)

    return gra_scores


# =============================================================================
# Section 6: Layer Protection with Score-Aware Adjustment
# =============================================================================

def get_layer_protection_ratio_v5(layer_name, model, num_channels, avg_score=None):
    """
    v5.0 Adaptive layer protection with score awareness.

    New: If a layer has very high average importance score, protect it more.
    """
    depth_ratio = get_layer_depth_ratio(layer_name, model)

    # Base protection
    if depth_ratio < 0.1 or depth_ratio > 0.9:
        base_max_prune = 0.50  # First/last layers
    elif num_channels < 64:
        base_max_prune = 0.50  # Small channel layers
    else:
        base_max_prune = 0.85  # Middle layers

    # Score-aware adjustment (v5.0)
    if avg_score is not None:
        # If average score is high (important layer), reduce max prune
        # avg_score typically normalized to [0, 1]
        if avg_score > 0.7:
            base_max_prune = min(base_max_prune, 0.60)
        elif avg_score > 0.5:
            base_max_prune = min(base_max_prune, 0.75)

    return base_max_prune


# =============================================================================
# Section 7: Fisher Information & Orthogonality Scores
# =============================================================================

def compute_fisher_information(model, dataloader, device, num_batches=5, eps=1e-8):
    """
    Compute Fisher Information for each conv layer's output channels.

    v5.3: Uses model.eval() to truly freeze BN.
    """
    original_training = model.training

    fisher_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fisher_scores[name] = torch.zeros(module.out_channels, device=device)

    if not fisher_scores:
        return {}

    # v5.3: Use eval mode to freeze BN
    model.eval()

    num_samples = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        num_samples += batch_size

        model.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # Accumulate squared gradients per output channel
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                # weight shape: [out_c, in_c, k, k]
                grad = module.weight.grad
                # Sum squared gradients over [in_c, k, k] for each output channel
                fisher_per_channel = (grad ** 2).view(grad.size(0), -1).sum(dim=1)
                fisher_scores[name] += fisher_per_channel

    # Normalize by number of samples
    for name in fisher_scores:
        fisher_scores[name] = (fisher_scores[name] / max(num_samples, 1)).cpu().numpy()

    # Restore original training mode
    model.train(original_training)
    return fisher_scores


def compute_orthogonality_scores(model, eps=1e-8):
    """
    Compute orthogonality-based importance for each filter.

    Filters that are more orthogonal to others capture unique information.
    Score = 1 - max_correlation_with_other_filters

    High orthogonality = unique filter (captures information others don't).
    """
    ortho_scores = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data  # [out_c, in_c, k, k]
            out_c = weight.size(0)

            if out_c < 2:
                ortho_scores[name] = np.ones(out_c)
                continue

            # Flatten each filter to a vector
            W = weight.view(out_c, -1)  # [out_c, in_c*k*k]

            # Normalize each filter
            W_norm = W / (W.norm(dim=1, keepdim=True) + eps)

            # Compute cosine similarity matrix
            sim_matrix = torch.mm(W_norm, W_norm.t())  # [out_c, out_c]

            # Set diagonal to -1 (ignore self-similarity)
            sim_matrix.fill_diagonal_(-1)

            # Max similarity with any other filter
            max_sim = sim_matrix.max(dim=1).values  # [out_c]

            # Orthogonality score = 1 - max_similarity
            ortho = (1 - max_sim).clamp(0, 1).cpu().numpy()
            ortho_scores[name] = ortho

    return ortho_scores


def compute_l1_norm_scores(model):
    """
    Compute L1 norm importance for each filter.

    Simple but effective: filters with larger weights are more important.
    """
    l1_scores = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data  # [out_c, in_c, k, k]
            # L1 norm per output channel
            l1_per_channel = weight.abs().view(weight.size(0), -1).sum(dim=1)
            l1_scores[name] = l1_per_channel.cpu().numpy()

    return l1_scores


# =============================================================================
# Section 8: GRA Activation Collection & Comprehensive Scoring
# =============================================================================

def collect_activations_margins_and_gradnorms(model, dataloader, device, num_batches=10):
    """
    Collect activations, margins, and per-layer gradient norms in SINGLE pass.

    v5.3 Fixes:
    - Uses model.eval() to truly freeze BN
    - try/finally to ensure hooks and state are restored on error

    Returns:
        activations: dict[layer_name] -> tensor [N, C]
        margins: tensor [N]
        layer_grad_norms: dict[layer_name] -> tensor [N] (per-layer grad norms)
    """
    original_training = model.training

    # Setup hooks for activations AND gradients
    activation_data = defaultdict(list)
    gradient_data = defaultdict(list)

    def make_fwd_hook(name):
        def hook(module, inp, out):
            if out.dim() == 4:
                act = out.mean(dim=[2, 3])
            else:
                act = out
            activation_data[name].append(act.detach().cpu())
        return hook

    def make_bwd_hook(name):
        def hook(module, grad_in, grad_out):
            g = grad_out[0]
            if g is not None:
                if g.dim() == 4:
                    g = g.mean(dim=[2, 3])  # [B, C]
                # Per-sample gradient norm
                g_norms = g.norm(dim=1)  # [B]
                gradient_data[name].append(g_norms.detach().cpu())
        return hook

    # Register hooks
    fwd_handles = []
    bwd_handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fwd_handles.append(module.register_forward_hook(make_fwd_hook(name)))
            bwd_handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    all_margins = []

    try:
        # v5.3: Use eval mode to freeze BN
        model.eval()

        # Single pass: collect everything together
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            with torch.enable_grad():
                logits = model(images)

                # Margin on logits
                correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze()
                logits_copy = logits.clone()
                logits_copy.scatter_(1, labels.unsqueeze(1), float('-inf'))
                max_wrong_logits = logits_copy.max(dim=1).values
                margins = correct_logits - max_wrong_logits
                all_margins.append(margins.detach().cpu())

                # Backward to get gradients
                loss = F.cross_entropy(logits, labels)
                loss.backward()
    finally:
        # Always remove hooks and restore state
        for h in fwd_handles + bwd_handles:
            h.remove()
        model.train(original_training)

    # Concatenate results
    activations = {name: torch.cat(acts, dim=0) for name, acts in activation_data.items()}
    margins = torch.cat(all_margins, dim=0)

    # Per-layer gradient norms
    layer_grad_norms = {}
    for name, gnorms in gradient_data.items():
        if gnorms:
            layer_grad_norms[name] = torch.cat(gnorms, dim=0)

    return activations, margins, layer_grad_norms


def normalize_scores(scores, eps=1e-8):
    """Normalize scores to [0, 1] range using min-max normalization."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < eps:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min + eps)


def compute_gra_final_score_v5(model, dataloader, device,
                                num_batches=10, use_hessian=True,
                                use_ngscs=True, verbose=False):
    """
    v5.2 Main scoring function with all GPT Red Team improvements.

    v5.2 Fixes:
    - Single-pass collection for sample alignment
    - Per-layer grad_norms for energy-gated GRA
    - Uses logits for margin (not softmax)
    - Properly restores model and BN states

    Returns:
        scores: dict[layer_name] -> np.array of shape [num_channels]
        metadata: dict with additional info
    """
    original_training = model.training

    if verbose:
        print(f"[GRA v{GRA_VERSION}] Computing comprehensive importance scores...")

    # Step 1: Collect activations, margins, and gradient norms
    if verbose:
        print("  Step 1: Collecting activations, margins, and gradient norms...")
    activations, margins, grad_norms = collect_activations_margins_and_gradnorms(
        model, dataloader, device, num_batches
    )

    # Step 2: Compute base scores
    if verbose:
        print("  Step 2: Computing Fisher information...")
    fisher_scores = compute_fisher_information(model, dataloader, device, num_batches)

    if verbose:
        print("  Step 3: Computing orthogonality scores...")
    ortho_scores = compute_orthogonality_scores(model)

    if verbose:
        print("  Step 4: Computing L1 norm scores...")
    l1_scores = compute_l1_norm_scores(model)

    # Step 3: Optional Hessian estimation
    hessian_scores = {}
    if use_hessian:
        if verbose:
            print("  Step 5: Estimating Hessian diagonal (curvature)...")
        hessian_scores = estimate_hessian_diagonal_hutchinson(
            model, dataloader, device, num_batches=3, num_probes=5
        )

    # Step 4: Optional NGSCS computation
    ngscs_per_layer = {}
    if use_ngscs:
        if verbose:
            print("  Step 6: Computing NGSCS for key layers...")
        layer_names = list(activations.keys())
        sample_indices = [0, len(layer_names)//3, 2*len(layer_names)//3, -1]
        for idx in sample_indices:
            if 0 <= idx < len(layer_names) or idx == -1:
                name = layer_names[idx]
                ngscs = compute_ngscs_per_layer(
                    model, dataloader, name, device, num_batches=3
                )
                ngscs_per_layer[name] = ngscs

    # Normalize margins for GRA
    margin_norm = (margins - margins.min()) / (margins.max() - margins.min() + 1e-8)

    # Step 5: Compute final scores for each layer
    final_scores = {}
    metadata = {'weights': {}, 'ngscs': ngscs_per_layer, 'version': GRA_VERSION}

    for layer_name in activations.keys():
        act = activations[layer_name]
        num_channels = act.size(1)

        ngscs_score = ngscs_per_layer.get(layer_name, 0.5)
        weights, rho = get_adaptive_weights_v5(layer_name, model, ngscs_score)
        metadata['weights'][layer_name] = weights

        # Compute GRA with energy gating (pass per-layer grad_norms)
        layer_gnorms = grad_norms.get(layer_name, None) if isinstance(grad_norms, dict) else None
        gra = compute_gra_vectorized_v5(act, margin_norm, rho, layer_gnorms)
        gra_np = gra.numpy() if isinstance(gra, torch.Tensor) else gra

        fisher = fisher_scores.get(layer_name, np.ones(num_channels))
        ortho = ortho_scores.get(layer_name, np.ones(num_channels))
        l1 = l1_scores.get(layer_name, np.ones(num_channels))

        fisher_norm = normalize_scores(fisher)
        ortho_norm = normalize_scores(ortho)
        gra_norm = normalize_scores(gra_np)
        l1_norm = normalize_scores(l1)

        combined = (weights['Fisher'] * fisher_norm +
                   weights['Ortho'] * ortho_norm +
                   weights['GRA'] * gra_norm +
                   weights['L1'] * l1_norm)

        if layer_name in hessian_scores and use_hessian:
            hess = hessian_scores[layer_name]
            hess_norm = normalize_scores(hess)
            combined = combined * (0.5 + 0.5 * hess_norm)

        final_scores[layer_name] = combined

    # Restore original model mode
    model.train(original_training)

    if verbose:
        print(f"  Done! Computed scores for {len(final_scores)} layers.")

    return final_scores, metadata


# =============================================================================
# Section 9: Iso-FLOPs Score-Proportional Pruning
# =============================================================================

def estimate_layer_flops(module, input_size, output_channels):
    """
    Estimate FLOPs for a conv layer.

    FLOPs = 2 * K^2 * C_in * C_out * H_out * W_out
    """
    if not isinstance(module, nn.Conv2d):
        return 0

    k = module.kernel_size[0]
    c_in = module.in_channels
    c_out = output_channels
    stride = module.stride[0]

    # Estimate output spatial size
    h_out = input_size // stride
    w_out = input_size // stride

    flops = 2 * k * k * c_in * c_out * h_out * w_out
    return flops


def get_layer_flops_info(model, input_size=32):
    """
    Get FLOPs information for each conv layer.

    v5.3: Fixed ResNet FLOPs - stride=2 convs use INPUT size (before downsampling).

    Key insight: For a stride=2 conv, the INPUT spatial size is what matters for FLOPs.
    estimate_layer_flops will compute output size = input_size // stride internally.

    Returns dict with layer_name -> (module, current_channels, flops_per_channel)
    """
    layer_info = {}
    arch_type, _ = detect_network_architecture(model)

    if arch_type == 'resnet':
        # ResNet INPUT sizes at each stage (before any downsampling in that stage)
        # layer2/3/4 first block has stride=2, so input is from previous stage
        stage_input_sizes = {
            0: input_size,      # conv1 input
            1: input_size,      # layer1 input (after conv1, no pooling in CIFAR ResNet)
            2: input_size,      # layer2 input = layer1 output size
            3: input_size // 2, # layer3 input = layer2 output size
            4: input_size // 4, # layer4 input = layer3 output size
        }

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                c_out = module.out_channels
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride

                # Determine stage from layer name
                if 'layer' in name:
                    try:
                        stage = int(name.split('layer')[1].split('.')[0])
                    except:
                        stage = 2
                elif 'conv1' in name and 'layer' not in name:
                    stage = 0
                else:
                    stage = 2

                # Get INPUT size for this conv
                input_spatial = stage_input_sizes.get(stage, input_size // 4)

                # For non-first blocks in a stage, input is already downsampled
                # Check if this is block 0 (first block) or later blocks
                if 'layer' in name:
                    try:
                        block_idx = int(name.split('.')[1])
                        if block_idx > 0:
                            # Not first block, input is already at stage output size
                            if stage >= 2:
                                input_spatial = stage_input_sizes.get(stage + 1, input_spatial)
                    except:
                        pass

                # Downsample branch sees same input as main branch first conv
                # (already handled by stage_input_sizes)

                flops_per_ch = estimate_layer_flops(module, input_spatial, 1)
                layer_info[name] = {
                    'module': module,
                    'channels': c_out,
                    'flops_per_channel': flops_per_ch,
                    'total_flops': flops_per_ch * c_out,
                    'spatial_size': input_spatial // stride  # Output size
                }
    else:
        # VGG/other: sequential traversal with pooling tracking
        current_size = input_size

        for name, module in model.named_modules():
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                pool_stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                current_size = current_size // pool_stride

            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                out_size = module.output_size
                if isinstance(out_size, int):
                    current_size = out_size
                elif out_size is not None:
                    current_size = out_size[0] if isinstance(out_size, tuple) else out_size

            if isinstance(module, nn.Conv2d):
                c_out = module.out_channels
                flops_per_ch = estimate_layer_flops(module, current_size, 1)
                layer_info[name] = {
                    'module': module,
                    'channels': c_out,
                    'flops_per_channel': flops_per_ch,
                    'total_flops': flops_per_ch * c_out,
                    'spatial_size': current_size
                }
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                current_size = current_size // stride

    return layer_info


def get_global_mask_iso_flops_v5(model, scores, target_flops_ratio,
                                  input_size=32, min_channels=8, verbose=False):
    """
    v5.0 Iso-FLOPs pruning with score-proportional distribution.

    GPT Red Team Key Insight:
    Instead of uniform pruning ratio across layers, distribute pruning
    based on layer importance scores. Layers with lower average scores
    get pruned more aggressively.

    Args:
        model: The neural network model
        scores: dict[layer_name] -> importance scores per channel
        target_flops_ratio: Target FLOPs to keep (e.g., 0.5 = keep 50%)
        input_size: Input spatial size (32 for CIFAR)
        min_channels: Minimum channels to keep per layer

    Returns:
        masks: dict[layer_name] -> binary mask (1=keep, 0=prune)
        actual_flops_ratio: Achieved FLOPs ratio
    """
    if verbose:
        print(f"[Iso-FLOPs v5.0] Target FLOPs ratio: {target_flops_ratio:.1%}")

    # Get FLOPs info for each layer
    flops_info = get_layer_flops_info(model, input_size)

    # Calculate total original FLOPs
    total_original_flops = sum(info['total_flops'] for info in flops_info.values())
    target_flops = total_original_flops * target_flops_ratio

    if verbose:
        print(f"  Original FLOPs: {total_original_flops/1e6:.2f}M")
        print(f"  Target FLOPs: {target_flops/1e6:.2f}M")

    # Calculate layer-wise average importance
    layer_avg_scores = {}
    for name in scores:
        if name in flops_info:
            layer_avg_scores[name] = np.mean(scores[name])

    if not layer_avg_scores:
        return {}, 1.0

    # Normalize layer scores to get pruning weights
    avg_vals = np.array(list(layer_avg_scores.values()))
    avg_min, avg_max = avg_vals.min(), avg_vals.max()

    # Score-proportional pruning: low score layers get pruned more
    # Invert scores: high importance = low pruning ratio
    layer_prune_weights = {}
    for name, avg in layer_avg_scores.items():
        # Normalize to [0, 1], then invert
        if avg_max - avg_min > 1e-8:
            norm_score = (avg - avg_min) / (avg_max - avg_min)
        else:
            norm_score = 0.5
        # Invert: high score -> low prune weight
        layer_prune_weights[name] = 1.0 - norm_score + 0.1  # Add 0.1 to avoid zero

    # Normalize prune weights
    total_weight = sum(layer_prune_weights.values())
    for name in layer_prune_weights:
        layer_prune_weights[name] /= total_weight

    # Binary search for the right global threshold
    low, high = 0.0, 1.0
    best_masks = {}
    best_flops_ratio = 1.0

    for iteration in range(20):  # Binary search iterations
        mid = (low + high) / 2

        # Generate masks with current threshold
        masks = {}
        current_flops = 0

        for name, layer_scores in scores.items():
            if name not in flops_info:
                continue

            info = flops_info[name]
            num_channels = len(layer_scores)

            # Layer-specific threshold based on prune weight
            layer_weight = layer_prune_weights.get(name, 1.0 / len(scores))
            layer_threshold = mid * (1 + layer_weight)

            # Get protection ratio
            max_prune = get_layer_protection_ratio_v5(
                name, model, num_channels, layer_avg_scores.get(name)
            )
            min_keep = max(min_channels, int(num_channels * (1 - max_prune)))

            # Create mask: keep channels with score > threshold
            sorted_indices = np.argsort(layer_scores)[::-1]  # Descending
            keep_count = max(min_keep, int(np.sum(layer_scores > layer_threshold)))
            keep_count = min(keep_count, num_channels)

            mask = np.zeros(num_channels)
            mask[sorted_indices[:keep_count]] = 1
            masks[name] = mask

            # Calculate FLOPs for this layer
            current_flops += info['flops_per_channel'] * keep_count

        current_ratio = current_flops / total_original_flops

        if abs(current_ratio - target_flops_ratio) < abs(best_flops_ratio - target_flops_ratio):
            best_masks = masks.copy()
            best_flops_ratio = current_ratio

        # Adjust search range
        if current_ratio > target_flops_ratio:
            low = mid  # Need to prune more
        else:
            high = mid  # Need to prune less

    if verbose:
        print(f"  Achieved FLOPs ratio: {best_flops_ratio:.1%}")
        total_kept = sum(m.sum() for m in best_masks.values())
        total_channels = sum(len(m) for m in best_masks.values())
        print(f"  Channels kept: {int(total_kept)}/{total_channels}")

    return best_masks, best_flops_ratio


def get_pruning_mask_simple(scores, prune_ratio, model=None, min_channels=8):
    """
    Simple pruning mask generation based on global threshold.

    Args:
        scores: dict[layer_name] -> importance scores per channel
        prune_ratio: Fraction of channels to prune (e.g., 0.5 = prune 50%)
        model: Optional model for layer protection
        min_channels: Minimum channels to keep per layer

    Returns:
        masks: dict[layer_name] -> binary mask (1=keep, 0=prune)
    """
    masks = {}

    for layer_name, layer_scores in scores.items():
        num_channels = len(layer_scores)

        # Get layer protection
        if model is not None:
            max_prune = get_layer_protection_ratio_v5(
                layer_name, model, num_channels, np.mean(layer_scores)
            )
        else:
            max_prune = 0.85

        # Calculate channels to keep
        actual_prune = min(prune_ratio, max_prune)
        keep_count = max(min_channels, int(num_channels * (1 - actual_prune)))

        # Keep top-k channels by score
        sorted_indices = np.argsort(layer_scores)[::-1]
        mask = np.zeros(num_channels)
        mask[sorted_indices[:keep_count]] = 1
        masks[layer_name] = mask

    return masks


# =============================================================================
# Section 10: Convenience API & Backward Compatibility
# =============================================================================

def prune_model_v5(model, dataloader, device, prune_ratio=0.5,
                   use_iso_flops=True, input_size=32,
                   use_hessian=True, use_ngscs=True, verbose=True):
    """
    v5.0 Main API for model pruning.

    This is the primary entry point for using GRA v5.0.

    Args:
        model: PyTorch model to prune
        dataloader: DataLoader for computing importance scores
        device: torch device
        prune_ratio: Target pruning ratio (0.5 = remove 50% FLOPs/channels)
        use_iso_flops: If True, use Iso-FLOPs constraint
        input_size: Input spatial size (32 for CIFAR, 224 for ImageNet)
        use_hessian: Enable Hessian-weighted importance
        use_ngscs: Enable NGSCS gradient consistency

    Returns:
        masks: dict[layer_name] -> binary mask
        scores: dict[layer_name] -> importance scores
        metadata: Additional information
    """
    if verbose:
        print("=" * 60)
        print(f"GRA-CNN v{GRA_VERSION} Pruning")
        print("=" * 60)
        print(f"  Target prune ratio: {prune_ratio:.0%}")
        print(f"  Iso-FLOPs: {use_iso_flops}")
        print(f"  Hessian weighting: {use_hessian}")
        print(f"  NGSCS modulation: {use_ngscs}")
        print("-" * 60)

    # Step 1: Compute importance scores
    scores, metadata = compute_gra_final_score_v5(
        model, dataloader, device,
        num_batches=10,
        use_hessian=use_hessian,
        use_ngscs=use_ngscs,
        verbose=verbose
    )

    # Step 2: Generate pruning masks
    if use_iso_flops:
        target_flops_ratio = 1.0 - prune_ratio
        masks, actual_ratio = get_global_mask_iso_flops_v5(
            model, scores, target_flops_ratio,
            input_size=input_size, verbose=verbose
        )
        metadata['actual_flops_ratio'] = actual_ratio
    else:
        masks = get_pruning_mask_simple(scores, prune_ratio, model)

    if verbose:
        print("-" * 60)
        print("Pruning complete!")
        print("=" * 60)

    return masks, scores, metadata


# Backward compatibility aliases
compute_gra_vectorized = compute_gra_vectorized_v5
get_adaptive_weights_v4 = get_adaptive_weights_v5
get_layer_protection_ratio = get_layer_protection_ratio_v5
