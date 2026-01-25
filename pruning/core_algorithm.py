import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# GRA-Fisher v2.0: Structural Pruning via Semantic Alignment
# ============================================================================
#
# Logic:
# 1. Fisher Information (Class-Balanced): Measures gradient sensitivity.
# 2. Gray Relational Analysis (GRA): Measures trend alignment with logits.
# 3. Channel Orthogonality: Penalizes redundant features.
# 4. L1-Norm: Base stability term.
#
# Importance = 0.4*Fisher + 0.2*Ortho + 0.25*GRA + 0.15*L1
# ============================================================================

def compute_gra_final_score(model, dataloader, device, num_batches=10, rho=0.5):
    """
    Computes the fused importance score for GRA-CNN (v2.0).
    
    Args:
        model: PyTorch model (nn.Module)
        dataloader: Training dataloader
        device: 'cuda' or 'cpu'
        num_batches: Number of batches to estimate Fisher/GRA (default: 10)
        rho: Resolution coefficient for GRA (default: 0.5)
        
    Returns:
        final_scores: Dict {layer_name: numpy_array_of_scores}
    """
    model.eval()
    
    # --- 1. Setup Hooks ---
    fisher_accumulator = {}
    class_counts = {}
    activations = {}
    gradients = {}
    hooks = []
    
    def save_act(name):
        def hook(m, i, o): activations[name] = o.detach()
        return hook
    def save_grad(name):
        def hook(m, gi, go): gradients[name] = go[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act(name)))
            hooks.append(module.register_full_backward_hook(save_grad(name)))
            
    # --- 2. Iterate Data (Fisher & Pre-computation) ---
    model.train() # Enable gradients
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print(f"Scanning {num_batches} batches for importance estimation...")
    
    # Buffers for GRA
    all_activations = {name: [] for name in activations}
    all_logits = []
    all_targets = []
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches: break
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Track Class Balance
        for t in targets:
            t_item = t.item()
            class_counts[t_item] = class_counts.get(t_item, 0) + 1
            
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.mean().backward()
        
        # A. Fisher Calculation
        for name in activations:
            if name in gradients:
                act = activations[name]
                grad = gradients[name]
                # Fisher = (g * a)^2
                f_val = (grad * act).pow(2).mean(dim=[2, 3]) 
                
                # Class Weighting
                batch_weights = torch.tensor([1.0/max(class_counts.get(t.item(), 1), 1) for t in targets], device=device)
                f_weighted = (f_val * batch_weights.unsqueeze(1)).sum(dim=0)
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = f_weighted
                else:
                    fisher_accumulator[name] += f_weighted

        # B. Store for GRA/Ortho
        if i < 4: # GRA uses fewer batches to save RAM
             with torch.no_grad():
                for name in activations:
                    all_activations[name].append(activations[name].detach().cpu())
                all_logits.append(outputs.detach().cpu())
                all_targets.append(targets.cpu())

    # Cleanup Hooks
    for h in hooks: h.remove()
    
    # --- 3. Compute Component Scores ---
    
    # Component 1: Orthogonality (Correlation Penalty)
    print("Computing Orthogonality...")
    ortho_scores = {}
    for name in fisher_accumulator:
        if name in all_activations and len(all_activations[name]) > 0:
            # Re-assemble a representative batch
            act_chunk = all_activations[name][0].to(device) # Use first batch
            act_flat = act_chunk.mean(dim=[2, 3]) # [B, C]
            
            # Normalize
            act_std = (act_flat - act_flat.mean(dim=0)) / (act_flat.std(dim=0) + 1e-8)
            corr = torch.matmul(act_std.t(), act_std) / act_std.size(0)
            
            # Redundancy = Sum of off-diagonal correlations
            redundancy = (corr.abs().sum(dim=1) - 1.0) / max(act_flat.size(1) - 1, 1)
            ortho_scores[name] = (1.0 - redundancy).cpu().numpy()
        else:
             ortho_scores[name] = np.zeros_like(fisher_accumulator[name].cpu().numpy())

    # Component 2: GRA (Semantic Alignment)
    print("Computing GRA Semantic Alignment...")
    gra_scores = {}
    if all_logits:
        logits_cat = torch.cat(all_logits, dim=0).to(device)
        targets_cat = torch.cat(all_targets, dim=0).to(device)
        
        # Calculate Margin (Decision Trend)
        correct = logits_cat.gather(1, targets_cat.view(-1, 1)).squeeze()
        mask = torch.ones_like(logits_cat, dtype=torch.bool)
        mask.scatter_(1, targets_cat.view(-1, 1), False)
        max_wrong = logits_cat.masked_fill(~mask, float('-inf')).max(dim=1)[0]
        margin = (correct - max_wrong).detach()
        margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
        
        for name in fisher_accumulator:
             if name in all_activations:
                # Merge cached batches
                act_cat = torch.cat(all_activations[name], dim=0).to(device)
                act_c = act_cat.mean(dim=[2, 3]) # [B_total, C]
                
                c_scores = []
                for c in range(act_c.size(1)):
                    a_channel = act_c[:, c]
                    a_norm = (a_channel - a_channel.min()) / (a_channel.max() - a_channel.min() + 1e-8)
                    
                    # GRA Formula
                    delta = (a_norm - margin_norm).abs()
                    gamma = (delta.min() + rho * delta.max()) / (delta + rho * delta.max() + 1e-8)
                    c_scores.append(gamma.mean().item())
                gra_scores[name] = np.array(c_scores)
    
    # Component 3: L1 (Structural Stability)
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            l1_scores[name] = w.abs().view(w.size(0), -1).sum(dim=1).cpu().numpy()

    # --- 4. Fusion (Stage-Aware Adaptive v3.1) ---
    print("Fusing Scores (Stage-Aware v3.1)...")
    
    def get_stage_adaptive_weights(layer_name):
        """
        Adaptive weights based on network stage (v3.1).
        - Shallow stages: emphasize structural stability (L1, Fisher).
        - Deep stages: emphasize semantic alignment (GRA).
        """
        # Default base weights
        w_fish = 0.40; w_orth = 0.20; w_gra_base = 0.25; w_l1_base = 0.15

        # Determine stage index from layer_name (ResNet convention: 'layer1', 'layer2', ...)
        stage_idx = 0
        if "layer" in layer_name:
            try:
                # Extract first digit after 'layer'
                part = layer_name.split('layer')[1]
                stage_idx = int(part.split('.')[0])
            except:
                stage_idx = 0 
        
        # Stage-wise Adjustment (Non-linear step function)
        # Stage 0 (Conv1): No change
        # Stage 1: +0.00 GRA
        # Stage 2: +0.05 GRA
        # Stage 3: +0.08 GRA
        # Stage 4: +0.10 GRA (Deepest)
        stage_adjustments = {
            0: 0.0,
            1: 0.0,
            2: 0.05,
            3: 0.08,
            4: 0.10
        }
        # Fallback to max stage if index is higher
        adj = stage_adjustments.get(stage_idx, 0.10)

        # Apply adjustment: boost GRA, reduce L1
        w_gra_new = w_gra_base + adj
        w_l1_new  = max(w_l1_base - adj, 0.0)
        
        # Normalize to keep sum=1
        total = w_fish + w_orth + w_gra_new + w_l1_new
        weights = (w_fish/total, w_orth/total, w_gra_new/total, w_l1_new/total)
        
        # Adaptive Rho also follows stage logic
        # Stage 1 -> 0.1 (Sensitive)
        # Stage 4 -> 1.0 (Stable)
        rho = 0.1 + (0.9 * min(stage_idx, 4) / 4.0)
        
        return weights, rho

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    final_scores = {}
    
    for name in fisher_accumulator:
        # Get Stage-Aware Config
        weights, adaptive_rho = get_stage_adaptive_weights(name)
        wf, wo, wg, wl = weights
        
        # 1. Fisher
        f_raw = fisher_accumulator[name].cpu().numpy()
        s_f = norm(f_raw)
        
        # 2. Ortho
        s_o = norm(ortho_scores.get(name, s_f))
        
        # 3. L1
        s_l = norm(l1_scores.get(name, s_f))
        
        # 4. GRA (Re-calc with stage-aware rho)
        if name in all_activations and all_logits:
            logits_cat = torch.cat(all_logits, dim=0).to(device)
            targets_cat = torch.cat(all_targets, dim=0).to(device)
            
            # Re-calc margin (using cached logic if possible, but fast enough to redo)
            correct = logits_cat.gather(1, targets_cat.view(-1, 1)).squeeze()
            mask = torch.ones_like(logits_cat, dtype=torch.bool)
            mask.scatter_(1, targets_cat.view(-1, 1), False)
            max_wrong = logits_cat.masked_fill(~mask, float('-inf')).max(dim=1)[0]
            margin = (correct - max_wrong).detach()
            margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
            
            act_cat = torch.cat(all_activations[name], dim=0).to(device)
            act_c = act_cat.mean(dim=[2, 3])
            
            c_scores = []
            for c in range(act_c.size(1)):
                a_channel = act_c[:, c]
                a_norm = (a_channel - a_channel.min()) / (a_channel.max() - a_channel.min() + 1e-8)
                delta = (a_norm - margin_norm).abs()
                # ADAPTIVE RHO (Stage-Aware)
                gamma = (delta.min() + adaptive_rho * delta.max()) / (delta + adaptive_rho * delta.max() + 1e-8)
                c_scores.append(gamma.mean().item())
            s_g = norm(np.array(c_scores))
        else:
            s_g = s_f # Fallback
            
        # Weighted Sum
        combined = wf * s_f + wo * s_o + wg * s_g + wl * s_l
        final_scores[name] = combined
        
    return final_scores
