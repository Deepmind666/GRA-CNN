"""
GRA-Hybrid: GRA与L1融合的混合评分方法
====================================

策略: 结合GRA的语义对齐能力和L1的稳定性
公式: score = α × GRA_score + (1-α) × L1_score (归一化后)

理论依据:
- L1在大多数场景表现稳定
- GRA在某些语义关键通道上有独特视角
- 融合可以取长补短
"""

import torch
import torch.nn as nn
import numpy as np

def compute_gra_hybrid_scores(model, dataloader, device, num_batches=10, rho=0.5, alpha=0.3):
    """
    GRA-Hybrid: GRA与L1融合的混合评分
    
    Args:
        alpha: GRA权重 (0-1), 1-alpha为L1权重
               建议0.2-0.4，让L1为主导但引入GRA的语义信息
    
    Returns:
        dict: {layer_name: hybrid_scores}
    """
    model.eval()
    
    # ============ 计算L1分数 ============
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            score = weight.abs().view(weight.size(0), -1).sum(dim=1)
            l1_scores[name] = score.cpu().numpy()
    
    # ============ 计算GRA分数 ============
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            act = output.mean(dim=[2, 3]).detach()
            activations[name].append(act)
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    all_logits, all_targets = [], []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.detach())
            all_targets.append(targets.detach())
    
    for h in hooks:
        h.remove()
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    ref_logits = logits.gather(1, targets.view(-1, 1)).squeeze()
    ref_min, ref_max = ref_logits.min(), ref_logits.max()
    ref_norm = (ref_logits - ref_min) / (ref_max - ref_min + 1e-8)
    
    gra_scores = {}
    for name, act_list in activations.items():
        acts = torch.cat(act_list, dim=0)
        C = acts.size(1)
        
        channel_scores = []
        for c in range(C):
            act_c = acts[:, c]
            act_min, act_max = act_c.min(), act_c.max()
            act_norm = (act_c - act_min) / (act_max - act_min + 1e-8)
            
            delta = (act_norm - ref_norm).abs()
            delta_min, delta_max = delta.min(), delta.max()
            gamma = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
            channel_scores.append(gamma.mean().item())
        
        gra_scores[name] = np.array(channel_scores)
    
    # ============ 融合: 归一化后加权 ============
    hybrid_scores = {}
    for name in l1_scores:
        if name in gra_scores:
            # Min-Max归一化到[0,1]
            l1 = l1_scores[name]
            gra = gra_scores[name]
            
            l1_norm = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-8)
            gra_norm = (gra - gra.min()) / (gra.max() - gra.min() + 1e-8)
            
            # 加权融合
            hybrid = alpha * gra_norm + (1 - alpha) * l1_norm
            hybrid_scores[name] = hybrid
    
    return hybrid_scores


if __name__ == "__main__":
    print("GRA-Hybrid: GRA与L1融合评分")
    print("建议 alpha=0.3 (L1为主导)")
