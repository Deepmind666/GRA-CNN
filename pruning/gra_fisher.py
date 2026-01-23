"""
GRA-Fisher: 基于 Fisher 信息量的语义剪枝
==========================================

理论基础:
- Fisher 信息量测量参数对任务的信息贡献
- Fisher(θ) = E[(∂log p(y|x,θ)/∂θ)²]
- 对于通道重要性：Fisher(c) ≈ E[gradient(c)²]

设计原理:
- Fisher_score (50%): 理论最优的重要性度量
- GRA_semantic (30%): 与分类 margin 的语义对齐
- L1_stability (20%): 权重范数提供稳定性

为什么这样设计:
1. Fisher 主导：直接测量"移除后损失增量"
2. GRA 保留：维持论文核心创新点
3. L1 兜底：防止 Fisher 计算不稳定
"""

import torch
import torch.nn as nn
import numpy as np

def compute_gra_fisher_scores(model, dataloader, device, num_batches=10):
    """
    GRA-Fisher: 基于 Fisher 信息量的语义剪枝
    
    融合3种评估维度:
    1. Fisher 信息量 (50%): E[gradient²]，理论最优
    2. GRA 语义 (30%): 与分类 margin 的对齐
    3. L1 稳定 (20%): 权重绝对值和
    
    Returns:
        dict: {layer_name: importance_scores}
    """
    model.eval()
    
    # ============ 1. Fisher 信息量 ============
    # Fisher(c) = E[(∂L/∂a_c)² × a_c²] ≈ E[gradient² × activation²]
    
    fisher_scores = {}
    activations_fisher = {}
    gradients_fisher = {}
    hooks = []
    
    def save_act_fisher(name):
        def hook(module, input, output):
            activations_fisher[name] = output.detach()
        return hook
    
    def save_grad_fisher(name):
        def hook(module, grad_input, grad_output):
            gradients_fisher[name] = grad_output[0].detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_act_fisher(name)))
            hooks.append(module.register_full_backward_hook(save_grad_fisher(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    fisher_accumulator = {}
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 累积 Fisher 信息: gradient² × activation²
        for name in activations_fisher:
            if name in gradients_fisher:
                act = activations_fisher[name]  # (B, C, H, W)
                grad = gradients_fisher[name]   # (B, C, H, W)
                
                # Fisher 信息 = (梯度 × 激活)² 按通道平均
                fisher = (grad * act).pow(2).mean(dim=[0, 2, 3])  # (C,)
                
                if name not in fisher_accumulator:
                    fisher_accumulator[name] = fisher
                else:
                    fisher_accumulator[name] += fisher
    
    for h in hooks:
        h.remove()
    
    for name, fisher in fisher_accumulator.items():
        fisher_scores[name] = fisher.cpu().numpy()
    
    # ============ 2. L1 范数 ============
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            l1_scores[name] = w.abs().view(w.size(0), -1).sum(dim=1).cpu().numpy()
    
    # ============ 3. GRA 语义 ============
    model.eval()
    activations_gra = {}
    hooks = []
    
    def make_hook_gra(name):
        def hook(module, input, output):
            if name not in activations_gra:
                activations_gra[name] = []
            activations_gra[name].append(output.mean(dim=[2, 3]).detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook_gra(name)))
    
    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            all_logits.append(model(inputs).detach())
            all_targets.append(targets.detach())
    
    for h in hooks:
        h.remove()
    
    logits = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    
    # 分类 margin
    correct = logits.gather(1, targets_cat.view(-1, 1)).squeeze()
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets_cat.view(-1, 1), False)
    max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
    margin = correct - max_wrong
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
    
    gra_scores = {}
    rho = 0.5
    for name, act_list in activations_gra.items():
        acts = torch.cat(act_list, dim=0)
        C = acts.size(1)
        scores = []
        for c in range(C):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            delta = (act_norm - margin_norm).abs()
            gamma = (delta.min() + rho * delta.max()) / (delta + rho * delta.max() + 1e-8)
            scores.append(gamma.mean().item())
        gra_scores[name] = np.array(scores)
    
    # ============ 4. 融合: Fisher(50%) + GRA(30%) + L1(20%) ============
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    final_scores = {}
    for name in fisher_scores:
        if name in l1_scores and name in gra_scores:
            combined = (0.50 * norm(fisher_scores[name]) +
                       0.30 * norm(gra_scores[name]) +
                       0.20 * norm(l1_scores[name]))
            final_scores[name] = combined
    
    return final_scores


if __name__ == "__main__":
    print("GRA-Fisher: 基于 Fisher 信息量的语义剪枝")
    print("融合: Fisher(50%) + GRA(30%) + L1(20%)")
    print("理论基础: Fisher 信息量 = E[(gradient × activation)²]")
