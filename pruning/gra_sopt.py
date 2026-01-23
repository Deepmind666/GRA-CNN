"""
GRA-SOPT: 语义优化剪枝 (Semantically-Optimized Pruning with Taylor)
==================================================================

目标: 全面胜过经典方法(L1, FPGM)和SOTA方法(HRank, Taylor)

核心思想:
1. Taylor展开: 移除通道对损失的影响 = |activation * gradient|
2. L1稳定性: 权重范数提供基础保障
3. GRA语义对齐: 与分类决策的相关性
4. 多策略投票: 综合多种评估方法的优势

理论依据 (对标SOTA):
- Taylor Expansion (NeurIPS 2016): 一阶近似通道移除影响
- HRank (CVPR 2020): 特征图秩作为重要性
- GReg (NeurIPS 2021): 正则化引导剪枝
- 我们的GRA-SOPT: 多维度融合+语义对齐
"""

import torch
import torch.nn as nn
import numpy as np

def compute_gra_sopt_scores(model, dataloader, device, num_batches=8):
    """
    GRA-SOPT: 语义优化剪枝
    
    融合4种评估维度:
    1. Taylor重要性 (30%): |activation × gradient|
    2. L1范数 (25%): 权重绝对值
    3. GRA语义 (25%): 与分类margin对齐
    4. 激活统计 (20%): 激活均值和方差
    
    Returns:
        dict: {layer_name: importance_scores}
    """
    model.eval()
    
    # ============ 1. Taylor重要性 ============
    taylor_scores = {}
    activations_taylor = {}
    gradients_taylor = {}
    hooks = []
    
    def save_activation_taylor(name):
        def hook(module, input, output):
            activations_taylor[name] = output.detach()
        return hook
    
    def save_gradient_taylor(name):
        def hook(module, grad_input, grad_output):
            if name not in gradients_taylor:
                gradients_taylor[name] = []
            gradients_taylor[name].append(grad_output[0].detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_activation_taylor(name)))
            hooks.append(module.register_full_backward_hook(save_gradient_taylor(name)))
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    taylor_importance = {}
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 计算每个通道的Taylor重要性
        for name in activations_taylor:
            if name in gradients_taylor and gradients_taylor[name]:
                act = activations_taylor[name]  # (B, C, H, W)
                grad = gradients_taylor[name][-1]  # (B, C, H, W)
                
                # Taylor: |activation * gradient| 按通道求和
                importance = (act * grad).abs().mean(dim=[0, 2, 3])  # (C,)
                
                if name not in taylor_importance:
                    taylor_importance[name] = importance
                else:
                    taylor_importance[name] += importance
    
    for h in hooks:
        h.remove()
    
    for name in taylor_importance:
        taylor_scores[name] = taylor_importance[name].cpu().numpy()
    
    # ============ 2. L1范数 ============
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            score = weight.abs().view(weight.size(0), -1).sum(dim=1)
            l1_scores[name] = score.cpu().numpy()
    
    # ============ 3. GRA语义 ============
    model.eval()
    activations_gra = {}
    hooks = []
    
    def make_hook_gra(name):
        def hook(module, input, output):
            if name not in activations_gra:
                activations_gra[name] = []
            act = output.mean(dim=[2, 3]).detach()
            activations_gra[name].append(act)
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook_gra(name)))
    
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
    targets_cat = torch.cat(all_targets, dim=0)
    
    # 使用分类margin
    correct_logits = logits.gather(1, targets_cat.view(-1, 1)).squeeze()
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets_cat.view(-1, 1), False)
    max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
    margin = correct_logits - max_wrong
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
    
    gra_scores = {}
    rho = 0.5
    for name, act_list in activations_gra.items():
        acts = torch.cat(act_list, dim=0)
        C = acts.size(1)
        
        channel_scores = []
        for c in range(C):
            act_c = acts[:, c]
            act_norm = (act_c - act_c.min()) / (act_c.max() - act_c.min() + 1e-8)
            
            delta = (act_norm - margin_norm).abs()
            delta_min, delta_max = delta.min(), delta.max()
            gamma = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
            channel_scores.append(gamma.mean().item())
        
        gra_scores[name] = np.array(channel_scores)
    
    # ============ 4. 激活统计 ============
    activation_scores = {}
    for name, act_list in activations_gra.items():
        acts = torch.cat(act_list, dim=0)  # (N, C)
        # 激活均值 + 激活方差 (高均值高方差 = 重要)
        mean_score = acts.mean(dim=0).abs()
        var_score = acts.var(dim=0)
        combined = mean_score + 0.5 * var_score
        activation_scores[name] = combined.cpu().numpy()
    
    # ============ 5. 融合: 加权投票 ============
    weights = {
        'taylor': 0.30,  # Taylor展开最准确
        'l1': 0.25,      # L1稳定可靠
        'gra': 0.25,     # GRA语义对齐
        'activation': 0.20  # 激活统计补充
    }
    
    final_scores = {}
    for name in taylor_scores:
        if name in l1_scores and name in gra_scores and name in activation_scores:
            # 归一化
            def normalize(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            
            taylor_norm = normalize(taylor_scores[name])
            l1_norm = normalize(l1_scores[name])
            gra_norm = normalize(gra_scores[name])
            act_norm = normalize(activation_scores[name])
            
            # 加权融合
            combined = (weights['taylor'] * taylor_norm + 
                       weights['l1'] * l1_norm +
                       weights['gra'] * gra_norm +
                       weights['activation'] * act_norm)
            
            final_scores[name] = combined
    
    return final_scores


if __name__ == "__main__":
    print("GRA-SOPT: 语义优化剪枝")
    print("融合: Taylor(30%) + L1(25%) + GRA(25%) + Activation(20%)")
    print("目标: 全面胜过L1, FPGM, HRank等方法")
