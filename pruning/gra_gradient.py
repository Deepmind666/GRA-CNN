"""
GRA-Gradient: 梯度引导的语义剪枝
================================

核心思想:
- 传统GRA只看激活与logit的相关性
- GRA-Gradient增加梯度信息：对损失敏感的通道更重要
- 结合语义对齐(GRA)和梯度敏感性，双重保障

理论依据:
- 梯度大的通道 = 对损失影响大 = 移除会显著损害性能
- GRA高的通道 = 与决策对齐 = 语义关键
- 两者结合 = 同时保留功能重要和语义重要的通道
"""

import torch
import torch.nn as nn
import numpy as np

def compute_gra_gradient_scores(model, dataloader, device, num_batches=5, rho=0.5, 
                                 gra_weight=0.4, grad_weight=0.6):
    """
    GRA-Gradient: 梯度引导的语义剪枝评分
    
    Args:
        gra_weight: GRA分数权重 (默认0.4)
        grad_weight: 梯度分数权重 (默认0.6)
    
    Returns:
        dict: {layer_name: combined_scores}
    """
    model.eval()
    
    # ============ 第一步: 计算梯度敏感性 ============
    gradient_scores = {}
    
    # 注册梯度钩子
    gradients = {}
    activations = {}
    hooks = []
    
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if name not in gradients:
                gradients[name] = []
            gradients[name].append(grad_output[0].detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(save_activation(name)))
            hooks.append(module.register_full_backward_hook(save_gradient(name)))
    
    # 前向+反向传播收集梯度
    model.train()  # 需要梯度
    criterion = nn.CrossEntropyLoss()
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    # 计算每个通道的平均梯度幅度
    for name in gradients:
        all_grads = torch.cat(gradients[name], dim=0)  # (N, C, H, W)
        # 通道梯度幅度 = 该通道梯度的L2范数
        channel_grad_mag = all_grads.abs().mean(dim=[0, 2, 3])  # (C,)
        gradient_scores[name] = channel_grad_mag.cpu().numpy()
    
    # ============ 第二步: 计算GRA分数 ============
    model.eval()
    activations_gra = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations_gra:
                activations_gra[name] = []
            act = output.mean(dim=[2, 3]).detach()
            activations_gra[name].append(act)
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
    targets_cat = torch.cat(all_targets, dim=0)
    
    # 使用分类 margin 作为参考序列 (正确类logit - 最大错误类logit)
    correct_logits = logits.gather(1, targets_cat.view(-1, 1)).squeeze()
    
    # 创建mask排除正确类
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets_cat.view(-1, 1), False)
    max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
    
    margin = correct_logits - max_wrong  # 分类 margin
    
    # Min-Max标准化
    margin_norm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-8)
    
    gra_scores = {}
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
    
    # ============ 第三步: 融合 ============
    combined_scores = {}
    for name in gradient_scores:
        if name in gra_scores:
            grad = gradient_scores[name]
            gra = gra_scores[name]
            
            # Min-Max归一化
            grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
            gra_norm = (gra - gra.min()) / (gra.max() - gra.min() + 1e-8)
            
            # 加权融合
            combined = gra_weight * gra_norm + grad_weight * grad_norm
            combined_scores[name] = combined
    
    return combined_scores


if __name__ == "__main__":
    print("GRA-Gradient: 梯度引导的语义剪枝")
    print("创新点:")
    print("  1. 使用分类margin作为参考序列")
    print("  2. 梯度敏感性加权")
    print("  3. GRA语义对齐")
