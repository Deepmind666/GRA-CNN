"""
GRA-MultiLayer: 多层语义传递增强版灰色关联剪枝
==============================================

核心创新:
1. 跨层语义传播: 考虑通道激活如何影响后续层的语义表达
2. 层深度加权: 深层通道与分类决策更直接相关，给予更高权重
3. 语义流聚合: 聚合多个样本的语义流向，减少噪声

理论依据:
- 浅层捕获低级特征(边缘、纹理)，深层捕获高级语义
- 重要通道应在整个语义传播链中保持一致的激活模式
- 与分类logit高度相关的通道更应该被保留
"""

import torch
import torch.nn as nn
import numpy as np

class GRAMultiLayerScorer:
    """多层语义传递增强版GRA评分器"""
    
    def __init__(self, rho=0.5, depth_weight_factor=1.5, confidence_threshold=0.7):
        """
        Args:
            rho: 灰色关联分辨系数
            depth_weight_factor: 深层权重放大因子
            confidence_threshold: 高置信样本阈值
        """
        self.rho = rho
        self.depth_weight_factor = depth_weight_factor
        self.confidence_threshold = confidence_threshold
    
    def compute_scores(self, model, dataloader, device, num_batches=10):
        """
        计算多层语义GRA分数
        
        Returns:
            dict: {layer_name: channel_scores}
        """
        model.eval()
        activations = {}  # {name: [list of (B, C) tensors]}
        layer_order = []  # 记录层顺序
        hooks = []
        
        # 注册hook捕获所有卷积层激活
        def make_hook(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                    layer_order.append(name)
                # 全局平均池化得到通道激活 (B, C)
                act = output.mean(dim=[2, 3]).detach()
                activations[name].append(act)
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # 前向传播收集激活
        all_logits = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                
                # 计算置信度 (softmax后的正确类概率)
                probs = torch.softmax(outputs, dim=1)
                conf = probs.gather(1, targets.view(-1, 1)).squeeze()
                
                all_logits.append(outputs.detach())
                all_targets.append(targets.detach())
                all_confidences.append(conf.detach())
        
        for h in hooks:
            h.remove()
        
        # 合并数据
        logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
        targets = torch.cat(all_targets, dim=0)  # (N,)
        confidences = torch.cat(all_confidences, dim=0)  # (N,)
        
        # 筛选高置信样本 (创新点1: 减少噪声)
        high_conf_mask = confidences > self.confidence_threshold
        if high_conf_mask.sum() < 50:  # 如果高置信样本太少，降低阈值
            high_conf_mask = confidences > 0.3
        
        # 获取正确类别logit作为参考序列
        ref_logits = logits.gather(1, targets.view(-1, 1)).squeeze()  # (N,)
        ref_logits = ref_logits[high_conf_mask]
        
        # 标准化参考序列
        ref_norm = self._robust_normalize(ref_logits)
        
        num_layers = len(layer_order)
        scores = {}
        
        for layer_idx, name in enumerate(layer_order):
            acts = torch.cat(activations[name], dim=0)  # (N, C)
            acts = acts[high_conf_mask]  # 只用高置信样本
            C = acts.size(1)
            
            # 创新点2: 层深度权重 (深层更重要)
            depth_ratio = (layer_idx + 1) / num_layers  # 0~1
            depth_weight = 1.0 + (self.depth_weight_factor - 1.0) * depth_ratio
            
            channel_scores = []
            for c in range(C):
                act_c = acts[:, c]
                
                # Min-Max标准化
                act_norm = self._robust_normalize(act_c)
                
                # 基础GRA分数
                gra_score = self._compute_gra(act_norm, ref_norm)
                
                # 创新点3: 考虑与后续层的语义传播
                if layer_idx < num_layers - 1:
                    next_layer_name = layer_order[layer_idx + 1]
                    next_acts = torch.cat(activations[next_layer_name], dim=0)
                    next_acts = next_acts[high_conf_mask]
                    
                    # 计算与下一层所有通道的平均相关性
                    propagation_score = self._compute_propagation_influence(
                        act_c, next_acts
                    )
                    
                    # 融合: GRA + 传播影响
                    final_score = 0.7 * gra_score + 0.3 * propagation_score
                else:
                    # 最后一层只用GRA
                    final_score = gra_score
                
                # 应用深度权重
                final_score *= depth_weight
                channel_scores.append(final_score)
            
            scores[name] = np.array(channel_scores)
        
        return scores
    
    def _robust_normalize(self, x):
        """鲁棒的Min-Max标准化 (抗异常值)"""
        # 使用5%-95%分位数而非min-max
        q5 = torch.quantile(x, 0.05)
        q95 = torch.quantile(x, 0.95)
        x_clipped = torch.clamp(x, q5, q95)
        x_norm = (x_clipped - q5) / (q95 - q5 + 1e-8)
        return x_norm
    
    def _compute_gra(self, x, ref):
        """计算灰色关联度"""
        delta = (x - ref).abs()
        delta_min = delta.min()
        delta_max = delta.max()
        
        gamma = (delta_min + self.rho * delta_max) / (delta + self.rho * delta_max + 1e-8)
        return gamma.mean().item()
    
    def _compute_propagation_influence(self, act_c, next_layer_acts):
        """
        计算当前通道对下一层的语义传播影响
        
        思路: 如果当前通道激活高时，下一层的整体激活也高，
        说明这个通道在语义传播中很重要
        """
        # 计算当前通道与下一层每个通道的相关性
        act_c_norm = (act_c - act_c.mean()) / (act_c.std() + 1e-8)
        
        correlations = []
        for c2 in range(min(next_layer_acts.size(1), 64)):  # 采样避免太慢
            next_c = next_layer_acts[:, c2]
            next_c_norm = (next_c - next_c.mean()) / (next_c.std() + 1e-8)
            corr = (act_c_norm * next_c_norm).mean().abs().item()
            correlations.append(corr)
        
        # 返回平均相关性 (归一化到0-1)
        avg_corr = np.mean(correlations) if correlations else 0
        return min(avg_corr * 2, 1.0)  # 放大并截断


def compute_gra_multilayer_scores(model, dataloader, device, num_batches=10, rho=0.5):
    """便捷函数: 计算多层语义GRA分数"""
    scorer = GRAMultiLayerScorer(rho=rho)
    return scorer.compute_scores(model, dataloader, device, num_batches)


if __name__ == "__main__":
    print("GRA-MultiLayer 多层语义传递增强版")
    print("创新点:")
    print("  1. 高置信样本筛选 - 减少噪声")
    print("  2. 层深度加权 - 深层更重要")
    print("  3. 跨层传播影响 - 考虑语义流向")
