"""
全网络剪枝实现 - ResNet统一通道剪枝
"""
import torch
import torch.nn as nn
import numpy as np


def get_stage_scores(model, scores, arch):
    """
    按stage聚合通道分数

    Returns:
        stage_scores: {stage_name: aggregated_scores}
    """
    stage_scores = {}

    for stage_name in ['layer1', 'layer2', 'layer3']:
        stage = getattr(model, stage_name, None)
        if stage is None:
            continue

        # 只收集conv1分数 (conv1输出=mid_planes，即被剪的维度)
        # conv2输出=out_planes(block输出)，不被剪，不应参与评分
        block_scores = []
        for i, block in enumerate(stage):
            conv1_name = f'{stage_name}.{i}.conv1'

            if conv1_name in scores:
                block_scores.append(scores[conv1_name])

        # 聚合分数
        if block_scores:
            stage_scores[stage_name] = np.mean(block_scores, axis=0)

    return stage_scores


def generate_stage_masks(stage_scores, ratio):
    """
    为每个stage生成通道掩码

    Args:
        stage_scores: {stage_name: scores}
        ratio: 剪枝比例

    Returns:
        masks: {stage_name: bool_array}
    """
    masks = {}
    for name, scores in stage_scores.items():
        num_ch = len(scores)
        keep = max(4, int(num_ch * (1 - ratio)))
        idx = np.argsort(scores)[-keep:]
        mask = np.zeros(num_ch, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    return masks


def build_pruned_resnet(model, masks, arch, num_classes=10):
    """
    根据掩码重建剪枝后的ResNet

    Args:
        model: 原始模型
        masks: {stage_name: bool_array}
        arch: 架构名称
        num_classes: 分类数

    Returns:
        new_model: 剪枝后的模型
    """
    from pruning.prune_model import build_resnet_stage_pruned
    return build_resnet_stage_pruned(model, masks, arch)
