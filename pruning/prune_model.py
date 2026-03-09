import torch
import torch.nn as nn
import numpy as np
from .gra_score import GrayRelationalChannelScorer
from .l1_score import get_l1_scores
from .fpgm_score import FPGMChannelScorer
from .hrank_score import HRankChannelScorer
from models.resnet_cifar import resnet20, resnet56, resnet110
from models.resnet18_tiny import ResNet18Tiny as resnet18_tiny
from models.vgg_cifar import vgg16

def get_scores(model, dataloader, method='gra', device='cuda', rho=0.5, scorer=None):
    model.eval()
    scores = {}
    
    # Define which layers to prune
    def is_prunable(name, module):
        if isinstance(module, nn.Conv2d):
            # ResNet
            if 'conv1' in name and 'layer' in name:
                return True
            # VGG
            if 'features' in name:
                return True
        return False

    if method == 'l1':
        # L1 scorer usually handles all convs, we might need to filter
        raw_scores = get_l1_scores(model)
        for name, s in raw_scores.items():
            # Check if this layer is in our prunable list logic (simplified)
            # Actually get_l1_scores likely returns all. We filter later.
            scores[name] = s
            
    elif method == 'fpgm':
        if scorer is None: scorer = FPGMChannelScorer(model)
        for name, module in model.named_modules():
            if is_prunable(name, module):
                s = scorer.get_score(name, module)
                scores[name] = s.detach().cpu().numpy()
                    
    elif method == 'hrank':
        if scorer is None:
            scorer = HRankChannelScorer(model)
            print("Warning: HRank scorer created inside get_scores.")
            scorer.collect_ranks(dataloader, device)
            
        for name, module in model.named_modules():
            if is_prunable(name, module):
                s = scorer.get_score(name, module)
                scores[name] = s.detach().cpu().numpy()
    
    elif method == 'gra':
        if scorer is None: scorer = GrayRelationalChannelScorer(rho=rho)
            
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in model.named_modules():
            if is_prunable(name, module):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        try:
            inputs, targets = next(iter(dataloader))
        except StopIteration:
            inputs, targets = next(iter(dataloader))

        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            for name, feats in activations.items():
                if hasattr(scorer, 'compute_score'):
                    s = scorer.compute_score(feats, logits, targets)
                else:
                    # Fallback
                    real_scorer = GrayRelationalChannelScorer(rho=0.5)
                    s = real_scorer.compute_score(feats, logits, targets)
                scores[name] = s.cpu().numpy()
                
        for h in hooks:
            h.remove()
            
    return scores

def prune_model(model, scorer=None, prune_ratio=0.5, method='gra', dataloader=None, device='cuda', rho=0.5, model_type=None):
    print(f"Calculating scores using {method} (rho={rho})...")
    scores = get_scores(model, dataloader, method, device, rho, scorer)
    
    # Filter scores to only include prunable layers
    # (Re-verify logic to match is_prunable)
    valid_scores = {}
    for name, s in scores.items():
        if ('conv1' in name and 'layer' in name) or ('features' in name):
            valid_scores[name] = s

    # Flatten scores to find threshold
    all_scores = np.concatenate([s.flatten() for s in valid_scores.values()])
    threshold = np.percentile(all_scores, prune_ratio * 100)
    print(f"Pruning threshold: {threshold:.4f}")
    
    mask_dict = {}
    
    # Determine model type
    if model_type is None:
        if 'VGG' in str(type(model)):
            model_type = 'vgg16'
        elif hasattr(model, 'layer4') and len(list(model.layer4)) == 2:
            # Tiny-ImageNet ResNet-18 has 4 stages with 2 blocks each.
            model_type = 'resnet18_tiny'
        elif hasattr(model, 'layer3'):
            num_blocks = len(list(model.layer3))
            if num_blocks >= 18:
                model_type = 'resnet110'
            elif num_blocks >= 9:
                model_type = 'resnet56'
            else:
                model_type = 'resnet20'
        elif 'ResNet18' in str(type(model)):
            model_type = 'resnet18_tiny'
        else:
            model_type = 'resnet20'

    print(f"Building new {model_type}...")

    if 'vgg' in model_type:
        return build_new_vgg(model, valid_scores, threshold, model_type)
    else:
        # ResNet Logic (Cfg building)
        new_cfg = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    s = valid_scores.get(name)
                    if s is not None:
                        mask = s >= threshold
                        if mask.sum() == 0: mask[np.argmax(s)] = True # Safety
                        mask_dict[name] = mask
                        new_cfg.append(int(mask.sum()))
        
        num_classes = model.fc.out_features
        return build_new_resnet(model, new_cfg, mask_dict, model_type, num_classes)


def build_new_vgg(old_model, scores, threshold, model_type='vgg16'):
    # VGG Config Building
    # Iterate features, if conv, calculate mask, append to cfg. If maxpool, append 'M'.
    new_cfg = []
    mask_dict = {}
    
    for name, module in old_model.features.named_children():
        full_name = f"features.{name}"
        if isinstance(module, nn.Conv2d):
            s = scores.get(full_name)
            if s is not None:
                mask = s >= threshold
                num_ch = len(s)
                min_keep = max(4, int(num_ch * 0.1))
                if mask.sum() < min_keep:
                    # Keep top-min_keep channels to prevent information collapse
                    topk_idx = np.argsort(s)[-min_keep:]
                    mask = np.zeros(num_ch, dtype=bool)
                    mask[topk_idx] = True
                mask_dict[full_name] = mask
                new_cfg.append(int(mask.sum()))
            else:
                # Should not happen for VGG16 if all convs are pruned
                new_cfg.append(module.out_channels)
        elif isinstance(module, nn.MaxPool2d):
            new_cfg.append('M')
            
    num_classes = old_model.classifier.out_features
    new_model = vgg16(num_classes=num_classes, cfg=new_cfg)
    new_model.to(next(old_model.parameters()).device)
    
    # Copy Weights
    old_features = list(old_model.features.children())
    new_features = list(new_model.features.children())
    
    last_mask = None
    
    # Assume 1-to-1 mapping
    for i, (old_m, new_m) in enumerate(zip(old_features, new_features)):
        if isinstance(old_m, nn.Conv2d):
            # Output Mask
            full_name = f"features.{i}" # name in named_children is index string
            mask = mask_dict.get(full_name)
            if mask is None:
                # Maybe fallback if names mismatch, but here index should align
                # Try finding by iteration order since we built cfg that way
                pass

            mask_indices = np.where(mask)[0]
            
            # 1. Output Dimension Pruning (Weight & Bias)
            # new_m.weight shape: [out, in, k, k]
            if last_mask is None:
                # First layer, keep input channels
                w = old_m.weight.data[mask_indices, :, :, :]
            else:
                # Prune input channels based on last_mask
                last_mask_indices = np.where(last_mask)[0]
                w = old_m.weight.data[mask_indices][:, last_mask_indices, :, :]
                
            new_m.weight.data = w
            if old_m.bias is not None:
                new_m.bias.data = old_m.bias.data[mask_indices]
            
            last_mask = mask
            
        elif isinstance(old_m, nn.BatchNorm2d):
            # Prune based on last_mask (which is this layer's channels)
            mask_indices = np.where(last_mask)[0]
            new_m.weight.data = old_m.weight.data[mask_indices]
            new_m.bias.data = old_m.bias.data[mask_indices]
            new_m.running_mean.data = old_m.running_mean.data[mask_indices]
            new_m.running_var.data = old_m.running_var.data[mask_indices]
            
    # Classifier (Linear)
    # VGG Cifar: Linear(512, num_classes) -> Input dim depends on last conv output
    # But wait, VGG usually has AvgPool before classifier, so feature size is 1x1?
    # vgg_cifar.py uses view(out.size(0), -1).
    # If AvgPool(1,1) is used (which is global avg pool effectively if size matches), then input to linear is just channels.
    # vgg_cifar.py: layers += [nn.AvgPool2d(kernel_size=1, stride=1)] -> This looks weird for Cifar (32x32).
    # Standard VGG reduces 32 -> 16 -> 8 -> 4 -> 2 -> 1 (5 maxpools).
    # Let's check vgg_cifar.py structure again.
    # VGG16 has 5 'M'. 32 / 2^5 = 1. So final feature map is 1x1.
    # So Linear input dim = last_conv_out_channels.
    
    old_linear = old_model.classifier
    new_linear = new_model.classifier
    
    last_mask_indices = np.where(last_mask)[0]
    new_linear.weight.data = old_linear.weight.data[:, last_mask_indices]
    new_linear.bias.data = old_linear.bias.data.clone()
    
    return new_model

def build_new_resnet(old_model, new_cfg, mask_dict, model_type='resnet20', num_classes=10):
    if model_type == 'resnet20':
        new_model = resnet20(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet56':
        new_model = resnet56(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet110':
        new_model = resnet110(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet18_tiny':
        new_model = resnet18_tiny(num_classes=num_classes, cfg=new_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    old_modules = list(old_model.named_modules())
    new_modules = dict(new_model.named_modules())
    
    last_mask = None
    
    for name, old_module in old_modules:
        if isinstance(old_module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            if name not in new_modules: continue # Safety
            new_module = new_modules[name]
            
            if 'conv1' in name and 'layer' in name:
                mask = mask_dict[name]
                mask_indices = np.where(mask)[0]
                
                if isinstance(old_module, nn.Conv2d):
                    new_module.weight.data = old_module.weight.data[mask_indices, :, :, :]
                    last_mask = mask
                    
            elif 'bn1' in name and 'layer' in name:
                mask = last_mask
                mask_indices = np.where(mask)[0]
                new_module.weight.data = old_module.weight.data[mask_indices]
                new_module.bias.data = old_module.bias.data[mask_indices]
                new_module.running_mean.data = old_module.running_mean.data[mask_indices]
                new_module.running_var.data = old_module.running_var.data[mask_indices]
                
            elif 'conv2' in name and 'layer' in name:
                mask = last_mask
                mask_indices = np.where(mask)[0]
                if isinstance(old_module, nn.Conv2d):
                    new_module.weight.data = old_module.weight.data[:, mask_indices, :, :]
                    
            else:
                # Copy directly (e.g. fc, or non-pruned convs)
                # Note: for Linear, ResNet GlobalAvgPool handles dimension matching (channels don't change at block output in BasicBlock pruning usually? 
                # Wait, BasicBlock pruning usually only prunes internal channels (conv1 output), conv2 output matches block expansion.
                # So FC input dimension is unchanged. Correct.
                if old_module.weight.shape == new_module.weight.shape:
                    new_module.weight.data = old_module.weight.data.clone()
                    if old_module.bias is not None:
                        new_module.bias.data = old_module.bias.data.clone()
                    if isinstance(old_module, nn.BatchNorm2d):
                        new_module.running_mean.data = old_module.running_mean.data.clone()
                        new_module.running_var.data = old_module.running_var.data.clone()
                else:
                    print(f"Warning: Shape mismatch for {name}, skipping copy.")

    return new_model


def _normalize_mask(mask, out_channels, layer_name=None):
    """Normalize a mask to a boolean numpy array with safe fallbacks."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).astype(bool)
    if mask.size != out_channels:
        print(f"Warning: mask size mismatch for {layer_name} "
              f"(got {mask.size}, expected {out_channels}). Keeping all channels.")
        mask = np.ones(out_channels, dtype=bool)
    if mask.sum() == 0:
        # Safety: keep at least one channel
        mask[0] = True
    return mask


def build_new_vgg_from_mask(old_model, mask_dict, model_type='vgg16'):
    """
    Build a new VGG model from channel masks (structured pruning).

    mask_dict: dict[layer_name] -> binary mask (1=keep, 0=prune)
    """
    # Build cfg from masks
    new_cfg = []
    for name, module in old_model.features.named_children():
        full_name = f"features.{name}"
        if isinstance(module, nn.Conv2d):
            mask = mask_dict.get(full_name, np.ones(module.out_channels, dtype=bool))
            mask = _normalize_mask(mask, module.out_channels, full_name)
            new_cfg.append(int(mask.sum()))
        elif isinstance(module, nn.MaxPool2d):
            new_cfg.append('M')

    num_classes = old_model.classifier.out_features
    new_model = vgg16(num_classes=num_classes, cfg=new_cfg)
    new_model.to(next(old_model.parameters()).device)

    # Copy weights
    old_features = list(old_model.features.children())
    new_features = list(new_model.features.children())
    last_mask = None
    last_mask_indices = None

    for i, (old_m, new_m) in enumerate(zip(old_features, new_features)):
        if isinstance(old_m, nn.Conv2d):
            full_name = f"features.{i}"
            mask = mask_dict.get(full_name, np.ones(old_m.out_channels, dtype=bool))
            mask = _normalize_mask(mask, old_m.out_channels, full_name)
            mask_indices = np.where(mask)[0]

            if last_mask is None:
                w = old_m.weight.data[mask_indices, :, :, :]
            else:
                w = old_m.weight.data[mask_indices][:, last_mask_indices, :, :]
            new_m.weight.data = w
            if old_m.bias is not None:
                new_m.bias.data = old_m.bias.data[mask_indices]

            last_mask = mask
            last_mask_indices = mask_indices

        elif isinstance(old_m, nn.BatchNorm2d):
            if last_mask_indices is None:
                continue
            new_m.weight.data = old_m.weight.data[last_mask_indices]
            new_m.bias.data = old_m.bias.data[last_mask_indices]
            new_m.running_mean.data = old_m.running_mean.data[last_mask_indices]
            new_m.running_var.data = old_m.running_var.data[last_mask_indices]

    # Classifier
    old_linear = old_model.classifier
    new_linear = new_model.classifier
    if last_mask_indices is None:
        last_mask_indices = np.arange(old_linear.weight.data.size(1))
    new_linear.weight.data = old_linear.weight.data[:, last_mask_indices]
    new_linear.bias.data = old_linear.bias.data.clone()

    return new_model


def build_new_resnet_from_mask(old_model, mask_dict, model_type=None):
    """
    Build a new ResNet (CIFAR) model from channel masks.

    Only conv1 inside BasicBlock is pruned; conv2 output channels stay unchanged.
    """
    # Determine model type if not provided
    if model_type is None:
        if hasattr(old_model, 'layer4') and len(list(old_model.layer4)) == 2:
            model_type = 'resnet18_tiny'
        elif hasattr(old_model, 'layer3'):
            num_blocks = len(list(old_model.layer3))
            if num_blocks >= 18:
                model_type = 'resnet110'
            elif num_blocks >= 9:
                model_type = 'resnet56'
            else:
                model_type = 'resnet20'
        else:
            model_type = 'resnet20'

    # Sanitize masks for conv1 in each block and build cfg in order
    new_cfg = []
    sanitized_masks = {}
    for name, module in old_model.named_modules():
        if isinstance(module, nn.Conv2d) and 'layer' in name and '.conv1' in name:
            mask = mask_dict.get(name, np.ones(module.out_channels, dtype=bool))
            mask = _normalize_mask(mask, module.out_channels, name)
            sanitized_masks[name] = mask
            new_cfg.append(int(mask.sum()))

    num_classes = old_model.fc.out_features
    return build_new_resnet(old_model, new_cfg, sanitized_masks, model_type, num_classes)


# =============================================================================
# 全网络剪枝接口 (Stage级统一通道剪枝)
# =============================================================================

def _copy_block_weights_mid_only(old_block, new_block, mask_idx):
    """
    复制BasicBlock权重 - 路径B: 只剪中间通道(mid_planes)

    结构: x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+shortcut) -> out
    - conv1: [inplanes -> mid_planes] 输出通道剪枝
    - conv2: [mid_planes -> planes] 输入通道剪枝，输出保持不变
    - shortcut/bn2/block输出: 保持原始维度不变
    """
    # conv1: 输入保持原样，输出按mask剪枝
    new_block.conv1.weight.data = old_block.conv1.weight.data[mask_idx, :, :, :]

    # bn1: 跟随conv1输出
    new_block.bn1.weight.data = old_block.bn1.weight.data[mask_idx]
    new_block.bn1.bias.data = old_block.bn1.bias.data[mask_idx]
    new_block.bn1.running_mean.data = old_block.bn1.running_mean.data[mask_idx]
    new_block.bn1.running_var.data = old_block.bn1.running_var.data[mask_idx]

    # conv2: 输入通道剪枝(mask_idx)，输出通道保持原样(planes不变)
    new_block.conv2.weight.data = old_block.conv2.weight.data[:, mask_idx, :, :]

    # bn2: 保持原样(跟随conv2输出=planes)
    new_block.bn2.weight.data = old_block.bn2.weight.data.clone()
    new_block.bn2.bias.data = old_block.bn2.bias.data.clone()
    new_block.bn2.running_mean.data = old_block.bn2.running_mean.data.clone()
    new_block.bn2.running_var.data = old_block.bn2.running_var.data.clone()

    # shortcut: 保持原样(不剪枝)
    if old_block.downsample is not None:
        for i, (old_m, new_m) in enumerate(zip(old_block.downsample, new_block.downsample)):
            if hasattr(old_m, 'weight'):
                new_m.weight.data = old_m.weight.data.clone()
            if hasattr(old_m, 'bias') and old_m.bias is not None:
                new_m.bias.data = old_m.bias.data.clone()
            if hasattr(old_m, 'running_mean'):
                new_m.running_mean.data = old_m.running_mean.data.clone()
                new_m.running_var.data = old_m.running_var.data.clone()


def build_resnet_stage_pruned(old_model, stage_masks, model_type='resnet20'):
    """
    路径B: 只剪中间通道(mid_planes)的结构化剪枝

    保持ResNet结构不变:
    - block输出通道(planes)不变: 16/32/64
    - 只剪conv1输出 + conv2输入 (mid_planes)
    - shortcut/fc保持原样

    Args:
        old_model: 原始模型
        stage_masks: {stage_name: bool_array} 每个stage的mid_planes掩码
        model_type: 架构类型

    Returns:
        new_model: 剪枝后的模型
    """
    num_classes = old_model.fc.out_features

    # 确定每个stage的block数量
    n_blocks = len(list(old_model.layer1))

    # 构建cfg: 长度为3*n，每个block一个mid_planes值
    new_cfg = []
    for stage_name in ['layer1', 'layer2', 'layer3']:
        mask = stage_masks.get(stage_name)
        if mask is not None:
            mid_ch = int(mask.sum())
        else:
            stage = getattr(old_model, stage_name)
            mid_ch = stage[0].conv1.out_channels
        # 该stage所有block使用相同的mid_planes
        new_cfg.extend([mid_ch] * n_blocks)

    # 创建新模型
    if model_type == 'resnet20':
        new_model = resnet20(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet56':
        new_model = resnet56(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet110':
        new_model = resnet110(num_classes=num_classes, cfg=new_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = next(old_model.parameters()).device
    new_model.to(device)

    # 复制首层conv1权重(不剪枝)
    new_model.conv1.weight.data = old_model.conv1.weight.data.clone()
    new_model.bn1.weight.data = old_model.bn1.weight.data.clone()
    new_model.bn1.bias.data = old_model.bn1.bias.data.clone()
    new_model.bn1.running_mean.data = old_model.bn1.running_mean.data.clone()
    new_model.bn1.running_var.data = old_model.bn1.running_var.data.clone()

    # 复制各stage的权重
    for stage_name in ['layer1', 'layer2', 'layer3']:
        old_stage = getattr(old_model, stage_name)
        new_stage = getattr(new_model, stage_name)
        mask = stage_masks.get(stage_name)
        if mask is None:
            mask = np.ones(old_stage[0].conv1.out_channels, dtype=bool)
        mask_idx = np.where(mask)[0]

        for block_idx in range(len(old_stage)):
            old_block = old_stage[block_idx]
            new_block = new_stage[block_idx]
            _copy_block_weights_mid_only(old_block, new_block, mask_idx)

    # 复制fc层(不剪枝)
    new_model.fc.weight.data = old_model.fc.weight.data.clone()
    new_model.fc.bias.data = old_model.fc.bias.data.clone()

    return new_model
