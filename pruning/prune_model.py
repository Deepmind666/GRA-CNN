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
        elif hasattr(model, 'layer3'):
             num_blocks = len(list(model.layer3))
             if num_blocks >= 18: model_type = 'resnet110'
             elif num_blocks >= 9: model_type = 'resnet56'
             else: model_type = 'resnet20'
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
                if mask.sum() == 0: mask[np.argmax(s)] = True
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
