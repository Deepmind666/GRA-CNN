import torch
import torch.nn as nn
import numpy as np
from .gra_score import GrayRelationalChannelScorer
from .l1_score import get_l1_scores
from .fpgm_score import FPGMChannelScorer
from .hrank_score import HRankChannelScorer
from ..models import resnet20, resnet56
from ..models.resnet18_tiny import ResNet18Tiny as resnet18_tiny

def get_scores(model, dataloader, method='gra', device='cuda', rho=0.5, scorer=None):
    model.eval()
    scores = {}
    
    if method == 'l1':
        scores = get_l1_scores(model)
        
    elif method == 'fpgm':
        if scorer is None:
            scorer = FPGMChannelScorer(model)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    s = scorer.get_score(name, module)
                    scores[name] = s.detach().cpu().numpy()
                    
    elif method == 'hrank':
        if scorer is None:
            scorer = HRankChannelScorer(model)
            # If scorer was just created, it has no ranks. 
            # We assume scorer passed from outside has collected ranks.
            # If not, we need to collect.
            print("Warning: HRank scorer created inside get_scores, might lack data.")
            scorer.collect_ranks(dataloader, device)
            
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    s = scorer.get_score(name, module)
                    scores[name] = s.detach().cpu().numpy()
    
    elif method == 'gra':
        if scorer is None:
            scorer = GrayRelationalChannelScorer(rho=rho)
            
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
            
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    hooks.append(module.register_forward_hook(get_activation(name)))
        
        try:
            inputs, targets = next(iter(dataloader))
        except StopIteration:
            inputs, targets = next(iter(dataloader)) # Retry if needed

        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            for name, feats in activations.items():
                # Ensure scorer can handle features
                # Check if scorer is our class or legacy
                if hasattr(scorer, 'compute_score'):
                    s = scorer.compute_score(feats, logits, targets)
                else:
                    # Fallback or error
                    s = torch.norm(feats, p=1, dim=(0,2,3)) # Dummy
                scores[name] = s.cpu().numpy()
                
        for h in hooks:
            h.remove()
            
    return scores

def prune_model(model, scorer=None, prune_ratio=0.5, method='gra', dataloader=None, device='cuda', rho=0.5):
    print(f"Calculating scores using {method} (rho={rho})...")
    scores = get_scores(model, dataloader, method, device, rho, scorer)
    
    # Flatten scores to find threshold
    all_scores = np.concatenate([s.flatten() for s in scores.values()])
    threshold = np.percentile(all_scores, prune_ratio * 100)
    print(f"Pruning threshold: {threshold:.4f}")
    
    new_cfg = []
    mask_dict = {}
    
    pruned_channels = 0
    total_channels = 0
    
    # Iterate in order to build cfg
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
             if 'conv1' in name and 'layer' in name:
                 s = scores.get(name)
                 if s is not None:
                     mask = s >= threshold
                     if mask.sum() == 0:
                         idx = np.argmax(s)
                         mask[idx] = True
                         
                     mask_dict[name] = mask
                     new_cfg.append(int(mask.sum()))
                     
                     pruned_channels += len(mask) - mask.sum()
                     total_channels += len(mask)
                 else:
                     # For layers not pruned (e.g. first conv or non-residual blocks if logic differs)
                     # Our logic focuses on 'layer*.conv1'
                     pass
                 
    print(f"Pruned {pruned_channels}/{total_channels} channels. Real Ratio: {pruned_channels/total_channels:.2%}")
    
    # Determine model type
    # This is a bit hacky, ideally pass model_type or infer from class name
    model_type = 'resnet20'
    if 'ResNet56' in str(type(model)):
        model_type = 'resnet56'
    elif 'ResNet18' in str(type(model)):
        model_type = 'resnet18_tiny'
        
    num_classes = model.fc.out_features
    
    return build_new_resnet(model, new_cfg, mask_dict, model_type, num_classes)


def build_new_resnet(old_model, new_cfg, mask_dict, model_type='resnet20', num_classes=10):
    if model_type == 'resnet20':
        new_model = resnet20(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet56':
        new_model = resnet56(num_classes=num_classes, cfg=new_cfg)
    elif model_type == 'resnet18_tiny':
        new_model = resnet18_tiny(num_classes=num_classes, cfg=new_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    old_modules = list(old_model.named_modules())
    new_modules = dict(new_model.named_modules())
    
    last_mask = None
    
    for name, old_module in old_modules:
        if isinstance(old_module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
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
                new_module.weight.data = old_module.weight.data.clone()
                if old_module.bias is not None:
                    new_module.bias.data = old_module.bias.data.clone()
                if isinstance(old_module, nn.BatchNorm2d):
                    new_module.running_mean.data = old_module.running_mean.data.clone()
                    new_module.running_var.data = old_module.running_var.data.clone()

    return new_model
