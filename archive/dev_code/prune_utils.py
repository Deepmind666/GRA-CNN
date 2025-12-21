import torch
import torch.nn as nn
import numpy as np

def get_model_complexity_info(model, input_res, print_per_layer_stat=False):
    """
    Simple FLOPs and Params counter for Conv2d and Linear layers.
    input_res: (C, H, W) tuple
    """
    flops = 0
    params = 0
    
    # This is a simplified counter. For rigorous counting, use thop or fvcore.
    # We'll simulate a forward pass to get shapes.
    
    def count_hooks(module, input, output):
        nonlocal flops, params
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * K * K * Cin * Cout * Hout * Wout / groups
            # (2 for multiply-add)
            batch_size = input[0].size(0)
            output_dims = list(output.size()[2:])
            
            kernel_dims = list(module.kernel_size)
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = 2 * kernel_dims[0] * kernel_dims[1] * in_channels * filters_per_channel
            active_elements_count = batch_size * int(np.prod(output_dims))
            
            layer_flops = conv_per_position_flops * int(np.prod(output_dims))
            
            flops += layer_flops
            params += sum(p.numel() for p in module.parameters())
            
        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * Cin * Cout
            weight_ops = module.weight.numel() * 2
            bias_ops = module.bias.numel() if module.bias is not None else 0
            flops += weight_ops + bias_ops
            params += sum(p.numel() for p in module.parameters())

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_hooks))

    input_t = torch.randn(1, *input_res).to(next(model.parameters()).device)
    model(input_t)

    for h in hooks:
        h.remove()
        
    return flops, params

def get_l1_scores(model):
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # L1 norm of weights: [Out, In, K, K] -> sum over [In, K, K] -> [Out]
            filters = module.weight.data
            l1 = filters.abs().view(filters.size(0), -1).sum(dim=1)
            scores[name] = l1
    return scores

def apply_pruning_mask(model, masks):
    """
    Zero out pruned channels in weight matrices.
    Note: This does not physically remove channels, just masks them.
    To physically remove, we need to rebuild the model.
    """
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name].to(module.weight.device)
            module.weight.data.mul_(mask.view(-1, 1, 1, 1))
            if module.bias is not None:
                module.bias.data.mul_(mask)
