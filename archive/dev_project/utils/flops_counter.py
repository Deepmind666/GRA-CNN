import torch
import torch.nn as nn
import numpy as np

def get_model_complexity_info(model, input_res):
    flops = 0
    params = 0
    
    def count_hooks(module, input, output):
        nonlocal flops, params
        if isinstance(module, nn.Conv2d):
            output_dims = list(output.size()[2:])
            kernel_dims = list(module.kernel_size)
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = 2 * kernel_dims[0] * kernel_dims[1] * in_channels * filters_per_channel
            
            layer_flops = conv_per_position_flops * int(np.prod(output_dims))
            
            flops += layer_flops
            params += sum(p.numel() for p in module.parameters())
            
        elif isinstance(module, nn.Linear):
            weight_ops = module.weight.numel() * 2
            bias_ops = module.bias.numel() if module.bias is not None else 0
            flops += weight_ops + bias_ops
            params += sum(p.numel() for p in module.parameters())

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_hooks))

    device = next(model.parameters()).device
    input_t = torch.randn(1, *input_res).to(device)
    model(input_t)

    for h in hooks:
        h.remove()
        
    return flops, params
