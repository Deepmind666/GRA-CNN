import torch
import torch.nn as nn

def get_l1_scores(model):
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv1' in name and 'layer' in name:
                filters = module.weight.data
                s = filters.abs().view(filters.size(0), -1).sum(dim=1)
                scores[name] = s.cpu().numpy()
    return scores

class L1ChannelScorer:
    def __init__(self, model):
        self.model = model
        
    def get_score(self, layer_name, layer_module, **kwargs):
        filters = layer_module.weight.data
        return filters.abs().view(filters.size(0), -1).sum(dim=1)
