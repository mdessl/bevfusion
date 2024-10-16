import copy
import torch
from collections import deque


__all__ = ["convert_sync_batchnorm", "convert_model_ghost_batchnorm"]


def convert_sync_batchnorm(input_model, exclude=[]):
    for name, module in input_model._modules.items():
        skip = sum([ex in name for ex in exclude])
        if skip:
            continue
        input_model._modules[name] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    return input_model

class GhostBatchNorm(torch.nn.BatchNorm2d):
    def __init__(self, num_features, num_splits=2, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * num_splits))
        self.register_buffer('running_var', torch.ones(num_features * num_splits))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            split_input = input.view(N * self.num_splits, C // self.num_splits, H, W)
            output = torch.nn.functional.batch_norm(
                split_input, self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps)
            return output.view(N, C, H, W)
        else:
            return torch.nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

def convert_ghost_batchnorm(module, num_splits=2, exclude=[]):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and \
       not any(ex in module.__class__.__name__ for ex in exclude):
        module_output = GhostBatchNorm(
            module.num_features,
            num_splits=num_splits,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
        module_output.running_mean.copy_(module.running_mean.repeat(num_splits))
        module_output.running_var.copy_(module.running_var.repeat(num_splits))
        module_output.num_batches_tracked.copy_(module.num_batches_tracked)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ghost_batchnorm(child, num_splits, exclude))
    del module
    return module_output

def convert_model_ghost_batchnorm(input_model, num_splits=2, exclude=[]):
    for name, module in input_model._modules.items():
        skip = sum([ex in name for ex in exclude])
        if skip:
            continue
        input_model._modules[name] = convert_ghost_batchnorm(module, num_splits)
    return input_model
