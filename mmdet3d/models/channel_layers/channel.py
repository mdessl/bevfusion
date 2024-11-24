from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import CHANNEL_LAYERS

__all__ = ["ChannelLayer"]


@CHANNEL_LAYERS.register_module()
class ChannelLayer(nn.Module):
    def __init__(self, in_channels: int = 80, out_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
