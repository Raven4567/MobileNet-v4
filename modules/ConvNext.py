# ConvNext-like block for MobileNetV4
# This module implements a ConvNext-like block with depthwise separable convolution

import torch as t
from torch import nn

class ConvNext(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            expanded_channels: int, 
            kernel_size: int | tuple[int, int], 
            stride: int,
            padding: int = 0
        ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.convnext = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.GroupNorm(in_channels // 8, in_channels),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.GroupNorm(expanded_channels // 8, expanded_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(expanded_channels, out_channels, kernel_size=1),
            # nn.GroupNorm(in_channels // 8, in_channels),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x: t.Tensor):
        return self.convnext(x) + x