# Fused Inverted Bottleneck Block

import torch as t
from torch import nn

class FusedIB(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            expanded_channels: int, 
            kernel_size: int | tuple[int, int], 
            stride: int = 1, 
            padding: int = 0
        ):
        super().__init__()

        self.fused_ib = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: t.Tensor):
        return self.fused_ib(x)