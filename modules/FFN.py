# Feed Forward Network (FFN) module for MobileNetV4
# This module implements a feed-forward network (FFN) with a linear layer,

import torch as t
from torch import nn

class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expanded_channels: int):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            
            nn.GroupNorm(in_channels // 8, expanded_channels),
            nn.SiLU(inplace=True),
        
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=True),


            # nn.GroupNorm(in_channels // 8, in_channels),
            # nn.SiLU(inplace=True),
        )

    def forward(self, x: t.Tensor):
        return self.ffn(x) + self.shortcut(x)