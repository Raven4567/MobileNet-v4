import torch as t
from torch import nn

# Squeeze-and-Excitation (SE) Block
class SE(nn.Module):
    def __init__(self, in_channels: int, squeeze_ratio: float):
        super().__init__()

        squeezed_channels = int(in_channels * squeeze_ratio)

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Conv2d(in_channels, squeezed_channels, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(squeezed_channels // 8, squeezed_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(squeezed_channels, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x: t.Tensor):
        return x * self.SE(x)