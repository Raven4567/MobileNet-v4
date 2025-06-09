# Stem Block (для обработки разных размеров изображений)
import torch as t
from torch import nn

class Stem(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int | tuple[int, int], 
            stride: int = 1, 
            padding: int = 0
        ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: t.Tensor):
        return self.stem(x)