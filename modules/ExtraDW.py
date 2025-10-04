import torch as t
from torch import nn

# Extra Depthwise (Extra DW) Block
class ExtraDW(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            expanded_channels: int, 
            kernel_size_1st: int | tuple[int, int], 
            kernel_size_2nd: int | tuple[int, int], 
            stride: int = 1,
            padding_1st: int = 0, 
            padding_2nd: int = 0
        ):
        super().__init__()

        self.extra_dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size_1st, 1, padding_1st, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, expanded_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(expanded_channels, expanded_channels, kernel_size_2nd, stride, padding_2nd, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(expanded_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: t.Tensor):
        return self.extra_dw(x)