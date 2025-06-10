import torch as t
from torch import nn

# ResNext module class
class ResNeXt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck_channels: int, groups: int = 1):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortuct = nn.Identity()

        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_channels, kernel_size=(1, 1), stride=1, bias=False),
                nn.GroupNorm(bottleneck_channels // 8, bottleneck_channels),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(3, 3), stride=1, padding=1, groups=groups, bias=False),
                nn.GroupNorm(bottleneck_channels // 8, bottleneck_channels),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(bottleneck_channels, out_channels, kernel_size=(1, 1), stride=1, bias=False),
                nn.GroupNorm(out_channels // 8, out_channels),
                nn.SiLU(inplace=True)
            ),
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        residual = self.shortcut(x)
        
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)

        return t.add(x, residual) # Residual connection