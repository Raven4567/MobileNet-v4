# Fused Inverted Bottleneck Block
from torch import nn

class FusedIB_block(nn.Module):
    def __init__(self, base_channels: int, expanded_channels: int):
        super().__init__()

        self.fused_ib = nn.Sequential(
            nn.Conv2d(base_channels, expanded_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(expanded_channels // 8, expanded_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(expanded_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(base_channels // 8, base_channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        return self.fused_ib(x)