from torch import nn

# Extra Depthwise (Extra DW) Block
class ExtraDW_block(nn.Module):
    def __init__(self, base_channels: int, expanded_channels: int):
        super().__init__()

        self.extra_dw = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=5, stride=1, padding=2, groups=base_channels, bias=False),
            nn.GroupNorm(base_channels // 8, base_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(base_channels, expanded_channels, kernel_size=1, bias=False),
            nn.GroupNorm(expanded_channels // 8, expanded_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=1, padding=1, groups=expanded_channels, bias=False),
            nn.GroupNorm(expanded_channels // 8, expanded_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(expanded_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(base_channels // 8, base_channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        return self.extra_dw(x)