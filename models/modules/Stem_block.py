# Stem Block (для обработки разных размеров изображений)
from torch import nn

class Stem_block(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(base_channels // 8, base_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.stem(x)