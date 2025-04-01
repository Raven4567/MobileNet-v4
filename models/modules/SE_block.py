from torch import nn

# Squeeze-and-Excitation (SE) Block
class SE_block(nn.Module):
    def __init__(self, base_channels: int, squeezed_channels: int):
        super().__init__()

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, squeezed_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed_channels, base_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.SE(x)