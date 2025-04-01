import torch as t
from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

# Medium модель – подходит для CIFAR10, CIFAR100, Atari-сред средней сложности
class mobilenet_v4_medium(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem_block(in_channels, base_channels=96)),

            *[nn.Sequential(FusedIB_block(base_channels=96, expanded_channels=192)) for _ in range(2)],

            *[nn.Sequential(UIB_block(base_channels=96, expanded_channels=192), ExtraDW_block(base_channels=96, expanded_channels=192)) for _ in range(5)],

            *[nn.Sequential(SE_block(base_channels=96, squeezed_channels=24), UIB_block(base_channels=96, expanded_channels=192)) for _ in range(6)],

            *[nn.Sequential(SE_block(base_channels=96, squeezed_channels=24), MobileMQA_block(base_channels=96, num_heads=8, downsample=False)) for _ in range(3)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        
            nn.Flatten(),
        
            nn.Linear(96, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x