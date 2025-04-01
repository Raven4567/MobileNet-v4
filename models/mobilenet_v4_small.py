import torch as t
from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

# Small модель – подходит для MNIST, KMNIST, FashionMNIST, простых Atari-сред
class mobilenet_v4_small(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem_block(in_channels, base_channels=48)),

            *[nn.Sequential(FusedIB_block(base_channels=48, expanded_channels=96)) for _ in range(2)],

            *[nn.Sequential(UIB_block(base_channels=48, expanded_channels=96), ExtraDW_block(base_channels=48, expanded_channels=96)) for _ in range(3)],
            
            *[nn.Sequential(SE_block(base_channels=48, squeezed_channels=12), UIB_block(base_channels=48, expanded_channels=96)) for _ in range(4)],
             
            *[nn.Sequential(SE_block(base_channels=48, squeezed_channels=12), MobileMQA_block(base_channels=48, num_heads=6, downsample=False)) for _ in range(2)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        
            nn.Flatten(),
        
            nn.Linear(48, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x