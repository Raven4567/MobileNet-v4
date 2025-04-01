import torch as t
from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

# nano -  самая маленькая модель.
class mobilenet_v4_nano(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem_block(in_channels, base_channels=24)),

            *[nn.Sequential(FusedIB_block(base_channels=24, expanded_channels=64)) for _ in range(2)],

            *[nn.Sequential(UIB_block(base_channels=24, expanded_channels=64), ExtraDW_block(base_channels=24, expanded_channels=64)) for _ in range(2)],
            
            *[nn.Sequential(SE_block(base_channels=24, squeezed_channels=6), UIB_block(base_channels=24, expanded_channels=64)) for _ in range(3)],
             
            *[nn.Sequential(SE_block(base_channels=24, squeezed_channels=6), MobileMQA_block(base_channels=24, num_heads=4, downsample=False)) for _ in range(1)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        
            nn.Flatten(),
        
            nn.Linear(24, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x