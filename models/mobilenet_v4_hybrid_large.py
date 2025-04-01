import torch as t
from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

# Hybrid-Large модель – для самых тяжелых задач, требующих максимальной точности и мощности
class mobilenet_v4_hybrid_large(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem_block(in_channels=in_channels, base_channels=384)),

            *[nn.Sequential(FusedIB_block(base_channels=384, expanded_channels=768)) for _ in range(3)],

            *[nn.Sequential(UIB_block(base_channels=384, expanded_channels=768), ExtraDW_block(base_channels=384, expanded_channels=768)) for _ in range(7)],

            *[nn.Sequential(SE_block(base_channels=384, squeezed_channels=96), UIB_block(base_channels=384, expanded_channels=768)) for _ in range(6)],

            *[nn.Sequential(SE_block(base_channels=384, squeezed_channels=96), MobileMQA_block(base_channels=384, num_heads=16, downsample=True)) for _ in range(4)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        
            nn.Flatten(),
        
            nn.Linear(384, 1280),
            nn.BatchNorm1d(1280),
            nn.SiLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x