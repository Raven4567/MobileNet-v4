import torch as t
from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

# Large модель – для задач с высокой точностью на CIFAR10, CIFAR100
class mobilenet_v4_large(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem_block(in_channels, base_channels=224)),

            *[nn.Sequential(FusedIB_block(base_channels=224, expanded_channels=512)) for _ in range(3)],

            *[nn.Sequential(UIB_block(base_channels=224, expanded_channels=512), ExtraDW_block(base_channels=224, expanded_channels=512)) for _ in range(6)],

            *[nn.Sequential(SE_block(base_channels=224, squeezed_channels=56), UIB_block(base_channels=224, expanded_channels=512)) for _ in range(5)],

            *[nn.Sequential(SE_block(base_channels=224, squeezed_channels=56), MobileMQA_block(base_channels=224, num_heads=14, downsample=False)) for _ in range(3)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        
            nn.Flatten(),
        
            nn.Linear(224, 1280),
            nn.BatchNorm1d(1280),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x