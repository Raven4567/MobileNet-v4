import torch as t
from torch import nn

### Module blocks

from modules import ExtraDW, FusedIB, \
                    MobileMQA, SE, \
                    Stem, UIB

# Medium модель – подходит для CIFAR10, CIFAR100, Atari-сред средней сложности
class mobilenet_v4_medium(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem(in_channels, base_channels=144)),

            *[nn.Sequential(FusedIB(base_channels=144, expanded_channels=384)) for _ in range(3)],

            *[nn.Sequential(UIB(base_channels=144, expanded_channels=384), ExtraDW(base_channels=144, expanded_channels=384)) for _ in range(23)],

            *[nn.Sequential(SE(base_channels=144, squeeze_ratio=0.25), UIB(base_channels=144, expanded_channels=384)) for _ in range(15)],

            *[nn.Sequential(SE(base_channels=144, squeeze_ratio=0.25), MobileMQA(base_channels=144, num_heads=12)) for _ in range(6)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        
            nn.Flatten(),
        
            nn.Linear(144, 1024),
            nn.GroupNorm(1024 // 8, 1024),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x