import torch as t
from torch import nn

### Module blocks

from modules import ExtraDW, FusedIB, \
                    MobileMQA, SE, \
                    Stem, UIB

# Large модель – для задач с высокой точностью на CIFAR10, CIFAR100
class mobilenet_v4_large(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem(in_channels, base_channels=224)),

            *[nn.Sequential(FusedIB(base_channels=224, expanded_channels=512)) for _ in range(3)],

            *[nn.Sequential(UIB(base_channels=224, expanded_channels=512), ExtraDW(base_channels=224, expanded_channels=512)) for _ in range(21)],

            *[nn.Sequential(SE(base_channels=224, squeeze_ratio=0.25), UIB(base_channels=224, expanded_channels=512)) for _ in range(14)],

            *[nn.Sequential(SE(base_channels=224, squeeze_ratio=0.25), MobileMQA(base_channels=224, num_heads=14)) for _ in range(5)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        
            nn.Flatten(),
        
            nn.Linear(224, 1280),
            nn.GroupNorm(1280 // 8, 1280),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x