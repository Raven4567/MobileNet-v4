import torch as t
from torch import nn

### Module blocks

from modules import ExtraDW, FusedIB, \
                    MobileMQA, SE, \
                    Stem, UIB

# Small модель – подходит для MNIST, KMNIST, FashionMNIST, простых Atari-сред
class mobilenet_v4_small(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem(in_channels, base_channels=112)),

            *[nn.Sequential(FusedIB(base_channels=112, expanded_channels=224)) for _ in range(3)],

            *[nn.Sequential(UIB(base_channels=112, expanded_channels=224), ExtraDW(base_channels=112, expanded_channels=224)) for _ in range(11)],
            
            *[nn.Sequential(SE(base_channels=112, squeeze_ratio=0.25), UIB(base_channels=112, expanded_channels=224)) for _ in range(7)],
             
            *[nn.Sequential(SE(base_channels=112, squeeze_ratio=0.25), MobileMQA(base_channels=112, num_heads=8)) for _ in range(3)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        
            nn.Flatten(),
        
            nn.Linear(112, 1024),
            nn.GroupNorm(1024 // 8, 1024),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x