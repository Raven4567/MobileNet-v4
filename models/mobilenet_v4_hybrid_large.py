import torch as t
from torch import nn

### Module blocks

from modules import ExtraDW, FusedIB, \
                    MobileMQA, SE, \
                    Stem, UIB

# Hybrid-Large модель – для самых тяжелых задач, требующих максимальной точности и мощности
class mobilenet_v4_hybrid_large(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(Stem(in_channels=in_channels, base_channels=384)),

            *[nn.Sequential(FusedIB(base_channels=384, expanded_channels=768)) for _ in range(4)],

            *[nn.Sequential(UIB(base_channels=384, expanded_channels=768), ExtraDW(base_channels=384, expanded_channels=768)) for _ in range(7)],

            *[nn.Sequential(SE(base_channels=384, squeeze_ratio=0.25), UIB(base_channels=384, expanded_channels=768)) for _ in range(6)],

            *[nn.Sequential(SE(base_channels=384, squeeze_ratio=0.25), MobileMQA(base_channels=384, num_heads=16)) for _ in range(6)]
        )

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        
            nn.Flatten(),
        
            nn.Linear(384, 1280),
            nn.GroupNorm(1280 // 8, 1280),
            nn.SiLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x