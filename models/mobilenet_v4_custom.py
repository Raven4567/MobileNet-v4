import torch as t
from torch import nn

### Module blocks

from modules import ExtraDW, FusedIB, \
                    MobileMQA, SE, \
                    Stem, UIB

class mobilenet_v4_custom(nn.Module):
    def __init__(
            self, in_channels: int=3, num_classes: int=1000,
            base_channels: int=16, expanded_channels: int=32, squeeze_ratio: int=0.25,
            num_heads: int=2, hidden_dim_classifier: int=64, dropout: float=0.2,
            FusedIBs: int=1, UIB_and_ExtraDWs: int=1, SE_and_UIBs: int=1, SE_and_MobileMQAs: int=1
        ):
        super().__init__()
        self.features = nn.Sequential(
            Stem(in_channels, base_channels=base_channels),

            *[FusedIB(base_channels=base_channels, expanded_channels=expanded_channels) for _ in range(FusedIBs)],

            *[nn.Sequential(UIB(base_channels=base_channels, expanded_channels=expanded_channels), ExtraDW(base_channels=base_channels, expanded_channels=expanded_channels)) for _ in range(UIB_and_ExtraDWs)],
            
            *[nn.Sequential(SE(base_channels=base_channels, squeeze_ratio=squeeze_ratio), UIB(base_channels=base_channels, expanded_channels=expanded_channels)) for _ in range(SE_and_UIBs)],
             
            *[nn.Sequential(SE(base_channels=base_channels, squeeze_ratio=squeeze_ratio), MobileMQA(base_channels=base_channels, num_heads=num_heads)) for _ in range(SE_and_MobileMQAs)]
        )

        #self.lstm = nn.LSTM(input_size=base_channels, hidden_size=base_channels, num_layers=1, batch_first=True)

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            
            nn.Linear(base_channels, hidden_dim_classifier),
            nn.GroupNorm(hidden_dim_classifier // 8, hidden_dim_classifier),
            nn.SiLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim_classifier, num_classes)
        )

    def forward(self, x: t.Tensor):
        x = self.features(x)
        x = self.classificator(x)

        return x