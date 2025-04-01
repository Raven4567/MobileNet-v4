from torch import nn

### Module blocks

from .modules import ExtraDW_block, FusedIB_block, \
                    MobileMQA_block, SE_block, \
                    Stem_block, UIB_block

class mobilenet_v4_custom(nn.Module):
    def __init__(
            self, in_channels: int=3, num_classes: int=1000,
            base_channels: int=16, expanded_channels: int=32, squeezed_channels: int=4,
            num_heads: int=2, downsample: bool=True,
            hidden_dim_classifier: int=64, dropout: float=0.2,
            FusedIB_blocks: int=1, UIB_and_ExtraDW_blocks: int=1, SE_and_UIB_blocks: int=1, SE_and_MobileMQA_blocks: int=1
        ):
        super().__init__()
        self.features = nn.Sequential(
            Stem_block(in_channels, base_channels=base_channels),

            *[FusedIB_block(base_channels=base_channels, expanded_channels=expanded_channels) for _ in range(FusedIB_blocks)],

            *[nn.Sequential(UIB_block(base_channels=base_channels, expanded_channels=expanded_channels), ExtraDW_block(base_channels=base_channels, expanded_channels=expanded_channels)) for _ in range(UIB_and_ExtraDW_blocks)],
            
            *[nn.Sequential(SE_block(base_channels=base_channels, squeezed_channels=squeezed_channels), UIB_block(base_channels=base_channels, expanded_channels=expanded_channels)) for _ in range(SE_and_UIB_blocks)],
             
            *[nn.Sequential(SE_block(base_channels=base_channels, squeezed_channels=squeezed_channels), MobileMQA_block(base_channels=base_channels, num_heads=num_heads, downsample=downsample)) for _ in range(SE_and_MobileMQA_blocks)]
        )

        #self.lstm = nn.LSTM(input_size=base_channels, hidden_size=base_channels, num_layers=1, batch_first=True)

        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            
            nn.Linear(base_channels, hidden_dim_classifier),
            nn.BatchNorm1d(hidden_dim_classifier),
            nn.SiLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim_classifier, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classificator(x)

        return x