import torch as t
from torch import nn

### Module blocks

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import Stem, FusedIB, ExtraDW, ConvNext

# The model for 384x384 ImageNet-1k classification
class MNv4_Conv_L(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                Stem(3, 24, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                FusedIB(24, 48, 96, kernel_size=(3, 3), stride=2, padding=1)
            ),
            nn.Sequential(
                ExtraDW(48, 96, 192, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=2, padding_1st=1, padding_2nd=2),
                ExtraDW(96, 96, 384, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=1, padding_1st=1, padding_2nd=2)
            ),
            nn.Sequential(
                ExtraDW(96, 192, 384, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=2, padding_1st=1, padding_2nd=2),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(192, 192, 768, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ConvNext(192, 192, 768, kernel_size=(3, 3), stride=1, padding=1)
            ),
            nn.Sequential(
                ExtraDW(192, 512, 768, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=2, padding_1st=2, padding_2nd=2),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(3, 3), stride=1, padding_1st=2, padding_2nd=1),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(3, 3), stride=1, padding_1st=2, padding_2nd=1),
                ExtraDW(512, 512, 2048, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2),
                ConvNext(512, 512, 2048, kernel_size=(5, 5), stride=1, padding=2)
            ),
            nn.Sequential(
                nn.Conv2d(512, 960, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(960 // 8, 960),
                nn.SiLU(inplace=True),

                nn.AvgPool2d((12, 12)),

                nn.Conv2d(960, 1280, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(1280 // 8, 1280),
                nn.SiLU(inplace=True),

                # nn.Dropout(0.2),

                nn.Conv2d(1280, 1000, kernel_size=(1, 1)),
                nn.Flatten()
            )
        )
    
    def forward(self, x: t.Tensor):
        return self.model(x)