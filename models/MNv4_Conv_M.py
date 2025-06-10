import torch as t
from torch import nn

### Module blocks

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import Stem, FusedIB, ExtraDW, ConvNext, FFN

# The model for 256x256 ImageNet-1k classification
class MNv4_Conv_M(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                Stem(3, 32, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                FusedIB(32, 48, 128, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                ExtraDW(48, 80, 192, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=2, padding_1st=1, padding_2nd=2),
                ExtraDW(80, 80, 160, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),    
            ),
            nn.Sequential(
                ExtraDW(80, 160, 480, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=2, padding_1st=1, padding_2nd=2),
                ExtraDW(160, 160, 640, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(160, 160, 640, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(160, 160, 640, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ExtraDW(160, 160, 960, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=1, padding_1st=1, padding_2nd=1),
                ConvNext(160, 160, 640, kernel_size=(3, 3), stride=1, padding=1),
                FFN(160, 160, 320),
                ConvNext(160, 160, 640, kernel_size=(3, 3), stride=1, padding=1),
            ),
            nn.Sequential(
                ExtraDW(160, 256, 960, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=2, padding_1st=2, padding_2nd=2),
                ExtraDW(256, 256, 1024, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                ExtraDW(256, 256, 1024, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=1, padding_1st=1, padding_2nd=2),
                ExtraDW(256, 256, 1024, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=1, padding_1st=1, padding_2nd=2),
                FFN(256, 256, 1024),
                ConvNext(256, 256, 1024, kernel_size=(3, 3), stride=1, padding=1),
                ExtraDW(256, 256, 1024, kernel_size_1st=(3, 3), kernel_size_2nd=(5, 5), stride=1, padding_1st=1, padding_2nd=2),
                ExtraDW(256, 256, 1024, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                FFN(256, 256, 1024),
                FFN(256, 256, 1024),
                ConvNext(256, 256, 512, kernel_size=(5, 5), stride=1, padding=2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 960, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(960 // 8, 960),
                nn.SiLU(inplace=True),

                nn.AvgPool2d((8, 8)),

                nn.Conv2d(960, 1280, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(1280 // 8, 1280),
                nn.SiLU(inplace=True),

                nn.Dropout(0.2),

                nn.Conv2d(1280, 1000, kernel_size=(1, 1)),
                nn.Flatten()
            ),
        )
    
    def forward(self, x: t.Tensor):
        return self.model(x)