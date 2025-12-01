import torch as t
from torch import nn

### Module blocks

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import Stem, ExtraDW, FusedIB, IB, ConvNext

# The model for 224x224 ImageNet-1k classification 
class MNv4_Conv_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                Stem(3, 32, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                FusedIB(32, 32, 32, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                FusedIB(32, 64, 96, kernel_size=(3, 3), stride=2, padding=1),
            ),
            nn.Sequential(
                ExtraDW(64, 96, 192, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=2, padding_1st=2, padding_2nd=2),
                IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
                IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
                IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
                IB(96, 96, 192, kernel_size=(3, 3), stride=1, padding=1),
                ConvNext(96, 96, 384, kernel_size=(3, 3), stride=1, padding=1),
            ),
            nn.Sequential(
                ExtraDW(96, 128, 576, kernel_size_1st=(3, 3), kernel_size_2nd=(3, 3), stride=2, padding_1st=1, padding_2nd=1),
                ExtraDW(128, 128, 512, kernel_size_1st=(5, 5), kernel_size_2nd=(5, 5), stride=1, padding_1st=2, padding_2nd=2),
                IB(128, 128, 512, kernel_size=(5, 5), stride=1, padding=2),
                IB(128, 128, 384, kernel_size=(5, 5), stride=1, padding=2),
                IB(128, 128, 384, kernel_size=(3, 3), stride=1, padding=1),
                IB(128, 128, 384, kernel_size=(3, 3), stride=1, padding=1),
            ),
            nn.Sequential(
                nn.Conv2d(128, 960, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(960),
                nn.ReLU(inplace=True),
                
                nn.AvgPool2d((7, 7)),

                nn.Conv2d(960, 1280, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU(inplace=True),

                # nn.Dropout(0.3),

                nn.Conv2d(1280, 1000, kernel_size=(1, 1)),
                nn.Flatten()
            )
        )
    
    def forward(self, x: t.Tensor):
        return self.model(x)