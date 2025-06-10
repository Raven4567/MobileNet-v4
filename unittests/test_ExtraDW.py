import unittest

import torch as t

import sys
import os
# This is the key part:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models # noqa: F401
import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestExtraDepthWiseConvModule(unittest.TestCase):
    def setUp(self):
        self.model = modules.ExtraDW(
            in_channels=16, 
            out_channels=32, 
            expanded_channels=64, 
            kernel_size_1st=(3, 3), 
            kernel_size_2nd=(5, 5), 
            stride=1, 
            padding_1st=1, 
            padding_2nd=2
        ).to(device)
    
    def test_forward(self):
        pred = self.model(
            t.randn(1, 16, 14, 14, device=device)
        )

        self.assertEqual(pred.shape, (1, 32, 14, 14))
        self.assertEqual(pred.dtype, t.float32)

if __name__ == '__main__':
    unittest.main()