import unittest

import torch as t

import sys
import os

import modules.ResNeXt
# This is the key part:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models # noqa: F401
import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

'''
Test of optional ResNeXt block that I'd added just cuz I wanted it here
as optional module.
'''
class TestResNeXtModule(unittest.TestCase):
    def setUp(self):
        self.model = modules.ResNeXt(
            in_channels=32, 
            out_channels=64,
            bottleneck_channels=8,
            groups=4
        ).to(device)
    
    def test_forward(self):
        pred = self.model(
            t.randn(1, 32, 14, 14, device=device)
        )

        self.assertEqual(pred.shape, (1, 64, 14, 14))
        self.assertEqual(pred.dtype, t.float32)

if __name__ == '__main__':
    unittest.main()