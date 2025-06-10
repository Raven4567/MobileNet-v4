import unittest

import torch as t

import sys
import os
# This is the key part:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models # noqa: F401
import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestInvertedBottleckModule(unittest.TestCase):
    def setUp(self):
        self.model = modules.IB(16, 32, 64, kernel_size=(3, 3), stride=1, padding=1).to(device)
    
    def test_forward(self):
        pred = self.model(
            t.randn(1, 16, 14, 14, device=device)
        )

        self.assertEqual(pred.shape, (1, 32, 14, 14))
        self.assertEqual(pred.dtype, t.float32)

if __name__ == '__main__':
    unittest.main()