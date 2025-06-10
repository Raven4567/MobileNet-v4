import unittest

import torch as t

import sys
import os
# This is the key part:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
import modules # noqa: F401

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestMobileNet_v4_Hybrid_L(unittest.TestCase):
    def setUp(self):
        self.model = models.MNv4_Conv_L().to(device)

    def test_forward(self):
        pred = self.model(
            t.randn(1, 3, 384, 384, device=device)
        )

        self.assertEqual(pred.shape, (1, 1000))
        self.assertEqual(pred.dtype, t.float32)

if __name__ == '__main__':
    unittest.main()