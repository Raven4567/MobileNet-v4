import unittest

import torch as t
import numpy as np

import models
from models import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestMobileNetV4(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.base_channels = 16
        self.expanded_channels = 32
        self.squeezed_channels = 4
        self.num_heads = 2
        self.num_classes = 10

    def test_stem_block(self):
        block = modules.Stem_block(self.in_channels, self.base_channels).to(device)
        output = block(t.randn(2, self.in_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 84, 84))

    def test_fused_ib_block(self):
        block = modules.FusedIB_block(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 42, 42))

    def test_uib_block(self):
        block = modules.UIB_block(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 84, 84))

    def test_extra_dw_block(self):
        block = modules.ExtraDW_block(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 84, 84))

    def test_se_block(self):
        block = modules.SE_block(self.base_channels, self.squeezed_channels).to(device)
        output = block(t.randn(2, self.base_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 84, 84))

    def test_mobile_mqa_block(self):
        block = modules.MobileMQA_block(self.base_channels, self.num_heads, downsample=False).to(device)
        output = block(t.randn(2, self.base_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 84, 84))

    # Тесты для проверки различных конфигураций модели

    def test_mobilenet_v4_small(self):
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_medium(self):
        model = models.mobilenet_v4_medium(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_large(self):
        model = models.mobilenet_v4_large(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_hybrid_large(self):
        model = models.mobilenet_v4_hybrid_large(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 84, 84).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    # Доп. тесты

    def test_invalid_input_shape(self):
        # Передаём изображение с неверным числом каналов, ожидаем ошибку
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        with self.assertRaises(RuntimeError):
            _ = model(t.randn(2, 1, 84, 84).to(device))

    def test_forward_gradient(self):
        # Проверяем, что обратное распространение (backward) работает корректно
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        model.train()
        input_tensor = t.randn(2, self.in_channels, 84, 84).to(device)
        output = model(input_tensor)
        loss = output.sum()
        model.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        nonzero_grad = any(g is not None and g.abs().sum() > 0 for g in grads)
        self.assertTrue(nonzero_grad)


if __name__ == '__main__':
    unittest.main()