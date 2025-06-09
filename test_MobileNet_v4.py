import unittest

import torch as t

import models
import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestMobileNetV4(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.base_channels = 16
        self.expanded_channels = 32
        self.squeezed_channels = 4
        self.num_heads = 2
        self.num_classes = 10

    def test_stem(self):
        block = modules.Stem(self.in_channels, self.base_channels).to(device)
        output = block(t.randn(2, self.in_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 64, 64))

    def test_fused_ib(self):
        block = modules.FusedIB(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 32, 32))

    def test_uib(self):
        block = modules.UIB(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 64, 64))

    def test_extra_dw(self):
        block = modules.ExtraDW(self.base_channels, self.expanded_channels).to(device)
        output = block(t.randn(2, self.base_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 64, 64))

    def test_se(self):
        block = modules.SE(self.base_channels, self.squeezed_channels).to(device)
        output = block(t.randn(2, self.base_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 64, 64))

    def test_mobile_mqa(self):
        block = modules.MobileMQA(self.base_channels, self.num_heads).to(device)
        output = block(t.randn(2, self.base_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.base_channels, 64, 64))

    # Тесты для проверки различных конфигураций модели

    def test_mobilenet_v4_small(self):
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_medium(self):
        model = models.mobilenet_v4_medium(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_large(self):
        model = models.mobilenet_v4_large(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_mobilenet_v4_hybrid_large(self):
        model = models.mobilenet_v4_hybrid_large(self.in_channels, self.num_classes).to(device)
        output = model(t.randn(2, self.in_channels, 64, 64).to(device))
        self.assertEqual(output.shape, (2, self.num_classes))

    # Доп. тесты

    def test_invalid_input_shape(self):
        # Передаём изображение с неверным числом каналов, ожидаем ошибку
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        with self.assertRaises(RuntimeError):
            _ = model(t.randn(2, 1, 64, 64).to(device))

    def test_forward_gradient(self):
        # Проверяем, что обратное распространение (backward) работает корректно
        model = models.mobilenet_v4_small(self.in_channels, self.num_classes).to(device)
        model.train()
        input_tensor = t.randn(2, self.in_channels, 64, 64).to(device)
        output = model(input_tensor)
        loss = output.sum()
        model.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        nonzero_grad = any(g is not None and g.abs().sum() > 0 for g in grads)
        self.assertTrue(nonzero_grad)


if __name__ == '__main__':
    unittest.main()