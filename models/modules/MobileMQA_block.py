import torch as t
from torch import nn

# Mobile Multi-Query-Attention block
class MobileMQA_block(nn.Module):
    def __init__(self, base_channels: int, num_heads: int, downsample: bool=False):
        super().__init__()

        self.num_heads = num_heads
        self.downsample = downsample

        # Conv2D для query, key и value
        self.query_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        # Если требуется даунсемплинг
        if self.downsample:
            self.downsample_query = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
            self.downsample_kv = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Получаем запросы (query), ключи (key) и значения (value)
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Если нужен даунсемплинг
        if self.downsample:
            query = self.downsample_query(query)
            key = self.downsample_kv(key)
            value = self.downsample_kv(value)

        # Разделяем запросы на несколько голов
        batch_size, channels, height, width = query.shape
        query = query.view(batch_size, self.num_heads, channels // self.num_heads, height, width)
        key = key.view(batch_size, self.num_heads, channels // self.num_heads, height, width)
        value = value.view(batch_size, self.num_heads, channels // self.num_heads, height, width)

        # Приводим размерность key для транспонирования
        key = key.view(batch_size, self.num_heads, channels // self.num_heads, -1)
        key = key.transpose(2, 3)

        # Матричное умножение для запроса и ключа (изменённый размер)
        attention_scores = t.matmul(query.view(batch_size, self.num_heads, channels // self.num_heads, height * width), key)

        # Применяем softmax для нормализации весов
        attention_scores = t.softmax(attention_scores, dim=-1)

        # Применение внимания к значениям
        value = value.view(batch_size, self.num_heads, channels // self.num_heads, -1)
        attention_output = t.matmul(attention_scores, value)

        # Реконструируем выход в исходный размер
        attention_output = attention_output.view(batch_size, channels, height, width)

        return attention_output