import torch as t
from torch import nn

# Mobile Multi-Query-Attention block
class MobileMQA(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, downsample: bool = False):
        super().__init__()

        self.num_heads = num_heads

        # Conv2D для query, key и value
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        if downsample:
            self.downsample_key_conv = nn.Sequential(
                # Depthwise conv for downsampling
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=2, padding=1, groups=in_channels, bias=False),
                nn.GroupNorm(in_channels // 8, in_channels), # Assuming in_channels is divisible by 8
                nn.SiLU(inplace=True),
                # Pointwise conv
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(in_channels // 8, in_channels),
                nn.SiLU(inplace=True)
            )
            self.downsample_value_conv = nn.Sequential(
                # Depthwise conv for downsampling
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=2, padding=1, groups=in_channels, bias=False),
                nn.GroupNorm(in_channels // 8, in_channels),
                nn.SiLU(inplace=True),
                # Pointwise conv
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(in_channels // 8, in_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.downsample_key_conv = nn.Identity()
            self.downsample_value_conv = nn.Identity()

    def forward(self, x: t.Tensor):
        batch_size, channels, height, width = x.shape
        D_head = channels // self.num_heads

        # Получаем запросы (query), ключи (key) и значения (value)
        query = self.query_conv(x)  # Shape: (B, C, H, W)
        key = self.key_conv(x)      # Shape: (B, C, H, W)
        value = self.value_conv(x)  # Shape: (B, C, H, W)

        # Downsample if downsample is enabled, or just get back the same values
        key, value = self.downsample_key_conv(key), self.downsample_value_conv(value)
        # key, value shape if downsampled: (B, C, H/2, W/2)

        # Разделяем запросы на несколько голов
        # Original view: (B, NH, D_h, N_spatial_elements)
        query = query.view(batch_size, self.num_heads, D_head, -1) # (B, NH, D_h, N_q) where N_q = H*W
        key = key.view(batch_size, self.num_heads, D_head, -1)     # (B, NH, D_h, N_kv)
        value = value.view(batch_size, self.num_heads, D_head, -1) # (B, NH, D_h, N_kv)

        # Permute to standard attention format: (B, NH, N_sequence, D_h)
        query = query.permute(0, 1, 3, 2) # (B, NH, N_q, D_h)
        key = key.permute(0, 1, 3, 2)     # (B, NH, N_kv, D_h)
        value = value.permute(0, 1, 3, 2)   # (B, NH, N_kv, D_h)

        # Матричное умножение для запроса и ключа (изменённый размер)
        # (B, NH, N_q, D_h) @ (B, NH, D_h, N_kv) -> (B, NH, N_q, N_kv)
        attention_scores = t.matmul(query, key.transpose(-2, -1))

        # Scale attention scores
        attention_scores = attention_scores / (D_head ** 0.5)

        # Применяем softmax для нормализации весов
        attention_scores = t.softmax(attention_scores, dim=-1)

        # Применение внимания к значениям
        # (B, NH, N_q, N_kv) @ (B, NH, N_kv, D_h) -> (B, NH, N_q, D_h)
        attention_output = t.matmul(attention_scores, value)

        # Реконструируем выход в исходный размер
        # (B, NH, N_q, D_h) -> (B, NH, D_h, N_q) via permute
        # -> (B, C, H, W) via view
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous()
        attention_output = attention_output.view(batch_size, channels, height, width)

        return attention_output