import torch as t
from torch import nn

# Mobile Multi-Query-Attention block
class MobileMQA(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, downsample: bool = False):
        super().__init__()

        self.num_heads = num_heads
        self.D_head = in_channels // num_heads

        # Conv2D projections for query, key and value
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        
        # Output projection (W^O in the paper)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        # Spatial Reduction for K and V (depthwise separable conv: 3x3 depthwise + 1x1 pointwise)
        if downsample:
            self.SR = nn.Sequential(
                # Depthwise convolution (spatial filtering)
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=2, padding=1, 
                         groups=in_channels, bias=False),
                # Pointwise convolution (channel mixing)
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False)
            )
        else:
            self.SR = nn.Identity()

    def forward(self, x: t.Tensor):
        batch_size, channels, height, width = x.shape

        # Apply spatial reduction to K and V (if downsample=True)
        kv = self.SR(x)
        # key, value shape if downsampled: (B, C, H/2, W/2)

        # Project Q, K, V
        query = self.query_conv(x)   # (B, C, H, W)
        key = self.key_conv(kv)      # (B, C, H_kv, W_kv)
        value = self.value_conv(kv)  # (B, C, H_kv, W_kv)

        # Reshape for multi-head attention: (B, C, H, W) -> (B, num_heads, D_head, seq_len)
        query = query.view(batch_size, self.num_heads, self.D_head, height * width)
        key = key.view(batch_size, self.num_heads, self.D_head, key.size(2) * key.size(3))
        value = value.view(batch_size, self.num_heads, self.D_head, kv.size(2) * kv.size(3))

        # Permute to standard attention format: (B, num_heads, seq_len, D_head)
        query = query.permute(0, 1, 3, 2)  # (B, num_heads, H*W, D_head)
        key = key.permute(0, 1, 3, 2)      # (B, num_heads, H_kv*W_kv, D_head)
        value = value.permute(0, 1, 3, 2)  # (B, num_heads, H_kv*W_kv, D_head)

        # Compute attention scores: (B, num_heads, H*W, H_kv*W_kv)
        attention_scores = t.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.D_head ** 0.5)
        attention_scores = t.softmax(attention_scores, dim=-1)

        # Apply attention to values: (B, num_heads, H*W, D_head)
        attention_output = t.matmul(attention_scores, value)

        # Reshape back: (B, num_heads, H*W, D_head) -> (B, num_heads, D_head, H*W) -> (B, C, H, W)
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous()
        attention_output = attention_output.view(batch_size, channels, height, width)

        # Apply output projection (W^O from paper Eq. 2)
        output = self.out_conv(attention_output)

        return output