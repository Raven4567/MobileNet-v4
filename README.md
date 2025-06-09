# MobileNet-v4 Implementation Documentation

**Languages:** [English](README.md) | [Русский](README.ru.md) | [Español](README.es.md) | [Deutsch](README.de.md) | [中文](README.zh-CN.md)

---

# Description
When I was working on my project related to reinforcement learning using images as states, I've encountered the problem that my network doesn't learn effectively and that my network works quite slowly, and since I haven't found MobileNet implementation in torchvision except for the third version and lower, I decided to write my own implementation of MobileNet-v4. Overall this is my practical work to understand better the architectural specifics, how to write good mobile and efficient network, and how to organize the project so that it does not become a mess is writen just in one file.

# Quick start

`MobileNet_v4.models.mobilenet_v4_nano.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_nano

model = mobilenet_v4_nano(in_channels=3, num_classes=1000)

features = model(
    t.randn(4, 3, 512, 512)
)
```

`MobileNet_v4.models.mobilenet_v4_small.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_small

model = mobilenet_v4_small(in_channels=3, num_classes=1000)

features = model(
    t.randn(4, 3, 512, 512)
)
```

`MobileNet_v4.models.mobilenet_v4_medium.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_medium

model = mobilenet_v4_medium(in_channels=3, num_classes=1000)

features = model(
    t.randn(4, 3, 512, 512)
)
```

`MobileNet_v4.models.mobilenet_v4_large.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_large

model = mobilenet_v4_large(in_channels=3, num_classes=1000)

features = model(
    t.randn(4, 3, 512, 512)
)
```

`MobileNet_v4.models.mobilenet_v4_hybrid_large.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_hybrid_large

model = mobilenet_v4_hybrid_large(in_channels=3, num_classes=1000)

features = model(
    t.randn(4, 3, 512, 512)
)
```

`MobileNet_v4.models.mobilenet_v4_custom.py`:
```python
import torch as t
from MobileNet_v4.models import mobilenet_v4_custom

model = mobilenet_v4_custom(
    in_channels=3, num_classes=1000,
    base_channels=16, expanded_channels=32, squeeze_ratio=0.25,
    num_heads=2, hidden_dim_classifier=64, dropout=0.2,
    FusedIBs=1, UIB_and_ExtraDWs=1, 
    SE_and_UIBs=1, SE_and_MobileMQAs=1
)

features = model(
    t.randn(4, 3, 512, 512)
)
```

# Project structure
Here is the map of project:
```
MobileNet_v4/
├── models/
│   ├── __init__.py
│   ├── mobilenet_v4_custom.py
│   ├── mobilenet_v4_hybrid_large.py
│   ├── mobilenet_v4_large.py
│   ├── mobilenet_v4_medium.py
│   ├── mobilenet_v4_small.py
│   └── mobilenet_v4_nano.py
├── modules/
|   ├── __init__.py
|   ├── ExtraDW.py          # Extra Depthwise convolutional block
|   ├── FusedIB.py          # Fused Inverted Bottleneck block
│   ├── MobileMQA.py        # Mobile Multi-Query Attention block
│   ├── SE.py               # Squeeze and Excitation block (channel attention)
│   ├── Stem.py             # Stem block for initial processing
│   └── UIB.py              # Universal Inverted Bottleneck block
├── LICENSE.txt          # License file
├── Main.py              # Main file for training of a model
├── README.md            # Documentation file
└── test_MobileNet_v4.py # Unittests for checking out models and modules workability
```

Because of here isn't a lot of files I can describe *each one*.

`Stem.py` - This is *first layer of any model configuration*, because it processes images with a shape: `(batch_size, num_channels, height, width)` into `(batch_size, num_filters, height, width)`.

`ExtraDW.py` - "*Extra Depthwise*", this is convolutional block with additional computations (and exactly, additional convolutional block with classic convolutional kernel_size=3, stride=1, padding=1) 

`FusedIB` - "*Fused Inverted BottleNeck*", this is *convolution with kernel_size=(3, 3), stride=2, padding=1*, and additional pointwise (i. e. convolution with kernel size 1x1). Is made for squeezing of input images while they are still big after the stem block, each block reduces the resolution *at two times* (e. g. 512x512 -> 256x256 -> 128x128 -> 64x64).  

`UIB.py` - "*Universal Inverted Bottleneck*", this is *general block in the model architectures*, it uses convolution with the big kernel size (whole 5x5) and twice pointwise convolution (i. e. 1x1 convolution).

`MobileMQA.py` - "*Mobile Multi Quety Attention*", is designed as alternative for more demanding MHA (Multi Head Attention), unlikely for MHA, `MobileMQA` uses the one only query, key, value layers (just 1x1 convolutions) for *all heads*, instead of an own layer for each head.

`SE.py` - "*Squeeze and Excitation*", this is *one of the self-attention layers*, it compresses image to 1x1 pixels with `torch.nn.AdaptiveAvgPool2d((1, 1))`, then squeezes filters with pointwise convolutions (1x1 convolution), then returns it back to origin number of channels, and applies sigmoid to output to get coefficients from 0 to 1 and multiply them to input data. This is cheap and good self-attention mechanism, but unlikely for `MQA`, `SE` applies are got coefficients **to channels** and not to pixels directly.

# Overall models' structure 
The models' structure *is module* and (as I think) *scalable*.

In these models, the architecture is divided into several stages. The first stage consists of a single stem layer that processes input images in either *grayscale* or *RGB* (i.e. one or three channels), converting them into a feature map with a varying number of filters (for example, 24, 32, or 64).

The second stage typically comprises one to four *Fused Inverted Bottleneck* (`FusedIB`) blocks. These blocks *downsample* and *compress* the images, thereby reducing computational cost (a common practice in convolutional neural networks).

In the third stage, *Universal Inverted Bottleneck* (`UIB`) and *Extra Depthwise* (`ExtraDW`) blocks are used *to extract medium and local features*, such as object edges.

The fourth stage utilises `UIB` and *Squeeze and Excitation* (`SE`) blocks to capture refined features, with the `SE` block *applying a light-weight channel attention mechanism*.

Finally, the last stage *extracts global features* using a combination of *SE* and *Mobile Multi-Query Attention* (`MobileMQA`) blocks. Here, the `SE` block generates coefficients between 0 and 1 for the filters, effectively reducing less useful features, while the `MobileMQA` block applies attention to interrelate different parts of the image.

# Configurations

- Reference: Parameters of configurations were calculated for models initialised with `in_channels=3` and `num_classes`=1000, i. e. for *ImageNet*.

`MobileNet_v4_nano` - The smallest network configuration (unless you create your custom configurate with even fewer parameters amount). Designed for simple tasks such as basic classification of MNIST, CIFAR, and other datasets, or simple video environments (more correct, environments where images - are states). *Contains 590 344 parameters.*

`MobileNet_v4_small` - A heavier network is designed for moderate tasks such as Atari environments and image classification. It includes additional blocks, filters, attention heads and a larger classifier. *Contains 1 303 904 parameters.*

`MobileNet_v4_medium` - A balanced network for medium-sized datasets and tasks with more complex features and greater class variation (e.g. CIFAR100, Food101). *Contains 2 276 384 parameters.*

`MobileNet_v4_large` - A heavy network suited to complex tasks, featuring larger filters, numerous attention heads and an even bigger classifier. *Contains 9 740 680 parameters.*

`MobileNet_v4_hybrid_large` - The most complex configuration, employing the largest filters, maximum block count and 32 attention heads. Designed for high-end ImageNet tasks. *Contains 35 886 888 parameters.*

# References:
- https://arxiv.org/abs/2404.10518
- https://arxiv.org/pdf/2404.10518