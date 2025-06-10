# MobileNet-v4 Implementation Documentation

**Languages:** [English](README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Español](README.es.md) | [中文](README.zh-CN.md)

This is my repository for implementing MobileNet-v4 architecture based on the original research paper.

# Quick start

## Installation
You've gotta install torch==2.7.0+cu118 and then run:
```
git clone https://github.com/Raven4567/MobileNet-v4
```

### MobileNet-v4 small
```python
import torch as t
from MobileNet_v4 import MNv4_Conv_S

model = MNv4_Conv_S()

pred = model(t.randn(1, 3, 224, 224))
```

### MobileNet-v4 medium
```python
import torch as t
from MobileNet_v4 import MNv4_Conv_M

model = MNv4_Conv_M()

pred = model(t.randn(1, 3, 256, 256))
```

### MobileNet-v4 large
```python
import torch as t
from MobileNet_v4 import MNv4_Conv_L

model = MNv4_Conv_L()

pred = model(t.randn(1, 3, 384, 384))
```

### MobileNet-v4 hybrid-M
(i. e. with self-attention mechanism)
```python
import torch as t
from MobileNet_v4 import MNv4_Hybrid_M

model = MNv4_Hybrid_M()

pred = model(t.randn(1, 3, 256, 256))
```

### MobileNet-v4 hybrid-large
(i. e. with self-attention mechanism)
```python
import torch as t
from MobileNet_v4 import MNv4_Hybrid_L

model = MNv4_Hybrid_L()

pred = model(t.randn(1, 3, 384, 384))
```

## Resolutions
- MNv4_Conv_S: (224x224)
- MNv4_Conv_M: (256x256)
- MNv4_Hybrid_M: (256x256)
- MNv4_Conv_L: (384x384)
- MNv4_Hybrid_L: (384x384)

## Parameters number
```
MNv4_Conv_S: 3,705,064
MNv4_Conv_M: 10,087,064
MNv4_Hybrid_M: 11,400,024
MNv4_Conv_L: 32,566,096
MNv4_Hybrid_L: 39,653,584
```

## Structure | Project map
```
MobileNet_v4/
├── models/
│   ├── MNv4_Conv_S.py
│   ├── MNv4_Conv_M.py
│   ├── MNv4_Conv_L.py
│   ├── MNv4_Hybrid_M.py
│   └── MNv4_Hybrid_L.py
├── modules/
│   ├── ConvNext.py
│   ├── ExtraDW.py
│   ├── FFN.py
│   ├── FusedIB.py
│   ├── IB.py
│   ├── MobileMQA.py
│   ├── ResNeXt.py
│   ├── SE.py
│   └── Stem.py
├── unittests/
│   ├── test_ConvNext.py
│   ├── test_ExtraDW.py
│   ├── test_FFN.py
│   ├── test_FusedIB.py
│   ├── test_IB.py
│   ├── test_MNv4_Conv_L.py
│   ├── test_MNv4_Conv_M.py
│   ├── test_MNv4_Conv_S.py
│   ├── test_MNv4_Hybrid_L.py
│   ├── test_MNv4_Hybrid_M.py
│   ├── test_MobileMQA.py
│   ├── test_ResNeXt.py
│   ├── test_SE.py
│   └── test_Stem.py
├── __init__.py
├── README.de.md
├── README.es.md
├── README.md
├── README.ru.md
├── README.zh-CN.md
└── requirements.txt
```

## References:
- Orignal research paper from Google: https://arxiv.org/pdf/2404.10518