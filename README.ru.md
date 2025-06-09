# MobileNet-v4 Документация к реализации

**Языки:** [English](README.md) | [Русский](README.ru.md) | [Español](README.es.md) | [Deutsch](README.de.md) | [中文](README.zh-CN.md)

---

# Описание
Когда я работал над моим проектом связанным с обучением с подкреплением используя изображения в качестве состояний, я столкнулся с проблемой что моя нейросеть не обучается достаточно эффективно и с тем, что моя нейросеть работает довольно медленно, и поскольку я не нашёл реализацию MobileNet в `torchvision` за исключением третьей версии и ниже, я решил написать мою собственную реализацию MobileNet-v4. В целом это моя практическая работа для лучшего понимания особенностей архитектур, как написать мобильную и эффективную нейросеть, и как организовать проект так, чтобы он не превратился в мусор собранный в одном файле.

# Быстрый старт

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

# Структура проекта
Вот карта структуры проекта:
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
|   ├── ExtraDW.py          # Extra Depthwise блок
|   ├── FusedIB.py          # Fused Inverted Bottleneck блок
│   ├── MobileMQA.py        # Mobile Multi-Query Attention блок
│   ├── SE.py               # Squeeze and Excitation блок (внимание по каналам)
│   ├── Stem.py             # Stem блок для начальной обработки
│   └── UIB.py              # Universal Inverted Bottleneck блок
├── LICENSE.txt          # Файл лицензии
├── Main.py              # Главный файл для обучения модели
├── README.md            # Файл документации
└── test_MobileNet_v4.py # Юниттесты для проверки работоспособности моделей и модулей
```

Поскольку здесь не так много файлов, я могу описать *каждый*.

`Stem.py` - Это *первый слой каждой конфигурации*, потому что он обрабатывает изображения с формой: `(batch_size, кол-во каналов, высота, ширина)` в `(batch_size, кол-во фильтров, высота, ширина)`.

`ExtraDW.py` - "*Extra Depthwise*", это свёрточный блок с дополнительными вычислениями (а именно, дополнительный блок с классической свёрткой kernel_size=3, stride=1, padding-1) 

`FusedIB` - "*Fused Inverted BottleNeck*", это *свёртка с kernel_size=(3, 3), stride=2, padding=1*, и дополнительной pointwise (т. е. свёртка с размером ядра 1 на 1). Сконструирован для сжатия входных изображений пока они всё ещё большие после stem блока, каждый блок снижает разрешение *в два раза* (например, 512x512 -> 256x256 -> 128x128 -> 64x64).  

`UIB.py` - "*Universal Inverted Bottleneck*", это *основной блок в архитектурах моделей*, он использует свёртку с большим ядром (аж 5x5) и двойной pointwise свёрткой (т. е. 1x1 свёртка).

`MobileMQA.py` - "*Mobile Multi Quety Attention*", сконструирован как альтернатива более требовательному MHA (Multi Head Attention), в отличии от MHA, `MobileMQA` использует один единственный query, key, value слои (просто 1x1 свёртки) для *всех голов*, вместо личного слоя для каждой головы.

`SE.py` - "*Squeeze and Excitation*", это *один из self-attention слоёв*, он сжимает изобраежние до 1x1 пикселей с `torch.nn.AdaptiveAvgPool2d((1, 1))`, затем сжимает фильтры с pointwise свёртками (1x1 свёртка), затем возвращает их обратно к исходному кол-ву фильтров, и применяет сигмоду к выходу, чтобы получить коэффициенты от 0 до 1 и умножает на входные данные. Это дешёвый и хороший механизм само-внимания, но в отличии от `MQA`, `SE` применяет полученные коэффициенты **к каналам**, а не к пикселям на прямую.

# Общая структура моделей
Структура моделей *модульная* и (по моему мнению) *масштабируемая*.

В этих моделях, архитектура разделена на несколько стадий. Первая стадия состоит из одного единственного stem слоя, который обрабатывает входные изображения в *чёрно-белом* или *цветном* (т. е. один или три канала), конвертируя их в карту признаков с различным кол-вом фильтров (например, 24, 32, или 64).

Вторая стадия обычно содержит от одного до четырёх *Fused Inverted Bottleneck* (`FusedIB`) блоков. Эти блоки *снижают разрешение* и *сжимают* изображения, тем самым снижая вычислительную стоимость (обычная практика в свёрточных нейронных сетях).

В третьей стадии, *Universal Inverted Bottleneck* (`UIB`) и *Extra Depthwise* (`ExtraDW`) блоки использованы *чтобы извлечь средние и локальные признакми*, такие как края объектов.

Чётвёртая стадия использует `UIB` и *Squeeze and Excitation* (`SE`), чтобы захватывать взвешенные признаки с `SE` блоков *применя a легко-весный канальный механизм внимания*.

Наконец-то, последняя стадия *извлекает глобальные признаки* используя комбинацию *SE* и *Mobile Multi-Query Attention* (`MobileMQA`) блоков. Здесь, `SE` создаёт коэффициенты от 0 до 1 для фильтров, эффективно снижая роль менее полезных признаков, пока `MobileMQA` применяет внимания к рахличным частям изображения.

# Configurations

- Справка: Параметры конфигураций расчитаны для моделей инициализированных с параметрами `in_channels=3` и `num_classes`=1000, т. е. для *ImageNet*.

---------------------------------------------------------------------------------------
| `MobileNet_v4_nano` | `MobileNet_v4_small` | `MobileNet_v4_medium` | `MobileNet_v4_large` | `MobileNet_v4_hybrid_large` |
---------------------------------------------------------------------------------------
| *590 344 параметров.* | *1 303 904 параметров.* | *2 276 384 параметров.* | *9 740 680 параметров.* | *35 886 888 параметров.* |

# References:
...