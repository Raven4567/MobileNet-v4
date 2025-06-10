from .Stem import Stem # noqa: F401
from .FusedIB import FusedIB # noqa: F401

from .ExtraDW import ExtraDW # noqa: F401
from .ConvNext import ConvNext # noqa: F401
from .IB import IB # noqa: F401
from .FFN import FFN # noqa: F401

from .MobileMQA import MobileMQA # noqa: F401

'''
Optional block that there's no in original architecture
but that you can add in your custom architectures.
'''

from .SE import SE  # noqa: E402, F401
from .ResNeXt import ResNeXt # noqa: E402, F401