"""注意力模块的统一出口。

所有实现都遵守 :class:`modules.attention.base.Attention` 的接口::

    forward(x, position_offset=0, kv_cache=None)

新增架构的步骤：在本目录加一个实现文件，再在这里导出一行即可。
"""

from .base import Attention
from .GQA import GQA, GQAConfig
from .MHA import MultiHeadAttention
from .MLA import MLA, MLAConfig
from .MQA import MQA, MQAConfig

__all__ = [
    "Attention",
    "MultiHeadAttention",
    "MQA",
    "MQAConfig",
    "GQA",
    "GQAConfig",
    "MLA",
    "MLAConfig",
]
