"""MoE（专家混合）前馈层。

.. warning::
   ``BasicMoE`` 是早期草稿，**当前跑不起来**，别用它做实验：

   1. ``self.experts[w.argmax(dim=-1)]`` 用 tensor 去索引 ``nn.ModuleList``，会抛
      TypeError（``ModuleList.__getitem__`` 只接受 int，而且语义上也没法"按 token
      取不同专家"）；
   2. ``argmax`` 不可导，router 拿不到梯度，等于从未被训练。

   要能训练 / 能基准测试的 MoE 请用 :class:`MoEFFN`。
"""

from .MoE import BasicMoE
from .MoEFFN import MoEFFN, MoEFFNConfig

__all__ = ["BasicMoE", "MoEFFN", "MoEFFNConfig"]
