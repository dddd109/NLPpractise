"""因果掩码缓存。

所有自回归注意力都需要一个 (q_len, kv_len) 的因果掩码：

    预填充：T == kv_len、offset == 0      -> 标准下三角
    带 cache 逐步解码：T == 1、offset > 0 -> 全可见（key 全是过去的位置）

朴素写法每次 forward 都 ``torch.ones(T, Tk).tril(...)`` 建一遍，训练时
(T, Tk, offset) 每步都一样，属于纯浪费（T=1024 时每次一个 1MB 的 bool 张量 +
两个 kernel，还要乘上 n_layer 层）。这里按 key 缓存最后一份即可。

约定：``True`` = 保留（允许 attention），与 ``F.scaled_dot_product_attention``
的 bool mask 语义一致；手工实现里用 ``masked_fill(~mask, -inf)``。
"""

from typing import Any, Optional

import torch


class CausalMaskCache:
    """只缓存最近一次的掩码（够用且不会无限增长）。"""

    __slots__ = ("_key", "_mask")

    def __init__(self) -> None:
        self._key: Optional[Any] = None
        self._mask: Optional[torch.Tensor] = None

    def get(self, q_len: int, kv_len: int, offset: int, device: torch.device) -> torch.Tensor:
        """返回 [q_len, kv_len] 的 bool 掩码，True = 保留。

        ``offset`` = 本次 query 之前已经缓存了多少个 key。于是 query i 能看到的最大
        key 下标是 ``i + offset``，即 ``j <= i + offset``，对应
        ``tril(diagonal=offset)``。
        """
        key = (q_len, kv_len, offset, device.type)
        if key != self._key:
            self._mask = torch.ones(q_len, kv_len, device=device, dtype=torch.bool).tril(
                diagonal=offset
            )
            self._key = key
        return self._mask

    def clear(self) -> None:
        self._key, self._mask = None, None


def neg_inf(dtype: torch.dtype) -> float:
    """softmax 之前用来"屏蔽"位置的极小值。

    ⚠️ 不要用 ``float("-inf")``：昇腾上 ``masked_fill(-inf)`` 会引入 ``isneginf``
    算子，而该算子在 NPU 上**不支持**（日志里那条
    ``kernel [isneginf.out] is not supported by NPU currently. Now this kernel is
    running on CPU``），于是每次都回退 CPU 并同步一次。12 层 × 前向+反向 × 每个
    micro-batch 都要来一遍，实测能把单步时间拖慢一个数量级。

    改用 dtype 能表示的最小有限值（fp16 是 -65504），softmax 之后的数值结果与 -inf
    完全一致，而且全程留在 NPU 上。
    """
    return torch.finfo(dtype).min
