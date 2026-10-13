"""Grouped-Query Attention（GQA）。

把 ``n_head`` 个 Q 头分成若干组，每组共享一组 K/V 头，是 MHA 与 MQA 的折中：

    n_kv_head == n_head  ->  退化为 MHA
    n_kv_head == 1       ->  退化为 MQA

相比 MHA：K/V 投影的参数量、计算量、以及 KV cache 的显存量都按
``n_kv_head / n_head`` 的比例下降；
相比 MQA：保留了更多 K/V 表达能力（MQA 所有头共用一组 K/V，压缩过狠）。

参考 Ainslie et al., 2023, "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints" (arXiv:2305.13245)。

接口与 ``modules/attention/base.py`` 的 :class:`Attention` 一致::

    forward(x, position_offset=0, kv_cache=None)

参数命名刻意与 ``MQA.py`` 对齐（``q`` / ``kv`` / ``proj``），这样可以互相
``load_state_dict`` 做等价性验证（n_kv_head=n_head 时 GQA 应当与 MHA 逐元素相等）。
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache.base import BaseKVCache
from ..positional.RoPE import RoPE
from .base import Attention


@dataclass
class GQAConfig:
    """GQA 结构参数。

    Args:
        n_embd:     嵌入维度（通道数 C）
        block_size: 最大序列长度 T
        n_head:     Q 头数
        n_kv_head:  K/V 头数；None 或 0 表示退化为 MHA
        rope:       是否使用 RoPE
    """

    n_embd: int
    block_size: int
    n_head: int
    n_kv_head: Optional[int] = None
    rope: bool = True

    @property
    def kv_head(self) -> int:
        """归一化后的 KV 头数（None -> n_head）。"""
        return self.n_kv_head if self.n_kv_head else self.n_head

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class GQA(Attention):
    r"""Grouped-Query Attention。

    Args:
        config:   :class:`GQAConfig`
        layer_idx: 使用 ``kv_cache`` 时必须给出层号

    Shape:
        - Input:  ``[B, T, C]``
        - Output: ``[B, T, C]``
    """

    def __init__(self, config: GQAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"
        assert config.n_head % config.kv_head == 0, (
            "n_head 必须能被 n_kv_head 整除，Q 头才能均匀分组"
        )

        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.kv_head = config.kv_head
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.q = nn.Linear(self.n_embd, self.n_embd)
        self.kv_ch = self.head_dim * self.kv_head
        self.kv = nn.Linear(self.n_embd, 2 * self.kv_ch)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.rope = RoPE(self.head_dim) if config.rope else None

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[BaseKVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # B,T,C -> B,T,H,d -> B,H,T,d

        k, v = self.kv(x).split(self.kv_ch, dim=-1)
        k = k.view(B, T, self.kv_head, self.head_dim).transpose(1, 2)  # B,Hkv,T,d
        v = v.view(B, T, self.kv_head, self.head_dim).transpose(1, 2)

        # 有 cache 时，本段 token 的绝对位置从 cached_before 开始（RoPE 要用绝对位置）
        cached_before = 0
        if kv_cache is not None:
            assert self.layer_idx is not None, "使用 kv_cache 需要传入 layer_idx"
            cached_before = kv_cache.get_seq_len(self.layer_idx)

        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(
                T, q.device, q.dtype, position_offset + cached_before
            )
            q, k = RoPE.apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        Tk = k.size(-2)  # 带 cache 时是"已缓存 + 本次"

        if self.kv_head != self.n_head:
            # 每组 KV 头复制给同组的 Q 头用（kv_head==1 时就是 MQA 的广播）
            repeat = self.n_head // self.kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))

        # query 的第 i 个位置能看到的最大 key 下标是 cached_before + i：
        #   - 预填充(T == Tk, cached_before == 0) -> 标准下三角
        #   - 单步解码(T == 1)                    -> 全可见，因为 key 都是过去的位置
        mask = torch.ones(T, Tk, device=x.device, dtype=torch.bool).tril(
            diagonal=cached_before
        )
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = att @ v  # B,H,T,Tk @ B,H,Tk,d -> B,H,T,d

        att = att.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.proj(att)


if __name__ == "__main__":
    # 快速自检：GQA(kv_head=n_head) 应与 MHA 等价；kv_head=1 时应与 MQA 等价
    from modules.attention.MQA import MQA, MQAConfig

    torch.manual_seed(0)
    B, T, C, H, Hkv = 2, 64, 128, 8, 2
    x = torch.randn(B, T, C)

    gqa = GQA(GQAConfig(n_embd=C, block_size=T, n_head=H, n_kv_head=Hkv, rope=True))
    mha_like = GQA(GQAConfig(n_embd=C, block_size=T, n_head=H, n_kv_head=None, rope=True))
    mqa = MQA(MQAConfig(n_embd=C, block_size=T, n_head=H, kv_head=None, rope=True))

    mha_like.load_state_dict(mqa.state_dict())
    y_gqa = gqa(x)
    y_mha = mha_like(x)
    y_mqa = mqa(x)
    print("gqa out :", tuple(y_gqa.shape))
    print("与 MHA 等价性:", float((y_mha - y_mqa).abs().max()))
    print(f"参数量 GQA(Hkv={Hkv})={sum(p.numel() for p in gqa.parameters()):,} "
          f"vs 退化 MHA={sum(p.numel() for p in mha_like.parameters()):,}")
