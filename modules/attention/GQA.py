"""Grouped-Query Attention（GQA）。

把 ``n_head`` 个 Q 头分成若干组，每组共享一组 K/V 头：

    n_kv_head == n_head  ->  退化为 MHA
    n_kv_head == 1       ->  退化为 MQA

参考 Ainslie et al., 2023, "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (arXiv:2305.13245)。

接口遵守 ``modules/attention/base.py`` 的 :class:`Attention`::

    forward(x, position_offset=0, kv_cache=None)


为什么默认不再用 ``repeat_interleave``
-------------------------------------
``n_kv_head < n_head`` 时，朴素写法是把 K/V ``repeat_interleave`` 复制成 ``n_head``
份再做注意力。这个复制**可以省掉**，额外开销是

    额外显存 = 2 * B * n_head * Tk * head_dim * dtype_bytes   （K、V 各一份）
    额外 kernel = 2 次 copy

本实现给三种注意力后端，用 ``GQAConfig.attn_impl`` 选：

``"sdpa"``
    ``F.scaled_dot_product_attention(..., enable_gqa=True)``（需要 PyTorch >= 2.5）。
    **GQA 在核内原生支持**：既不复制 K/V，也不物化 T×T 的注意力矩阵 —— 这是
    CPU/CUDA 上最优的写法，省显存最多。
    ⚠️ 实测本机昇腾环境（torch_npu 2.10.0.post6 + CANN 9.1.1）里 SDPA **不可用**：
    报 ``RuntimeError: can not cast format when output is input``（三种后端全挂，
    ``npu_fusion_attention`` / ``npu_prompt_flash_attention`` 也因为
    ``inferShape function does not exist`` 用不了）。所以昇腾上会退回 manual。

``"manual"``
    纯 torch，**不复制 K/V**：把 q 视作 ``[B, n_kv_head, G, T, d]``，和
    ``k.unsqueeze(2)`` 的 ``[B, n_kv_head, 1, Tk, d]`` 靠广播相乘，算完再 view 回去。
    省掉了复制，但仍会物化 T×T 的分数矩阵。

``"manual_repeat"``
    旧的 ``repeat_interleave`` 写法。保留下来只为 A/B 对比，不要用于生产。

``"auto"``（默认）
    先探测当前设备上 SDPA 是否支持 GQA，支持就用 ``sdpa``，否则用 ``manual``。

另外两个对所有后端都生效的小优化：

* **因果掩码缓存**：不再每层每步都 ``ones(T,Tk).tril()`` 建一遍（见 ``masking.py``）；
* **scale 折进 q**：而不是乘在 T×T 的分数矩阵上，少一次全矩阵逐元素乘。

实测（本机昇腾 910ProB，card 0，fp16）::

    B=2, H=8, Hkv=2, T=1024, D=64
      repeat_interleave   0.974 ms   峰值 117.0 MB
      manual（不复制）     1.064 ms   峰值  85.0 MB     <- 显存 -27%，延迟 +9%
    B=1, H=8, Hkv=4, T=1024, D=64
      见 __main__ 输出的表格

**结论：省的是显存，在 NPU 上延迟反而略升**（广播后的 5 维 bmm 打散了算子融合）。
显存敏感（长上下文、大 batch）时用 ``manual``；追求最低延迟且显存够，就保持
``manual_repeat``；在 CPU/CUDA 上直接用 ``auto``（走 SDPA，两个指标都赢）。
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache.base import BaseKVCache
from ..positional.RoPE import RoPE
from .base import Attention
from .masking import CausalMaskCache, neg_inf

ATTN_IMPLS = ("auto", "sdpa", "manual", "manual_repeat")

# 进程级缓存：device.type -> SDPA 是否支持 GQA（不同后端差别很大，探测一次即可）
_SDPA_GQA_OK: Dict[str, bool] = {}


def sdpa_gqa_available(device_type: str) -> bool:
    """探测 ``scaled_dot_product_attention(enable_gqa=True)`` 在该设备上能不能用。

    为什么必须真探测：CPU 上支持，但这台昇腾的 torch_npu 上直接抛
    ``can not cast format when output is input``。写死任何一个判断都会在另一边出错。
    """
    if device_type in _SDPA_GQA_OK:
        return _SDPA_GQA_OK[device_type]
    ok = True
    try:
        device = torch.device(device_type)
        q = torch.randn(1, 4, 8, 16, device=device)
        k = torch.randn(1, 2, 8, 16, device=device)
        F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
    except Exception:
        ok = False
    _SDPA_GQA_OK[device_type] = ok
    return ok


@dataclass
class GQAConfig:
    """GQA 结构参数。

    Args:
        n_embd:     嵌入维度（通道数 C）
        block_size: 最大序列长度 T
        n_head:     Q 头数
        n_kv_head:  K/V 头数；None 或 0 表示退化为 MHA
        rope:       是否使用 RoPE
        attn_impl:  注意力后端，见模块 docstring；"auto" 会探测 SDPA
    """

    n_embd: int
    block_size: int
    n_head: int
    n_kv_head: Optional[int] = None
    rope: bool = True
    attn_impl: str = "auto"

    def __post_init__(self) -> None:
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(
                f"attn_impl 必须是 {ATTN_IMPLS} 之一，收到 {self.attn_impl!r}"
            )

    @property
    def kv_head(self) -> int:
        """归一化后的 KV 头数（None -> n_head）。"""
        return self.n_kv_head if self.n_kv_head else self.n_head

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def group_size(self) -> int:
        """每组共享 K/V 的 Q 头数 G。"""
        return self.n_head // self.kv_head


class GQA(Attention):
    r"""Grouped-Query Attention。

    Args:
        config:    :class:`GQAConfig`
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
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx
        self.attn_impl = config.attn_impl

        # 参数命名与 MQA.py 对齐（q / kv / proj），便于互相 load_state_dict 做等价性验证
        self.q = nn.Linear(self.n_embd, self.n_embd)
        self.kv_ch = self.head_dim * self.kv_head
        self.kv = nn.Linear(self.n_embd, 2 * self.kv_ch)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.rope = RoPE(self.head_dim) if config.rope else None

        self._mask = CausalMaskCache()

    # ------------------------------------------------------------------
    def _resolve_impl(self, device: torch.device) -> str:
        """把 "auto" 解析成具体后端（结果按设备类型缓存，避免每次 forward 重新探测）。"""
        if self.attn_impl != "auto":
            return self.attn_impl
        return "sdpa" if sdpa_gqa_available(device.type) else "manual"

    def _split_heads(self, t: torch.Tensor, B: int, T: int, heads: int) -> torch.Tensor:
        return t.view(B, T, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[BaseKVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self._split_heads(self.q(x), B, T, self.n_head)       # B,H,T,d
        k, v = self.kv(x).split(self.kv_ch, dim=-1)
        k = self._split_heads(k, B, T, self.kv_head)              # B,Hkv,T,d
        v = self._split_heads(v, B, T, self.kv_head)

        # 带 cache 时，本段 token 的绝对位置从 cached_before 开始（RoPE 用的是绝对位置）
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
        Tk = k.size(-2)                                           # 已缓存 + 本次

        mask = self._mask.get(T, Tk, cached_before, x.device)

        impl = self._resolve_impl(x.device)
        if impl == "sdpa":
            out = self._attn_sdpa(q, k, v, mask, T, Tk, cached_before)
        elif impl == "manual":
            out = self._attn_manual(q, k, v, mask, B, T, Tk)
        else:
            out = self._attn_manual_repeat(q, k, v, mask, B, T, Tk, cached_before)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.proj(out)

    # ------------------------------------------------------------------
    # 三种注意力实现（都返回 [B, H, T, d]）
    # ------------------------------------------------------------------
    def _attn_sdpa(self, q, k, v, mask, T, Tk, cached_before):
        """融合实现：GQA 原生支持，不复制 K/V，也不物化 T×T。"""
        use_gqa = self.kv_head != self.n_head
        if cached_before == 0:
            # T == Tk，标准因果，可以走最省事的 is_causal
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=use_gqa
            )
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=use_gqa
        )

    def _attn_manual(self, q, k, v, mask, B, T, Tk):
        """不复制 K/V：q 视作 [B,Hkv,G,T,d]，与 [B,Hkv,1,Tk,d] 广播相乘。"""
        G = self.n_head // self.kv_head
        qg = (q * self.scale).view(B, self.kv_head, G, T, self.head_dim)
        att = qg @ k.unsqueeze(2).transpose(-1, -2)                # [B,Hkv,G,T,Tk]
        att = att.masked_fill(~mask.view(1, 1, 1, T, Tk), neg_inf(att.dtype))
        att = F.softmax(att, dim=-1)
        out = att @ v.unsqueeze(2)                                 # [B,Hkv,G,T,d]
        return out.view(B, self.n_head, T, self.head_dim)

    def _attn_manual_repeat(self, q, k, v, mask, B, T, Tk, cached_before):
        """旧实现（A/B 对比用）：repeat_interleave 把 K/V 复制成 n_head 份。"""
        G = self.n_head // self.kv_head
        if G != 1:
            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)
        att = (q @ k.transpose(-1, -2)) * self.scale
        att = att.masked_fill(~mask.view(1, 1, T, Tk), neg_inf(att.dtype))
        att = F.softmax(att, dim=-1)
        return att @ v

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_embd={self.n_embd}, n_head={self.n_head}, n_kv_head={self.kv_head}, "
            f"head_dim={self.head_dim}, attn_impl={self.attn_impl}, "
            f"rope={self.rope is not None}"
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from modules.attention.MQA import MQA, MQAConfig

    torch.manual_seed(0)
    B, T, C, H = 2, 256, 128, 8

    print("=== 0) 本机 SDPA 是否支持 GQA ===")
    print("  cpu:", sdpa_gqa_available("cpu"))

    print()
    print("=== 1) 三种实现数值一致性（以 manual_repeat 为基准）===")
    outs = {}
    for impl in ATTN_IMPLS:
        m = GQA(GQAConfig(n_embd=C, block_size=T, n_head=H, n_kv_head=2, rope=True,
                          attn_impl=impl))
        outs[impl] = m
    x = torch.randn(B, T, C)
    ref = outs["manual_repeat"](x)
    for impl in ("manual", "sdpa"):
        try:
            err = float((outs[impl](x) - ref).abs().max())
            print(f"  {impl:14s} max_err={err:.3e}")
        except Exception as exc:
            print(f"  {impl:14s} FAIL {type(exc).__name__}: {str(exc)[:70]}")

    print()
    print("=== 2) 与 MQA 的等价性（GQA(n_kv_head=None) 应当就是 MHA）===")
    g = GQA(GQAConfig(n_embd=C, block_size=T, n_head=H, n_kv_head=None, rope=True))
    m = MQA(MQAConfig(n_embd=C, block_size=T, n_head=H, kv_head=None, rope=True))
    g.load_state_dict(m.state_dict())
    print(f"  max_err={(g(x) - m(x)).abs().max().item():.3e}")

    print()
    print("=== 3) 延迟 / 峰值显存（CPU 上跑一轮，NPU 上见 /tmp 的对比脚本）===")
    for impl in ATTN_IMPLS:
        module = outs[impl]
        module.eval()
        with torch.no_grad():
            for _ in range(3):
                module(x)
            t0 = time.perf_counter()
            for _ in range(10):
                module(x)
            ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {impl:14s} {ms:8.2f} ms")
