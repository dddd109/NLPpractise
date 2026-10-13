"""Multi-head Latent Attention（MLA）。

DeepSeek-V2 提出的注意力形式：**不直接缓存 K/V**，而是把 K/V 信息压进一个低秩
latent 向量，用的时候再上投影回 K/V；位置编码单独走一条"解耦"的路径。

    压缩：   c_kv   = x @ W_dkv                # [B,T,r]         r = kv_lora_rank
    解码键： k_pe   = x @ W_kr                 # [B,T,d_rope]    所有头共享
    上投影： [k_nope, v] = c_kv @ W_ukv        # 每头 d_nope + d_v

推理时只需要缓存 ``latent = [c_kv, k_pe]``，每 token 的缓存量从

    2 * n_head * head_dim
降到
    kv_lora_rank + qk_rope_head_dim

以 n_head=16、head_dim=64、r=512、d_rope=64 为例：2048 -> 576，约省 3.6 倍。
（这正是本项目 README 里"研究 KV cache 对长上下文的影响"最直接的抓手。）

参考: DeepSeek-V2, arXiv:2405.04434, 第 2.1 节。

接口与 ``modules/attention/base.py`` 的 :class:`Attention` 一致::

    forward(x, position_offset=0, kv_cache=None)

注意：MLA 的 cache 是**单槽**的（只存 latent，没有独立的 V），要用
:class:`modules.cache.MLAKVCache`。
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
class MLAConfig:
    """MLA 结构参数。

    Args:
        n_embd:            嵌入维度（通道数 C）
        block_size:        最大序列长度 T
        n_head:            头数（Q 和输出都是这个头数）
        kv_lora_rank:      K/V 压缩后的低秩维度 r
        qk_nope_head_dim:  每个头里"不做旋转"的那部分 q/k 维度
        qk_rope_head_dim:  每个头里"做旋转"的那部分 q/k 维度（也是共享 k_pe 的维度）
        v_head_dim:        每个头的 value 维度
        rope:              是否使用 RoPE（关了就是不旋转的朴素版本，用来做消融）
    """

    n_embd: int
    block_size: int
    n_head: int
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 64
    rope: bool = True

    @property
    def qk_head_dim(self) -> int:
        """拼起来之后每个头 q/k 的完整维度。"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """每个 token 需要缓存的长度。"""
        return self.kv_lora_rank + self.qk_rope_head_dim


class MLA(Attention):
    r"""Multi-head Latent Attention。

    Shape:
        - Input:  ``[B, T, C]``
        - Output: ``[B, T, C]``
    """

    def __init__(self, config: MLAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        c = config
        assert c.n_head > 0, "n_head 必须为正"
        assert c.qk_rope_head_dim % 2 == 0, "RoPE 要求 qk_rope_head_dim 为偶数"

        self.config = c
        self.n_embd = c.n_embd
        self.n_head = c.n_head
        self.d_nope = c.qk_nope_head_dim
        self.d_rope = c.qk_rope_head_dim
        self.d_v = c.v_head_dim
        self.layer_idx = layer_idx

        # 压缩到 latent / 解耦的 rope key
        self.w_dkv = nn.Linear(c.n_embd, c.kv_lora_rank, bias=False)
        self.w_kr = nn.Linear(c.n_embd, c.qk_rope_head_dim, bias=False)
        # latent 上投影回 [k_nope, v]
        self.w_ukv = nn.Linear(
            c.kv_lora_rank, c.n_head * (c.qk_nope_head_dim + c.v_head_dim), bias=False
        )
        # 查询
        self.w_dq = nn.Linear(c.n_embd, c.n_head * c.qk_head_dim, bias=False)
        # 输出
        self.w_o = nn.Linear(c.n_head * c.v_head_dim, c.n_embd, bias=False)

        self.rope = RoPE(c.qk_rope_head_dim) if c.rope else None

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[BaseKVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        head = self.n_head

        # ---- 查询 -----------------------------------------------------------
        q = self.w_dq(x).view(B, T, head, self.config.qk_head_dim).transpose(1, 2)
        # B,H,T,qk_head_dim
        q_nope, q_rope = q[..., : self.d_nope], q[..., self.d_nope :]

        # ---- 低秩 latent（真正要缓存的东西）----------------------------------
        c_kv = self.w_dkv(x)                          # B,T,r
        k_pe = self.w_kr(x)                           # B,T,d_rope（所有头共享）
        latent = torch.cat([c_kv, k_pe], dim=-1)      # B,T,r+d_rope

        cached_before = 0
        if kv_cache is not None:
            assert self.layer_idx is not None, "使用 kv_cache 需要传入 layer_idx"
            cached_before = kv_cache.get_seq_len(self.layer_idx)
            # 单槽 cache：只存 latent，返回值的第二位是 None
            latent, _ = kv_cache.update(self.layer_idx, latent)

        Tk = latent.size(1)                           # 已缓存 + 本次

        # ---- 上投影回 K_nope / V（缓存里存的是没旋转的 k_pe，这里统一旋转）----
        c_kv, k_pe = torch.split(
            latent, [self.config.kv_lora_rank, self.d_rope], dim=-1
        )
        kv = self.w_ukv(c_kv).view(B, Tk, head, self.d_nope + self.d_v)
        k_nope, v = kv[..., : self.d_nope], kv[..., self.d_nope :]
        k_nope = k_nope.transpose(1, 2)               # B,H,Tk,d_nope
        v = v.transpose(1, 2)                         # B,H,Tk,d_v

        # ---- RoPE：只作用在 rope 那一段，且 k_pe 是所有头共享的 ----------------
        k_pe = k_pe.view(B, 1, Tk, self.d_rope)
        if self.rope is not None:
            # q 的绝对位置从 cached_before 开始；k 覆盖 0..Tk-1 的全部位置
            q_rope = self.rope(q_rope, offset=position_offset + cached_before)
            k_pe = self.rope(k_pe, offset=position_offset)

        k = torch.cat([k_nope, k_pe.expand(B, head, Tk, self.d_rope)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)       # B,H,T,qk_head_dim

        # ---- 注意力 ---------------------------------------------------------
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.config.qk_head_dim))

        # 与 GQA 相同的掩码约定：query i 能看到的 key 下标上限是 cached_before + i
        mask = torch.ones(T, Tk, device=x.device, dtype=torch.bool).tril(
            diagonal=cached_before
        )
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v                                 # B,H,T,d_v

        out = out.transpose(1, 2).contiguous().view(B, T, head * self.d_v)
        return self.w_o(out)

    # ------------------------------------------------------------------
    # 便于和标准 MHA / MQA 做对比：每 token 的 KV cache 大小
    # ------------------------------------------------------------------
    def kv_cache_per_token(self) -> int:
        """每个 token 缓存多少个标量（MLA）。"""
        return self.config.latent_dim

    @staticmethod
    def kv_cache_per_token_baseline(n_head: int, head_dim: int) -> int:
        """标准 MHA 每个 token 缓存多少个标量（K 和 V 各 n_head*head_dim）。"""
        return 2 * n_head * head_dim


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H = 2, 64, 256, 8
    cfg = MLAConfig(
        n_embd=C,
        block_size=T,
        n_head=H,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
    )
    mla = MLA(cfg)
    x = torch.randn(B, T, C)
    y = mla(x)
    print("out:", tuple(y.shape))
    print(
        "每 token 缓存标量数: MLA=%d, 同配置 MHA=%d, 省 %.1f 倍"
        % (
            mla.kv_cache_per_token(),
            MLA.kv_cache_per_token_baseline(H, cfg.qk_nope_head_dim),
            MLA.kv_cache_per_token_baseline(H, cfg.qk_nope_head_dim) / mla.kv_cache_per_token(),
        )
    )
