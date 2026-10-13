"""MLA 专用 KV cache：只缓存低秩 latent。

普通注意力把 K/V 原样存下来（每 token ``2 * n_head * head_dim`` 个标量）；
MLA 只需要缓存 ``latent = [c_kv, k_pe]``（每 token ``kv_lora_rank + qk_rope_head_dim``）。
这才是 MLA 省显存的关键，所以这里是个**单槽** cache。

和 :class:`BaseKVCache` 的双槽签名（``update(layer_idx, key, value)``）略有出入：

* ``value`` 参数保留但可选（默认 None）；
* 返回值第二位固定是 ``None``，因为 MLA 本来就没有独立的 V。

调用方（``modules/attention/MLA.py``）的写法是::

    latent, _ = kv_cache.update(layer_idx, latent)
"""

from typing import List, Optional, Tuple

import torch

from .base import BaseKVCache


class MLAKVCache(BaseKVCache):
    """单槽 latent cache。

    形状约定：``latent`` 是 ``[B, T, latent_dim]``（T 在第 1 维），
    与标准 KV cache 的 ``[B, H, T, C]`` 不同。
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.latents: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        latent: torch.Tensor,
        value: Optional[torch.Tensor] = None,  # 兼容基类签名，实际不用
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """写入新的 latent 并返回该层当前可用（已缓存 + 本次）的 latent。"""
        if self.latents[layer_idx] is None:
            self.latents[layer_idx] = latent
        else:
            self.latents[layer_idx] = torch.cat([self.latents[layer_idx], latent], dim=1)
        return self.latents[layer_idx], None

    def get_seq_len(self, layer_idx: int) -> int:
        latent = self.latents[layer_idx]
        return int(latent.size(1)) if latent is not None else 0

    def reset(self) -> None:
        self.latents = [None] * self.num_layers

    # ------------------------------------------------------------------
    def memory_bytes(self, dtype_size: int = 2) -> int:
        """当前缓存占多少字节（默认按 bf16/fp16 = 2 字节算），用于和标准 KV cache 对比。"""
        total = 0
        for latent in self.latents:
            if latent is not None:
                total += latent.numel() * dtype_size
        return total


if __name__ == "__main__":
    B, T, D = 1, 5, 64
    cache = MLAKVCache(num_layers=2)

    latent, _ = cache.update(0, torch.randn(B, 5, D))
    print("prefill seq_len =", cache.get_seq_len(0))  # 5

    for _ in range(3):
        cache.update(0, torch.randn(B, 1, D))
    print("decode  seq_len =", cache.get_seq_len(0))  # 8

    cache.reset()
    print("reset   seq_len =", cache.get_seq_len(0))  # 0
