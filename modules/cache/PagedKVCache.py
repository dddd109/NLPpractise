import torch
from typing import Optional

from .base import BaseKVCache


class PagedKVCache(BaseKVCache):
    """
    Educational paged KV cache.

    Layout:
        [B, H, num_blocks, block_size, D]

    Internally, each logical sequence is stored as a list of
    fixed-size physical blocks.
    """

    def __init__(
        self,
        num_layers: int,
        block_size: int = 16,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        self.num_layers = num_layers
        self.block_size = block_size

        self.device = device
        self.dtype = dtype

        self.key_blocks = [
            [] for _ in range(num_layers)
        ]
        self.value_blocks = [
            [] for _ in range(num_layers)
        ]

        self.seq_lens = [
            0 for _ in range(num_layers)
        ]

    def _alloc_block(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        return torch.empty(
            (
                batch_size,
                num_heads,
                self.block_size,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"invalid layer_idx={layer_idx}"
            )

        if key.shape != value.shape:
            raise ValueError(
                "key and value must have the same shape"
            )

        if key.ndim != 4:
            raise ValueError(
                "expected [B, H, T, D]"
            )

        B, H, T_new, D = key.shape

        if self.device is None:
            self.device = key.device

        if self.dtype is None:
            self.dtype = key.dtype

        pos = self.seq_lens[layer_idx]

        offset = 0

        while offset < T_new:

            logical_block_idx = pos // self.block_size
            block_offset = pos % self.block_size

            # allocate a new physical block
            if logical_block_idx == len(
                self.key_blocks[layer_idx]
            ):
                self.key_blocks[layer_idx].append(
                    self._alloc_block(
                        B,
                        H,
                        D,
                        key.device,
                        key.dtype,
                    )
                )

                self.value_blocks[layer_idx].append(
                    self._alloc_block(
                        B,
                        H,
                        D,
                        value.device,
                        value.dtype,
                    )
                )

            k_block = self.key_blocks[layer_idx][
                logical_block_idx
            ]

            v_block = self.value_blocks[layer_idx][
                logical_block_idx
            ]

            write_size = min(
                T_new - offset,
                self.block_size - block_offset,
            )

            k_block[
                :,
                :,
                block_offset:block_offset + write_size,
                :,
            ].copy_(
                key[
                    :,
                    :,
                    offset:offset + write_size,
                    :,
                ]
            )

            v_block[
                :,
                :,
                block_offset:block_offset + write_size,
                :,
            ].copy_(
                value[
                    :,
                    :,
                    offset:offset + write_size,
                    :,
                ]
            )

            pos += write_size
            offset += write_size

        self.seq_lens[layer_idx] = pos

        return self.get_kv(layer_idx)

    def get_seq_len(self, layer_idx: int) -> int:
        return self.seq_lens[layer_idx]

    def get_kv(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_len = self.seq_lens[layer_idx]

        if seq_len == 0:
            raise RuntimeError(
                "cache is empty"
            )

        keys = self.key_blocks[layer_idx]
        values = self.value_blocks[layer_idx]

        k_parts = []
        v_parts = []

        remaining = seq_len

        for k_block, v_block in zip(keys, values):

            length = min(
                remaining,
                self.block_size,
            )

            k_parts.append(
                k_block[:, :, :length, :]
            )

            v_parts.append(
                v_block[:, :, :length, :]
            )

            remaining -= length

            if remaining <= 0:
                break

        return (
            torch.cat(k_parts, dim=-2),
            torch.cat(v_parts, dim=-2),
        )

    def reset(self):
        self.key_blocks = [
            [] for _ in range(self.num_layers)
        ]

        self.value_blocks = [
            [] for _ in range(self.num_layers)
        ]

        self.seq_lens = [
            0 for _ in range(self.num_layers)
        ]