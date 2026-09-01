from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseKVCache(ABC):
    @abstractmethod
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        写入新的 K/V，并返回 attention 当前应该使用的 K/V。
        """
        pass

    @abstractmethod
    def get_seq_len(self, layer_idx: int) -> int:
        """返回该 layer 当前缓存了多少 token。"""
        pass

    @abstractmethod
    def reset(self):
        """清空 cache。"""
        pass