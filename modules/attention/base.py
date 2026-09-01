from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Attention(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        kv_cache=None,
    ) -> torch.Tensor:
        pass