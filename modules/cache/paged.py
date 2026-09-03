import torch
from .base import BaseKVCache
from typing import Tuple,Optional
class PagedKVCache(BaseKVCache):
    def __init__(self):
        pass

    def _alloc_block(self):
        pass

    def get_seq_len(self, layer_idx):
        pass

    def update(self, layer_idx, key, value):
        pass

    def reset(self):
        pass
