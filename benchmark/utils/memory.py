# benchmark/utils/memory.py

"""
reset_peak_memory()
y = model(x)
torch.cuda.synchronize()
memory = get_memory_stats()
"""
import torch


def reset_peak_memory():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_memory_stats():
    """
    单位：MB
    """
    return {
        "allocated_mb": (
            torch.cuda.memory_allocated() / 1024**2
        ),
        "reserved_mb": (
            torch.cuda.memory_reserved() / 1024**2
        ),
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved() / 1024**2
        ),
    }