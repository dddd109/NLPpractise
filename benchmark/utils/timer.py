#/benchmark/utils/timer.py

from typing import Callable
import torch

def benchmark_cuda(
        fn:Callable,
        warmup:int = 10,
        iters:int = 100,
) -> float:
    r"""
    return average GPU execution latency (ms)
    fn 应该是一个无参数callable
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)/iters
