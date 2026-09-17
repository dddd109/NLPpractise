#/benchmark/utils/timer.py
import numpy as np
from typing import Callable
import torch
from ..result import TimingMetrics

def benchmark_cuda(
        fn:Callable,
        warmup:int = 20,
        iters:int = 100,
) -> TimingMetrics:
    r"""
    return TimingMetrics 
    fn 应该是一个无参数callable
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    time =[]

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        time.append(start.elapsed_time(end))
    time=np.array(time)
    time.sort()
    mean = time.mean()
    p50 = np.percentile(time, 50)
    median = p50
    p90 = np.percentile(time, 90)
    p99 = np.percentile(time, 99)
    std = time.std()
    return TimingMetrics(mean_ms=mean,
                  median_ms=median,
                  p50_ms=p50,
                  p90_ms=p90,
                  p99_ms=p99,
                  std_ms=std,)
    # return start.elapsed_time(end)/iters
