#/benchmark/utils/timer.py
import numpy as np
from typing import Callable
import torch
from ..result import TimingMetrics

def get_percent(array,percent):
    iters = len(array)
    if iters*percent%1==0: 
        out = (array[iters*percent] + array[iters*percent+1])/2
    out = array[int(iters*percent)+1]
    return out

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
        time.qppend(start.elapsed_time(end))
    time=np.array(time.sort())
    mean = time.mean()
    median = get_percent(time,0.5)
    p50 = median
    p90 = get_percent(time,0.9)
    p99 = get_percent(time,0.99)
    std = time.std()
    return TimingMetrics(mean_ms=mean,
                  median_ms=median,
                  p50_ms=p50,
                  p90_ms=p90,
                  p99_ms=p99,
                  std_ms=std,)
    # return start.elapsed_time(end)/iters
