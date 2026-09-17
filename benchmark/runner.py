# benchmark/runner.py

import torch

from .case import BenchmarkCase
from .result import BenchmarkResult
from .utils.timer import benchmark_cuda
from .utils.memory import (
    reset_peak_memory,
    get_memory_stats
)
from .utils.flop import flops_to_tflops

class BenchmarkRunner:
    def __init__(self,warmup:int=10,iters:int=100):
        self.warmup=warmup
        self.iters = iters

    @torch.no_grad()
    def run(
        self,
        case:BenchmarkCase,
        ) -> BenchmarkResult:
        # 1. reset memory statistics
        reset_peak_memory()
        # 2. latency
        timing = benchmark_cuda(
            case.fn,
            warmup=self.warmup,
            iters=self.iters,
        )
        # 3. throughput

        throughput = (
            case.num_tokens
            / (timing.mean / 1000)
        )
        # 4. memory

        memory = get_memory_stats()
        # 5. FLOPs
        theoretical_flops = None
        tflops = None
        if case.flop_fn is not None:
            theoretical_flops = case.flop_fn()
            tflops = flops_to_tflops(
                theoretical_flops,
                timing.mean,
            )

        return BenchmarkResult(
            name=case.name,
            workload=,
            timing=timing,
            compute=,
            memory=memory,
            metadata=,
        )