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
    def __init__(self,
                 warmup:int=10,
                 iters:int=100):
        self.warmup=warmup
        self.iters = iters

    @torch.no_grad()
    def run(
        self,
        case:BenchmarkResult,
    ) -> BenchmarkResult:
        # 1. reset memory statistics
        reset_peak_memory()
        # 2. latency

        latency_ms = benchmark_cuda(
            case.fn,
            warmup=self.warmup,
            iters=self.iters,
        )
        # 3. throughput

        throughput = (
            case.num_tokens
            / (latency_ms / 1000)
        )
        # 4. memory

        peak_memory_mb = get_memory_stats()
        # 5. FLOPs
        theoretical_flops = None
        tflops = None
        if case.flop_fn is not None:
            theoretical_flops = case.flop_fn()
            tflops = flops_to_tflops(
                theoretical_flops,
                latency_ms,
            )

        return BenchmarkResult(
            name=case.name,
            latency_ms=latency_ms,
            throughput=throughput,
            theoretical_flop=theoretical_flops,
            tflops=tflops,
            peak_memory_mb=peak_memory_mb["peak_allocated_mb"],
        )