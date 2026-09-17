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
from .utils.profiler import TorchProfiler

class BenchmarkRunner:
    def __init__(self,warmup:int=10,iters:int=100):
        self.warmup=warmup
        self.iters = iters

    @torch.no_grad()
    def run(self,case:BenchmarkCase,
            enable_profile:bool=False) -> BenchmarkResult:
        # 1. reset memory statistics
        reset_peak_memory()
        # 2. latency
        timing = benchmark_cuda(
            case.fn,
            warmup=self.warmup,
            iters=self.iters,
        )
        # 3. throughput

        throughput = case.num_tokens/ (timing.mean / 1000)
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
        #profiler
        if enable_profile:
            profiler = TorchProfiler()
            prof = profiler.profile(case.fn,memory)
            print(
            prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
)
        return BenchmarkResult(
            name=case.name,
            workload=,
            timing=timing,
            compute=,
            memory=memory,
            profile=prof,
            metadata=,
        )