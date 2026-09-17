# benchmark/runner.py

import torch
from typing import Optional

from .case import BenchmarkCase
from .result import BenchmarkResult,ComputeMetrics
from .utils.timer import benchmark_cuda
from .utils.memory import (
    reset_peak_memory,
    get_memory_stats
)
from .utils.flop import flops_to_tflops
from .utils.profiler import TorchProfiler

class BenchmarkRunner:
    def __init__(self,warmup:int=10,iters:int=100,
                 enable_profile:bool=False,
                 profile: Optional[TorchProfiler]=None):
        self.warmup=warmup
        self.iters = iters
        self.enable_profile = enable_profile
        self.profiler = profile or TorchProfiler()

    @torch.no_grad()
    def run(self,case:BenchmarkCase,) -> BenchmarkResult:
        # 1. reset memory statistics
        # --------------------------
        reset_peak_memory()
        # ---------------------------------
        # 2. latency
        # ---------------------------
        timing = benchmark_cuda(
            case.fn,
            warmup=self.warmup,
            iters=self.iters,
        )
        # 3. throughput
# --------------------------------------------------
        throughput = None
        if case.throughput_fn is not None:
            throughput = case.throughput_fn(
                timing,
                case.workload,
            )
        # 4. memory/自定义 memory
        kv_cache_mb = None
        if case.memory_fn is not None:
            kv_cache_mb = case.memory_fn
        
        memory = get_memory_stats(kv_cache_mb=kv_cache_mb)
        # ---------------------------------------
        # 5. FLOPs
        theoretical_flops = None
        achieved_tflops = None
        estimated_memory_bytes = None
        arithmetic_intensity = None

        if case.flop_fn is not None:
            theoretical_flops = case.flop_fn()
            achieved_tflops = flops_to_tflops(
                theoretical_flops,
                timing.mean_ms,
            )

        if case.byte_fn is not None:
            estimated_memory_bytes = case.byte_fn()

            if theoretical_flops is not None and estimated_memory_bytes > 0:
                arithmetic_intensity = (
                    theoretical_flops /
                    estimated_memory_bytes
                )
# -------------------------------------------------------
        #profiler
        profile_result = None

        if self.enable_profiler:
            profile_result = self.profiler.profile(
                case.fn,
                name=case.name,
            )
#------------------------------------------------------------------------------
        compute = ComputeMetrics(
            theoretical_flops=theoretical_flops,
            profiler_flops=(profile_result.profiler_flops
            if profile_result is not None else None),
            achieved_tflops=achieved_tflops,
            estimated_memory_bytes= estimated_memory_bytes,
            arithmetic_intensity=arithmetic_intensity,
        )
        # ---------------------------------------------------------
        return BenchmarkResult(
            name=case.name,
            workload=case.workload,
            timing=timing,
            compute=compute,
            memory=memory,
            profile=profile_result,
            metadata=case.metadata,
        )