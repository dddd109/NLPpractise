# benchmark/result.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class WorkloadInfo:
    batch_size:int
    q_len:int
    kv_len:int

    hidden_size:int

    num_q_heads:Optional[int] = None
    num_kv_heads:Optional[int] = None
    dtype:Optional[str] = None

    mode:Optional[str] = None

@dataclass
class TimingMetrics:

    mean_ms: float
    median_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    std_ms: Optional[float] = None

@dataclass
class ComputeMetrics:

    theoretical_flops: Optional[int] = None#理论值

    profiler_flops: Optional[int] = None#profiler 的FLOPS估算
    achieved_tflops: Optional[float] = None# flops/time 吞吐
    arithmetic_intensity: Optional[float] = None#flop/byte 计算强度 单位内存流量承载了多少计算
@dataclass
class MemoryMetrics:
    allocated_mb:Optional[float] = None
    reserved_md:Optional[float] = None
    peak_allocated_mb:Optional[float] = None
    peak_reserved_mb:Optional[float] = None
    kv_cache_mb:Optional[float] = None

@dataclass
class BenchmarkResult:

    name: str
    workload: WorkloadInfo
    timing: TimingMetrics
    compute: ComputeMetrics
    memory: MemoryMetrics
    metadata: dict