# benchmark/result.py

from dataclasses import dataclass,field
from typing import Optional

@dataclass
class WorkloadInfo:
    name: str

    batch_size: Optional[int] = None

    input_shapes: Optional[dict] = None
    output_shapes: Optional[dict] = None

    dtype: Optional[str] = None
    device: Optional[str] = None

    metadata: dict = field(default_factory=dict)

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
    profiler_flops: Optional[int] = None#profiler 的FLOPS估算 torch的profiler
    achieved_tflops: Optional[float] = None#  理论/profiler的FLOPS
    arithmetic_intensity: Optional[float] = None#flop/byte 计算强度 单位内存流量承载了多少计算
    
@dataclass
class MemoryMetrics:
    allocated_mb:Optional[float] = None
    reserved_mb:Optional[float] = None
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
    
@dataclass
class ProfileResult:

    total_cuda_time_us: Optional[float] = None
    total_cpu_time_us: Optional[float] = None

    profiler_flops: Optional[int] = None
    operator_table: Optional[str] = None
    trace_path: Optional[str] = None

    metadata: dict = field(default_factory=dict)