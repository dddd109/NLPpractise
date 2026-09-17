#banchmark/case.py

from dataclasses import dataclass, field
from typing import Callable,Any,Optional
from .result import WorkloadInfo,TimingMetrics

@dataclass
class BenchmarkCase:
    # name:str
    # fn:Callable[[],Any]
    # num_tokens:int
    # flop_fn:Optional[Callable[[],int]]=None
    name: str
    fn: Callable[[], Any]
    workload: WorkloadInfo
    
    flop_fn: Optional[Callable[[], int]] = None
    memory_fn: Optional[Callable[[], float]] = None # workload-specific memory capacity, e.g. KV cache 表示内存占用
    byte_fn:Optional[Callable[[],int]] = None # 给arithmetric intensity 估计数据移动
    metadata: dict = field(default_factory=dict)
    throughput_fn: Optional[
    Callable[[TimingMetrics, WorkloadInfo], float]
] = None