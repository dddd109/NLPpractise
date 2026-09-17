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
    memory_fn: Optional[Callable[[], float]] = None
    metadata: dict = field(default_factory=dict)
    throughput_fn: Optional[
    Callable[[TimingMetrics, WorkloadInfo], float]
]