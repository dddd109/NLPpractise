#banchmark/case.py

from dataclasses import dataclass, field
from typing import Callable,Any,Optional
from .result import WorkloadInfo

@dataclass
class BenchmarkCase:
    # name:str
    # fn:Callable[[],Any]
    # num_tokens:int
    # flop_fn:Optional[Callable[[],int]]=None
    name: str
    fn: Callable[[], Any]
    workload: WorkloadInfo
    num_tokens: int = 0
    flop_fn: Optional[Callable[[], int]] = None
    memory_fn: Optional[Callable[[], float]] = None
    metadata: dict = field(default_factory=dict)