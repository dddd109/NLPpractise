#banchmark/case.py

from dataclasses import dataclass
from typing import Callable,Any,Optional

@dataclass
class BenchmarkCase:
    name:str
    fn:Callable[[],Any]
    num_tokens:int
    flop_fn:Optional[Callable[[],int]]=None