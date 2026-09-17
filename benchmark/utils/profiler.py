# benchmark/profiler/profiler.py

import torch
from dataclasses import dataclass
from torch.profiler import profiler,ProfilerAction
from typing import Optional
from ..result import ProfileResult

class TorchProfiler:
    def __init__(self,
                 record_shapes=True,
                 profile_memory=True,
                 with_flops=True,
                 ):
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        
    @torch.no_grad()
    def profile(self,fn):
        #预热
        fn()
        torch.cuda.synchronize()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        ) as prof:
            fn()
        return prof