# benchmark/profiler/profiler.py

import torch
from dataclasses import dataclass
from torch.profiler import profile,ProfilerActivity,DeviceType
from typing import Optional,Callable
from ..result import ProfileResult
from pathlib import Path

class TorchProfiler:
    def __init__(self,
                 record_shapes:bool=True,
                 profile_memory:bool=True,
                 with_flops:bool=True,
                 with_stack:bool=False, #增加开销
                 warmup_times:int =3,
                 repeat_times: int =1,
                 trace_save_dir: Optional[str] = None,
                 ):
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        self.with_stack = with_stack
        self.warmup_times = warmup_times
        self.repeat_times = repeat_times
        self.trace_save_dir = trace_save_dir
        
    @torch.no_grad()
    def profile(self,fn:Callable,name:str='profile') -> ProfileResult: 
        #预热
        activitise = [ProfilerActivity.CPU]
        
        if torch.cuda.is_available():
            activitise.append(ProfilerActivity.CUDA)
            torch.cuda.empty_cache() # 清空显存碎片，减少memory统计干扰
            
        # 预热
        for _ in range(self.warmup_times):
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
        with profile(
            activities=activitise,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
            with_stack=self.with_stack
        ) as prof:
            fn()
            
        if torch.cuda.is_available():
                torch.cuda.synchronize()

        key_avg = prof.key_averages(group_by_input_shape=self.record_shapes)    

        # 填充指标
        total_cpu_time_us = key_avg.total_average().self_cpu_time_total

        total_cuda_time_us = sum(
            evt.self_device_time_total
            for evt in prof.events()
            if evt.device_type == DeviceType.CUDA
        )

        # FLOPs求和：profiler里每个算子的flops，累加 ，仅是torch内部支持的
        total_flops = 0
        for evt in key_avg:
            if hasattr(evt, "flops") and evt.flops is not None:
                total_flops += evt.flops
        operator_table = key_avg.table(sort_by="self_device_time_total", row_limit=-1)
        
        profiler_memory_sum_mb = None
        if self.profile_memory:
            mem_bytes = 0
            for evt in key_avg:
                if hasattr(evt,"self_device_memory_bytes"):
                    mem_bytes +=evt.self_device_memory_bytes
            profiler_memory_sum_mb = mem_bytes/(1024**2)
        
        # 导出trace
        trace_path: Optional[str] = None
        if self.trace_save_dir is not None:
            save_dir = Path(self.trace_save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            trace_file = save_dir / f"{name}.json"
            prof.export_chrome_trace(str(trace_file))
            trace_path = str(trace_file)

        # metadata保存附加信息
        metadata = {
            "warmup_times": self.warmup_times,
            "repeat_times": self.repeat_times,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_flops": self.with_flops,
            "with_stack": self.with_stack,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "profile_total_tensor_mb":profiler_memory_sum_mb, #profile算子内存总和
        }
            
        return  ProfileResult(
            total_cuda_time_us=total_cuda_time_us,
            total_cpu_time_us=total_cpu_time_us,
            profiler_flops=total_flops,
            operator_table=operator_table,
            trace_path=trace_path,
            metadata=metadata,
        )
