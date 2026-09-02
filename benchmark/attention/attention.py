import torch
from typing import Optional
from ...modules.cache import BaseKVCache
@torch.no_grad()
def bench_attn(
        model:torch.Module,
        warmup:int=10,
        iters:int=50,
        device:Optional[torch.device]=None
):
    B,T,C = 32,256,1024
    x = torch.randn((B,T,C),device=device)

    for  _ in range(warmup):
        __ = model(x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = model(x)
    end.record()
    end.synchronize()
    elapsed_ms = end.elapsed_time(start)
    return elapsed_ms/iters

@torch.no_grad()
def bench_decode(
    model:torch.Module,
    cache:BaseKVCache,
    n_embd:int,
    predill:int=1024,
    decode:int=256,
    batch:Optional[int]=1,
    warmup:int = 10,
    iters:int = 50,
    device:Optional[torch.device]=None
):
    model.eval()
    