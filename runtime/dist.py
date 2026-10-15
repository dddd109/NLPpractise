"""多卡并行（DistributedDataParallel）的薄封装。

为什么单独一个文件
------------------
同一份训练代码要跑在三类后端上，后端名和设备绑定方式都不一样：

    * 昇腾 NPU   ->  hccl   （必须**先**把 local_rank 绑到 torch.npu.set_device，再建组）
    * NVIDIA GPU ->  nccl
    * CPU        ->  gloo

而且"单卡调试"和"多卡训练"必须共用同一份训练代码：没有 torchrun 环境变量时，
下面所有函数都要安全退化成单进程（world_size=1、is_main=True、集合通信变 no-op）。
这和 :mod:`runtime.device` 是同一个思路 —— 把"环境差异"关在一个文件里。

用法::

    from runtime.dist import setup, wrap_model, all_reduce_mean, is_main, cleanup

    ctx = setup()                       # 建 process group + 绑定设备（单卡时直接返回）
    model = wrap_model(model, ctx)      # DDP 包装（单进程时原样返回）
    ...
    loss = all_reduce_mean(loss)        # 多卡求平均（单进程时 no-op）
    if is_main(ctx):
        torch.save(...)                 # 只有 rank0 写盘，避免 7 个进程抢同一个文件
    cleanup(ctx)

启动（本机物理卡 3 的 Health=Critical，用 ASCEND_RT_VISIBLE_DEVICES 把它排除）::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
        torchrun --nproc_per_node=7 models/build_models.py -C configs/pretrain/pretrain_llama.yaml

⚠️ ``ASCEND_RT_VISIBLE_DEVICES`` 把物理卡**重映射**成逻辑 0..6，所以 ``LOCAL_RANK``
指的是"可见列表里的下标"，不是物理卡号。例如上面这个映射里，LOCAL_RANK=3 其实对应
物理卡 4。
"""

from __future__ import annotations

import datetime
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist

from .device import auto_backend, get_device, set_device

# 设备类型 -> 集合通信后端
BACKEND_FOR_DEVICE = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def distributed_requested() -> bool:
    """是否有 torchrun 注入的环境变量（WORLD_SIZE > 1 且有 RANK）。"""
    return _env_int("WORLD_SIZE", 1) > 1 and _env_int("RANK") is not None


def default_backend(device_type: Optional[str] = None) -> str:
    """按设备类型选集合通信后端；可用 ``RUNTIME_DIST_BACKEND`` 强制覆盖。"""
    device_type = device_type or auto_backend()
    return os.environ.get("RUNTIME_DIST_BACKEND") or BACKEND_FOR_DEVICE.get(device_type, "gloo")


def dist_ready() -> bool:
    """process group 是否已经建好（没建时所有集合通信都要跳过）。"""
    return dist.is_available() and dist.is_initialized()


@dataclass
class DistContext:
    """一次训练任务的并行上下文。单卡时 ``enabled=False``，其它字段取默认值。"""

    enabled: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def __str__(self) -> str:
        return (
            f"[dist] enabled={self.enabled} backend={self.backend} "
            f"rank={self.rank}/{self.world_size} local_rank={self.local_rank} "
            f"device={self.device}"
        )


def setup(timeout_minutes: int = 30, device_type: Optional[str] = None) -> DistContext:
    """初始化多卡环境并绑定设备；没有 torchrun 变量时返回单进程上下文。"""
    device_type = device_type or auto_backend()

    if not distributed_requested():
        device = get_device(device_type, 0)
        set_device(device)
        ctx = DistContext(False, "none", 0, 0, 1, device)
        print(ctx, flush=True)
        return ctx

    rank = _env_int("RANK", 0) or 0
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    world_size = _env_int("WORLD_SIZE", 1) or 1
    backend = default_backend(device_type)

    # ★ 昇腾要求在 init_process_group 之前先 set_device，否则 hccl 建链会失败
    device = get_device(device_type, local_rank)
    set_device(device)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )

    ctx = DistContext(True, backend, rank, local_rank, world_size, device)
    print(ctx, flush=True)
    barrier()
    return ctx


def wrap_model(
    model: torch.nn.Module,
    ctx: DistContext,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> torch.nn.Module:
    """用 DDP 包一层（单进程时原样返回）。

    注意 ``broadcast_buffers=False``：RoPE 的 ``theta_arr``、注意力 mask 这些都是
    由参数推导出来的常量，每个 rank 本来就一样，没必要每步广播（省一次通信）。
    """
    if not ctx.enabled:
        return model

    from torch.nn.parallel import DistributedDataParallel as DDP

    kwargs: dict[str, Any] = dict(
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=False,
        static_graph=static_graph,
    )
    # 昇腾/CPU 由 set_device / 进程自身决定设备，传 device_ids 会被拒绝
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
    return DDP(model, **kwargs)


@contextmanager
def no_sync(model: torch.nn.Module) -> Iterator[None]:
    """梯度累积时，非最后一个 micro-step 不通信梯度（``DDP.no_sync()`` 的包装）。"""
    if hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield


def barrier() -> None:
    if dist_ready():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """求和（单进程时原样返回）。返回的是 detach 的副本，不破坏计算图。"""
    if not dist_ready():
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """求平均（单进程时原样返回）。

    用 SUM 再除以 world_size，而不是 ``ReduceOp.AVG`` —— 后者在部分 hccl 版本上
    不一定支持。
    """
    if not dist_ready():
        return tensor
    out = all_reduce_sum(tensor)
    return out / dist.get_world_size()


def rank0_print(ctx: DistContext, *args: Any, **kwargs: Any) -> None:
    """只有主进程打印（多卡时避免 7 份重复日志）。"""
    if ctx.is_main:
        print(*args, **kwargs, flush=True)


def global_batch_tokens(
    micro_batch_size: int, seq_len: int, grad_accum_steps: int, ctx: DistContext
) -> int:
    """一次参数更新总共喂了多少 token（梯度累积 × 数据并行）。"""
    return micro_batch_size * seq_len * grad_accum_steps * ctx.world_size


def cleanup(ctx: DistContext) -> None:
    """收尾：所有进程对齐后销毁 process group。"""
    if dist_ready():
        barrier()
        dist.destroy_process_group()
    rank0_print(ctx, f"[dist] rank {ctx.rank} 正常退出")


def main() -> None:
    """自检：``python -m runtime.dist``（或 torchrun 起多进程）打印上下文并做一次 all_reduce。"""
    ctx = setup()
    x = torch.ones(2, 2) * (ctx.rank + 1)
    total = all_reduce_sum(x)
    rank0_print(ctx, f"all_reduce_sum 之和 =\n{total}")
    rank0_print(ctx, f"all_reduce_mean 之和 =\n{all_reduce_mean(x)}")
    cleanup(ctx)


__all__ = [
    "BACKEND_FOR_DEVICE",
    "DistContext",
    "all_reduce_mean",
    "all_reduce_sum",
    "barrier",
    "cleanup",
    "default_backend",
    "dist_ready",
    "distributed_requested",
    "global_batch_tokens",
    "main",
    "no_sync",
    "rank0_print",
    "setup",
    "wrap_model",
]


if __name__ == "__main__":
    main()
