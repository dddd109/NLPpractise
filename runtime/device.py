"""硬件抽象层：昇腾 NPU / CUDA GPU / CPU 的统一入口。

为什么单独一个文件
------------------
这个项目要在两类硬件上跑：

    * 开发机    NVIDIA GPU   -> torch.cuda.*
    * 训练机    昇腾 NPU      -> torch.npu.*      (Ascend 910ProB)

两边 API 不一样，而且昇腾侧还有一个很容易踩的坑：

    如果 shell 没有 ``source /usr/local/Ascend/ascend-toolkit/set_env.sh``，
    那么 ``import torch`` 会直接抛
        RuntimeError: Failed to load the backend extension: torch_npu
    因为 torch 启动时会自动尝试加载已经安装的 torch_npu，而加载失败是致命错误。

所以这里做了两件事：

1. 在 ``import torch`` **之前**判断 CANN 环境在不在；不在就先关掉设备后端自动加载
   （``TORCH_DEVICE_BACKEND_AUTOLOAD=0``），让 ``import torch`` 至少能以 CPU 模式成功，
   真正要用 NPU 时再用 :func:`require_npu` 给出**可操作的报错**；
2. 把设备相关操作收敛成少数几个函数（get_device / synchronize / manual_seed /
   autocast ...），上层代码里不再直接出现 ``torch.cuda``、``torch.npu``。

用法::

    from runtime.device import get_device, device_info, synchronize, manual_seed

    device = get_device()                 # 自动：npu -> cuda -> cpu
    device = get_device("npu", index=0)   # 强制指定
    print(device_info())

直接自检::

    python -m runtime.device
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# 1) 必须在 import torch 之前完成的事
# ---------------------------------------------------------------------------
CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
_CANN_ENV_KEYS = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")

BACKENDS = ("npu", "cuda", "cpu")


def cann_env_ready() -> bool:
    """CANN 的 set_env.sh 是否已经 source 过。"""
    return any(os.environ.get(key) for key in _CANN_ENV_KEYS)


if not cann_env_ready():
    # 别让 import torch 因为 torch_npu 加载失败而直接崩掉；
    # 后面 npu_available() 会是 False，并给出怎么修的提示。
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402  （必须放在上面那段之后）


# ---------------------------------------------------------------------------
# 2) 后端探测
# ---------------------------------------------------------------------------
def npu_available() -> bool:
    """昇腾 NPU 是否可用（torch_npu 已加载且能识别到设备）。"""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    """CUDA GPU 是否可用。"""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def auto_backend() -> str:
    """按 npu -> cuda -> cpu 的顺序挑一个可用后端。"""
    if npu_available():
        return "npu"
    if cuda_available():
        return "cuda"
    return "cpu"


def device_count(backend: Optional[str] = None) -> int:
    backend = (backend or auto_backend()).lower()
    if backend == "npu":
        return int(torch.npu.device_count()) if npu_available() else 0
    if backend == "cuda":
        return int(torch.cuda.device_count()) if cuda_available() else 0
    return 1


def visible_devices(backend: Optional[str] = None) -> str:
    """打印用：当前进程能看到哪些卡（含被 *_VISIBLE_DEVICES 过滤掉的情况）。"""
    backend = (backend or auto_backend()).lower()
    env_key = "ASCEND_RT_VISIBLE_DEVICES" if backend == "npu" else "CUDA_VISIBLE_DEVICES"
    env_val = os.environ.get(env_key)
    text = f"{backend}: {device_count(backend)} 张可见设备"
    if env_val:
        text += f"（{env_key}={env_val}）"
    return text


def require_npu() -> None:
    """要求 NPU 可用，否则抛出带修复步骤的错误。"""
    if npu_available():
        return
    if not cann_env_ready():
        raise RuntimeError(
            "NPU 不可用：检测到 CANN 环境没有加载（没有 ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH）。\n"
            f"请先执行：\n    source {CANN_SET_ENV}\n"
            "再启动 python；确认方式：\n"
            '    python -c "import torch, torch_npu; print(torch.npu.is_available())"'
        )
    raise RuntimeError(
        "NPU 不可用：CANN 环境看起来已加载，但 torch.npu.is_available() 为 False。\n"
        "常见原因：\n"
        "  1) 驱动/固件与 CANN 版本不匹配（`npu-smi info` 看 Software Version）\n"
        "  2) 目标卡处于异常状态（`npu-smi info` 里 Health 不是 OK，比如某些卡是 Critical）\n"
        "  3) 需要 root 权限或设备被别的容器独占\n"
        f"当前 ASCEND_TOOLKIT_HOME={os.environ.get('ASCEND_TOOLKIT_HOME')!r}, "
        f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH')!r}"
    )


# ---------------------------------------------------------------------------
# 3) device 的统一入口
# ---------------------------------------------------------------------------
def get_device(backend: Optional[str] = None, index: int = 0) -> torch.device:
    """返回 ``torch.device``；``backend=None`` 时自动选择。"""
    backend = (backend or auto_backend()).lower()
    if backend not in BACKENDS:
        raise ValueError(f"未知 backend: {backend!r}，可选 {BACKENDS}")
    if backend == "npu":
        require_npu()
        return torch.device(f"npu:{index}")
    if backend == "cuda":
        if not cuda_available():
            raise RuntimeError("CUDA 不可用（torch.cuda.is_available() 为 False）")
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def set_device(device) -> torch.device:
    """把当前进程绑定到指定设备（NPU/CUDA 需要，CPU 忽略）。"""
    device = torch.device(device)
    if device.type == "npu":
        require_npu()
        torch.npu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def synchronize(device=None) -> None:
    """计时/读显存前调用，确保算子真的跑完。"""
    backend = torch.device(device).type if device is not None else auto_backend()
    if backend == "npu" and npu_available():
        torch.npu.synchronize()
    elif backend == "cuda" and cuda_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 4) 随机种子 / 混合精度 / 显存
# ---------------------------------------------------------------------------
def manual_seed(seed: int) -> None:
    """torch + 当前加速器的随机种子。"""
    torch.manual_seed(seed)
    if npu_available():
        torch.npu.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed(seed)


def seed_all(seed: int = 1337) -> None:
    """python / numpy / torch 一起设种子。"""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    manual_seed(seed)


def autocast_dtype(device=None) -> Optional[torch.dtype]:
    """混合精度该用哪个 dtype（None 表示不做 autocast）。

    ⚠️ 两边支持的不一样，这里是**实测结论**（torch_npu 2.10.0.post6 + CANN 9.1.1）：

    * 昇腾 NPU：autocast **只支持 fp16**。传 bf16 不会报错，但 torch 会打一条
      "npu Autocast only supports dtypes of torch.float16, torch.float32" 然后
      **静默关掉 autocast** —— 也就是你以为在用混合精度，其实在跑 fp32。
    * CUDA：用 bf16（动态范围大、不需要 loss scaling，训练更稳）。

    所以**不要在配置里硬写死 bfloat16**，交给这里按后端决定。
    """
    backend = torch.device(device).type if device is not None else auto_backend()
    if backend == "npu":
        return torch.float16
    if backend == "cuda":
        return torch.bfloat16
    return None


@contextmanager
def autocast(device=None, enabled: bool = True, dtype: Optional[torch.dtype] = None) -> Iterator[None]:
    """统一的混合精度上下文，屏蔽 'npu' / 'cuda' 的差异。

    ``dtype`` 显式给了就用它；否则按 :func:`autocast_dtype` 选后端支持的。
    """
    backend = torch.device(device).type if device is not None else auto_backend()
    if not enabled or backend == "cpu":
        yield
        return
    with torch.autocast(device_type=backend, dtype=dtype or autocast_dtype(device)):
        yield


def empty_cache() -> None:
    if npu_available():
        torch.npu.empty_cache()
    elif cuda_available():
        torch.cuda.empty_cache()


def memory_allocated_gb() -> Optional[float]:
    if npu_available():
        return torch.npu.memory_allocated() / 1024 ** 3
    if cuda_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return None


def max_memory_allocated_gb() -> Optional[float]:
    if npu_available():
        return torch.npu.max_memory_allocated() / 1024 ** 3
    if cuda_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return None


# ---------------------------------------------------------------------------
# 5) 环境自检
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    device_count: int
    name: str
    torch_version: str
    default_dtype: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"backend       : {self.backend}",
            f"device_count  : {self.device_count}",
            f"name          : {self.name}",
            f"torch         : {self.torch_version}",
            f"default dtype : {self.default_dtype}",
        ]
        if self.backend == "npu":
            lines.append(f"torch_npu     : {self.extra.get('torch_npu_version')}")
            lines.append(f"cann home     : {self.extra.get('ascend_toolkit_home')}")
        return "\n".join(lines)


def device_info(backend: Optional[str] = None, index: int = 0) -> DeviceInfo:
    backend = (backend or auto_backend()).lower()
    name = "CPU"
    extra: Dict[str, Any] = {}
    if backend == "npu" and npu_available():
        try:
            name = torch.npu.get_device_name(index)
        except Exception:
            name = "Ascend NPU"
        try:
            import torch_npu

            extra["torch_npu_version"] = torch_npu.__version__
        except Exception:
            extra["torch_npu_version"] = "unknown"
        extra["ascend_toolkit_home"] = os.environ.get("ASCEND_TOOLKIT_HOME")
    elif backend == "cuda" and cuda_available():
        name = torch.cuda.get_device_name(index)
    return DeviceInfo(
        backend=backend,
        device_count=device_count(backend),
        name=name,
        torch_version=torch.__version__,
        default_dtype=str(torch.get_default_dtype()),
        extra=extra,
    )


def summary() -> str:
    """一句话环境摘要，日志开头打一行很有用。"""
    backend = auto_backend()
    return (
        f"[runtime] {visible_devices(backend)} | torch={torch.__version__} | "
        f"cann_ready={cann_env_ready()}"
    )


def main() -> None:
    print(summary())
    print("-" * 60)
    print(device_info())
    if not npu_available():
        print("-" * 60)
        print("提示：要启用昇腾 NPU，请先 `source %s` 再运行。" % CANN_SET_ENV)


__all__ = [
    "BACKENDS",
    "CANN_SET_ENV",
    "DeviceInfo",
    "auto_backend",
    "autocast",
    "autocast_dtype",
    "cann_env_ready",
    "cuda_available",
    "device_count",
    "device_info",
    "empty_cache",
    "get_device",
    "main",
    "manual_seed",
    "max_memory_allocated_gb",
    "memory_allocated_gb",
    "npu_available",
    "require_npu",
    "seed_all",
    "set_device",
    "summary",
    "synchronize",
    "visible_devices",
]


if __name__ == "__main__":
    main()
