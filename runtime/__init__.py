"""runtime：把"跑在哪、怎么配、怎么并行"这三件事从模型/训练代码里抽出来。

- :mod:`runtime.device` 硬件抽象（昇腾 NPU / CUDA / CPU 统一入口）
- :mod:`runtime.dist`   多卡并行（DDP + hccl/nccl/gloo 的选择与安全退化）
- :mod:`runtime.config` 配置入口（dataclass 默认值 < YAML < 命令行 --set）
"""

from .config import (
    ARCH_REGISTRY_NAMES,
    FFN_REGISTRY_NAMES,
    Config,
    build_attention,
    build_ffn,
    deep_merge,
    load_config,
    parse_overrides,
)
from .device import (
    CANN_SET_ENV,
    DeviceInfo,
    auto_backend,
    autocast,
    autocast_dtype,
    cann_env_ready,
    cuda_available,
    device_count,
    device_info,
    empty_cache,
    get_device,
    manual_seed,
    max_memory_allocated_gb,
    memory_allocated_gb,
    npu_available,
    require_npu,
    seed_all,
    set_device,
    summary,
    synchronize,
    visible_devices,
)
from .dist import (
    DistContext,
    all_reduce_mean,
    all_reduce_sum,
    barrier,
    cleanup,
    default_backend,
    global_batch_tokens,
    no_sync,
    rank0_print,
    setup,
    wrap_model,
)

__all__ = [
    # config
    "ARCH_REGISTRY_NAMES",
    "FFN_REGISTRY_NAMES",
    "Config",
    "build_attention",
    "build_ffn",
    "deep_merge",
    "load_config",
    "parse_overrides",
    # device
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
    # dist
    "DistContext",
    "all_reduce_mean",
    "all_reduce_sum",
    "barrier",
    "cleanup",
    "default_backend",
    "global_batch_tokens",
    "no_sync",
    "rank0_print",
    "setup",
    "wrap_model",
]
