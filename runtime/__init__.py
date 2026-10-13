"""runtime：把"跑在哪、怎么配"这两件事从模型/训练代码里抽出来。

- :mod:`runtime.device` 硬件抽象（昇腾 NPU / CUDA / CPU 统一入口）
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
]
