"""配置入口：dataclass 默认值 < YAML 文件 < 命令行 ``--set`` 覆盖。

设计目标
--------
1. **单一入口**：所有脚本都只经过 :func:`load_config`，不再各自 argparse / 硬编码；
2. **可复现**：``--dump`` 把最终生效的配置原样写出来，日志里带上来源文件列表；
3. **加架构零改动**：``configs/model/*.yaml`` 里只写 ``arch: gqa``，
   由 :func:`build_attention` 按注册表查表构造模块 —— 新增架构只要加一个类 +
   在 :func:`_attn_registry` 里注册一行，训练 / benchmark 代码都不用动。

用法::

    from runtime.config import load_config, build_attention

    cfg = load_config("configs/train/base.yaml",
                      overrides=["model.arch=gqa", "train.batch_size=8"])
    print(cfg.model["arch"])
    attn = build_attention(cfg, layer_idx=0)

命令行::

    python -m runtime.config --config configs/train/base.yaml --set train.lr=1e-4 model.arch=mla
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

SECTIONS: Tuple[str, ...] = ("model", "train", "data", "run")

ARCH_REGISTRY_NAMES: Tuple[str, ...] = ("mha", "mqa", "gqa", "mla")


# ---------------------------------------------------------------------------
# 配置对象
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """最终生效的配置。四个 section 各自是普通 dict，方便随手加字段。"""

    model: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    # 记录加载过哪些文件，写进日志便于复现
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {name: copy.deepcopy(getattr(self, name)) for name in SECTIONS}

    def dumps(self) -> str:
        payload = {"sources": self.sources, **self.to_dict()}
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)

    def dump(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.dumps(), encoding="utf-8")

    # 便捷取值（找不到时给默认值，避免到处 .get(...) 里三层外三层）
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.to_dict()
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


# ---------------------------------------------------------------------------
# 合并 / 覆盖
# ---------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 里的叶子值覆盖 base，dict 则继续往下并。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """把命令行字符串转成合适的 python 类型。"""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _set_nested(tree: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        nxt = node.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"配置路径冲突（{key} 已经是标量）: {dotted}")
        node = nxt
    node[keys[-1]] = value


def parse_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """把 ``["model.arch=gqa", "train.lr=1e-4"]`` 解析成嵌套 dict。"""
    tree: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--set 需要 k=v 形式，收到 {item!r}")
        key, _, value = item.partition("=")
        _set_nested(tree, key.strip(), _coerce(value.strip()))
    return tree


def load_yaml(path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "配置入口需要 pyyaml：pip install pyyaml（或 conda install pyyaml）"
        ) from exc
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是映射（mapping）")
    return data


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Config:
    """按 默认值 < YAML(含 ``extends`` 基线) < ``--set`` 的顺序合成配置。

    Args:
        config_path: YAML 文件路径；None 表示只用 defaults + overrides。
        overrides:   ``["model.arch=gqa", ...]``
        defaults:    代码里的兜底默认值（优先级最低）。
    """
    merged: Dict[str, Any] = copy.deepcopy(defaults or {})
    sources: List[str] = []

    if config_path is not None:
        merged = deep_merge(merged, load_yaml(config_path))
        sources.append(str(config_path))

    # 支持 `extends: base.yaml` 做基线继承（相对当前文件所在目录解析）
    parent = merged.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if config_path is not None and not parent_path.is_absolute():
            parent_path = Path(config_path).parent / parent_path
        merged = deep_merge(load_yaml(parent_path), merged)
        sources.insert(0, str(parent_path))

    if overrides:
        merged = deep_merge(merged, parse_overrides(overrides))

    cfg = Config(sources=sources)
    for name in SECTIONS:
        setattr(cfg, name, merged.pop(name, {}) or {})
    # 没归类的顶层键不丢，塞进 run
    if merged:
        cfg.run = deep_merge(cfg.run, merged)
    return cfg


# ---------------------------------------------------------------------------
# 架构注册表：yaml 里写 arch: xxx，这里查表构造
# ---------------------------------------------------------------------------
def _attn_registry() -> Dict[str, Tuple[Type, Optional[Type]]]:
    from modules.attention.GQA import GQA, GQAConfig
    from modules.attention.MHA import MultiHeadAttention
    from modules.attention.MLA import MLA, MLAConfig
    from modules.attention.MQA import MQA, MQAConfig

    return {
        "mha": (MultiHeadAttention, None),  # 老接口：直接吃 kwargs
        "mqa": (MQA, MQAConfig),
        "gqa": (GQA, GQAConfig),
        "mla": (MLA, MLAConfig),
    }


def _accepted_keys(target: Type) -> set:
    """目标类能接受哪些关键字（dataclass 按字段，普通类按签名）。"""
    if dataclasses.is_dataclass(target):
        return {f.name for f in dataclasses.fields(target) if f.init}
    return set(inspect.signature(target).parameters)


def _filter_kwargs(data: Dict[str, Any], target: Type) -> Dict[str, Any]:
    allowed = _accepted_keys(target)
    return {k: v for k, v in data.items() if k in allowed}


def build_attention(cfg: Config, layer_idx: Optional[int] = None):
    """按 ``cfg.model["arch"]`` 构造 attention 模块。

    ``configs/model/*.yaml`` 里多写的字段（n_layer / vocab_size / norm ...）会被忽略，
    只有目标架构认识的字段才传下去 —— 所以一份 model 配置能喂给不同架构。
    """
    model_cfg = dict(cfg.model)
    arch = str(model_cfg.get("arch", "mqa")).lower()
    registry = _attn_registry()
    if arch not in registry:
        raise KeyError(f"未知架构 {arch!r}；已注册：{sorted(registry)}")

    impl, cfg_cls = registry[arch]
    if cfg_cls is None:
        return impl(**_filter_kwargs(model_cfg, impl))

    arch_cfg = cfg_cls(**_filter_kwargs(model_cfg, cfg_cls))
    try:
        return impl(arch_cfg, layer_idx=layer_idx)
    except TypeError:
        # 少数实现不接受 layer_idx
        return impl(arch_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="配置入口：dataclass 默认值 < YAML < --set 覆盖"
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], help="形如 model.arch=gqa"
    )
    parser.add_argument("--dump", default=None, help="把最终配置写到文件")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    print(cfg.dumps())
    if args.dump:
        cfg.dump(args.dump)
        print(f"\n[config] 已写出 {args.dump}")


__all__ = [
    "ARCH_REGISTRY_NAMES",
    "SECTIONS",
    "Config",
    "build_attention",
    "deep_merge",
    "load_config",
    "load_yaml",
    "main",
    "parse_overrides",
]


if __name__ == "__main__":
    main()
