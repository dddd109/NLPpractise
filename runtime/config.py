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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

SECTIONS: Tuple[str, ...] = ("model", "train", "data", "run")

# 段名别名：yaml 里写 training: 等价于 train:（兼容老配置）
SECTION_ALIASES: Dict[str, str] = {"training": "train"}

# 模型字段别名：把 HF 风格的命名映射到内部名字，这样不同风格的 yaml 都能直接用
MODEL_KEY_ALIASES: Dict[str, str] = {
    "hidden_size": "n_embd",
    "context_length": "block_size",
    "max_position_embeddings": "block_size",
    "num_attention_heads": "n_head",
    "num_key_value_heads": "n_kv_head",
    "intermediate_size": "ffn_hidden_dim",
    "num_hidden_layers": "n_layer",
}
_ROPE_NAMES_TRUE = ("rope", "rotary", "rope2d")
_ROPE_NAMES_FALSE = ("none", "learned", "absolute", "abs")


def _canonical_section(name: str) -> str:
    return SECTION_ALIASES.get(name, name)

ARCH_REGISTRY_NAMES: Tuple[str, ...] = ("mha", "mqa", "gqa", "mla")
FFN_REGISTRY_NAMES: Tuple[str, ...] = ("gelu", "swiglu", "moe")


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

    # ---- 兼容两种取值风格 ------------------------------------------------
    #   cfg.model["n_embd"]      （属性 + 下标）
    #   cfg["model"]["n_embd"]   （纯字典风格，老脚本里到处都是这种写法）
    def __getitem__(self, key: str) -> Any:
        key = _canonical_section(key)
        if key in SECTIONS:
            return getattr(self, key)
        if key == "sources":
            return self.sources
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        key = _canonical_section(key)
        if key not in SECTIONS:
            raise KeyError(f"{key!r} 不是配置段，可选 {SECTIONS}")
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        key = _canonical_section(key)
        return key in SECTIONS or key == "sources"

    def keys(self) -> Tuple[str, ...]:
        return SECTIONS

    def items(self):
        return ((name, getattr(self, name)) for name in SECTIONS)

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
    # 段名别名（training -> train）：两边都写了就深合并，别名不覆盖正式名
    for alias, target in SECTION_ALIASES.items():
        if alias in merged:
            setattr(
                cfg,
                target,
                deep_merge(getattr(cfg, target), merged.pop(alias) or {}),
            )
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


def normalize_model_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把常见的字段命名变体统一成内部名字，让不同风格的 yaml 都能直接用。

    例：HF 风格写 ``hidden_size`` / ``context_length`` / ``num_key_value_heads`` /
    ``intermediate_size``，内部用 ``n_embd`` / ``block_size`` / ``n_kv_head`` /
    ``ffn_hidden_dim``；``position_embedding: rope`` 等价于 ``rope: true``。
    """
    out = dict(model_cfg)
    for alias, canonical in MODEL_KEY_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]

    if "rope" not in out:
        pe = str(out.get("position_embedding") or out.get("pos") or "").lower()
        if pe in _ROPE_NAMES_TRUE:
            out["rope"] = True
        elif pe in _ROPE_NAMES_FALSE:
            out["rope"] = False
    return out


def select_arch(model_cfg: Dict[str, Any]) -> str:
    """架构名：``arch`` 优先，其次 ``attention``（两种写法都接受）。"""
    return str(model_cfg.get("arch") or model_cfg.get("attention") or "mqa").lower()


def build_attention(cfg: Config, layer_idx: Optional[int] = None):
    """按 ``cfg.model["arch"]``（或 ``model["attention"]``）构造 attention 模块。

    ``configs/model/*.yaml`` 里多写的字段（n_layer / vocab_size / norm ...）会被忽略，
    只有目标架构认识的字段才传下去 —— 所以一份 model 配置能喂给不同架构。
    """
    model_cfg = normalize_model_cfg(cfg.model)
    arch = select_arch(model_cfg)
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


def _ffn_builders() -> Dict[str, "Callable[[Dict[str, Any]], Any]"]:
    """FFN 构建器：yaml 里写 ``ffn: moe`` 就能切到 MoE 前馈层。

    这里用"函数式构建器"而不是 ``(类, config类)`` 对，是因为像 ``SwiGLU`` 这种
    老模块的签名要求显式 ``hidden_dim``，需要在构建时补默认值。
    """
    from modules.ffn.GeLU import GeLUFFN
    from modules.ffn.SWiGLU import SwiGLU
    from modules.moe.MoEFFN import MoEFFN, MoEFFNConfig

    def _dim(model_cfg: Dict[str, Any]) -> int:
        if "n_embd" not in model_cfg:
            raise KeyError("构建 FFN 需要 model.n_embd")
        return int(model_cfg["n_embd"])

    def _hidden(model_cfg: Dict[str, Any], dim: int) -> int:
        return int(model_cfg.get("ffn_hidden_dim", 4 * dim))

    def _swiglu(model_cfg: Dict[str, Any]):
        dim = _dim(model_cfg)
        return SwiGLU(dim=dim, hidden_dim=_hidden(model_cfg, dim))

    def _gelu(model_cfg: Dict[str, Any]):
        dim = _dim(model_cfg)
        return GeLUFFN(dim=dim, hidden_dim=_hidden(model_cfg, dim))

    def _moe(model_cfg: Dict[str, Any]):
        data = _filter_kwargs(model_cfg, MoEFFNConfig)
        data.setdefault("dim", _dim(model_cfg))
        data.setdefault("hidden_dim", _hidden(model_cfg, data["dim"]))
        return MoEFFN(MoEFFNConfig(**data))

    return {"gelu": _gelu, "swiglu": _swiglu, "moe": _moe}


def build_ffn(cfg: Config):
    """按 ``cfg.model["ffn"]`` 构造前馈层（gelu / swiglu / moe）。"""
    model_cfg = normalize_model_cfg(cfg.model)
    name = str(model_cfg.get("ffn", "swiglu")).lower()
    builders = _ffn_builders()
    if name not in builders:
        raise KeyError(f"未知 FFN {name!r}；已注册：{sorted(builders)}")
    return builders[name](model_cfg)


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
    "FFN_REGISTRY_NAMES",
    "MODEL_KEY_ALIASES",
    "SECTION_ALIASES",
    "SECTIONS",
    "Config",
    "build_attention",
    "build_ffn",
    "deep_merge",
    "load_config",
    "load_yaml",
    "main",
    "normalize_model_cfg",
    "parse_overrides",
    "select_arch",
]


if __name__ == "__main__":
    main()
