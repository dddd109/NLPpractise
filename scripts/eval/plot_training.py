"""训练曲线绘图：把 runs/pretrain/*.jsonl 画成报告可用的图。

产出（默认写到 ``runs/plots/``）::

    <run>_dashboard.png   2×2 面板：loss / lr / grad-norm / throughput
    <run>_loss.png        单张 loss 图（train EMA + val），适合直接放进报告
    compare_loss.png      多个 run 的 val_loss 对比（给了多个 jsonl 时）

用法::

    # 自动找最新的 run
    python scripts/eval/plot_training.py

    # 指定一个或多个
    python scripts/eval/plot_training.py runs/pretrain/train_a_rank0.jsonl runs/pretrain/train_b_rank0.jsonl

    # 横轴用 token 数、y 轴用对数、平滑窗口调大
    python scripts/eval/plot_training.py --x tokens --logy --smooth 101

说明
----
* ``--smooth`` 是 EMA 窗口（按**记录条数**算，不是 step 数）。训练 loss 每 10 步一个点、
  抖动约 ±0.1，默认 51 条（≈500 步）能把曲线抹平又不失真。
* val_loss 只在 eval 步变化，画成带 marker 的折线，并标出最好的一次。
* 字体：默认 ``--lang auto`` —— 检测到 CJK 字体就用中文，否则退回英文，避免出现方块。
  这台机器上是 ``Droid Sans Fallback``；想强制英文用 ``--lang en``。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")                     # 服务器无显示环境
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.font_manager as fm       # noqa: E402
import numpy as np                         # noqa: E402

# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------
PALETTE = {
    "train": "#4C78A8",      # 蓝
    "ema": "#1F4E79",        # 深蓝
    "val": "#E45756",        # 红
    "lr": "#54A24B",         # 绿
    "grad": "#B279A2",       # 紫
    "tps": "#F58518",        # 橙
    "grid": "#D9D9D9",
}

CJK_CANDIDATES = (
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "Droid Sans Fallback", "SimHei", "Microsoft YaHei",
)

LABELS = {
    "zh": {
        "loss": "损失 Loss", "step": "训练步数 Step", "train": "训练 loss",
        "val": "验证 loss", "best": "最佳验证", "lr": "学习率", "grad": "梯度范数",
        "tps": "吞吐 (tokens/s)", "token": "已见 token 数",
        "final": "最终验证 loss", "dash": "训练曲线",
    },
    "en": {
        "loss": "Loss", "step": "Step", "train": "train loss",
        "val": "val loss", "best": "best val", "lr": "Learning rate",
        "grad": "Grad norm", "tps": "Throughput (tokens/s)", "token": "Tokens seen",
        "final": "final val loss", "dash": "Training curves",
    },
}


def _cjk_font_usable(name: str, probe: str = "损失率步数曲线") -> bool:
    """检查某个字体是否**真的**覆盖这些汉字。

    为什么不能只看字体名：这台机器上唯一的 CJK 字体是
    ``DroidSansFallbackFull.ttf``，字体名看着没问题，实测却缺 ``损/失/率/步/数...``
    这些常用字（渲染出来就是方块）。所以这里直接查 cmap。
    """
    try:
        path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        from matplotlib.ft2font import FT2Font

        face = FT2Font(path)
        return all(face.get_char_index(ord(ch)) != 0 for ch in probe)
    except Exception:
        return False


def setup_style(lang: str):
    """选字体 + 统一风格。返回实际使用的语言（'zh' 或 'en'）。"""
    available = {f.name for f in fm.fontManager.ttflist}
    cjk = next((n for n in CJK_CANDIDATES if n in available and _cjk_font_usable(n)), None)
    if cjk is None:
        if lang == "zh":
            print("[warn] 没有可用的中文字体（现有 CJK 字体缺汉字），图里改用英文标签。"
                  "\n       想用中文：装 fonts-noto-cjk，或把 NotoSansSC*.ttf 丢进 ~/.fonts 再跑一次。")
        lang = "en"
    if lang == "zh":
        # 主字体管拉丁字母/数字，CJK 字体跟在后面管中文（matplotlib >= 3.6 支持按字形回退）
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", cjk]
        plt.rcParams["axes.unicode_minus"] = False     # 否则负号会变方块
        print(f"[plot] 字体: DejaVu Sans + {cjk}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4D4D4D",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
    })
    return lang


# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------
def load_run(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue           # 训练还在写，最后一行可能是半截
    if not rows:
        raise ValueError(f"{path} 里没有有效记录")

    def col(key, default=np.nan):
        out = []
        for r in rows:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(default)
            else:
                out.append(float(v))
        return np.array(out)

    steps = col("step")
    loss = col("loss")

    # val_loss 是阶梯值：只在变化处记点
    vx, vy = [], []
    last = None
    for r in rows:
        v = r.get("val_loss")
        if v is None:
            continue
        v = float(v)
        if math.isnan(v) or (last is not None and v == last):
            continue
        vx.append(float(r["step"]))
        vy.append(v)
        last = v

    return {
        "path": path,
        "name": path.name.replace("train_", "").replace("_rank0.jsonl", ""),
        "step": steps,
        "loss": loss,
        "lr": col("lr"),
        "grad": col("grad_norm"),
        "tps": col("tokens_per_sec"),
        "vstep": np.array(vx),
        "vloss": np.array(vy),
        "last_step": float(steps[-1]),
    }


def ema(y: np.ndarray, window: int) -> np.ndarray:
    """指数滑动平均，顺带跳过 nan。"""
    if window <= 1:
        return y.copy()
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(y)
    m = np.nan
    for i, v in enumerate(y):
        if np.isnan(v):
            out[i] = m
            continue
        m = v if np.isnan(m) else alpha * v + (1 - alpha) * m
        out[i] = m
    return out


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
def _finish(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}  ({path.stat().st_size/1024:.0f} KB)")


def plot_dashboard(run, out: Path, lang: str, smooth: int, xmode: str, logy: bool, suffix: str = ""):
    L = LABELS[lang]
    x = run["step"] / 1e9 if xmode == "tokens" else run["step"]
    xlabel = L["token"] + " (B)" if xmode == "tokens" else L["step"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.subplots_adjust(hspace=0.32, wspace=0.22)
    (ax_loss, ax_lr), (ax_grad, ax_tps) = axes

    # ---- (1) loss ----
    ax_loss.plot(x, run["loss"], color=PALETTE["train"], alpha=0.16, lw=0.8)
    ax_loss.plot(x, ema(run["loss"], smooth), color=PALETTE["ema"], lw=2.0,
                 label=L["train"] + f" (EMA {smooth})")
    if len(run["vstep"]):
        vx = run["vstep"] / 1e9 if xmode == "tokens" else run["vstep"]
        ax_loss.plot(vx, run["vloss"], color=PALETTE["val"], lw=1.6, marker="o",
                     ms=3, label=L["val"])
        i_best = int(np.argmin(run["vloss"]))
        ax_loss.scatter([vx[i_best]], [run["vloss"][i_best]], s=46, zorder=5,
                        facecolor="white", edgecolor=PALETTE["val"], linewidth=1.6)
        ax_loss.annotate(
            f"{L['best']} {run['vloss'][i_best]:.4f} @ {run['vstep'][i_best]:.0f}",
            xy=(vx[i_best], run["vloss"][i_best]),
            xytext=(8, 14), textcoords="offset points",
            fontsize=8.5, color=PALETTE["val"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["val"], lw=0.8, alpha=0.9),
        )
    if logy:
        ax_loss.set_yscale("log")
    ax_loss.set_title(L["loss"])
    ax_loss.set_xlabel(xlabel)
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right")

    # ---- (2) lr ----
    ax_lr.plot(x, run["lr"], color=PALETTE["lr"], lw=1.8)
    ax_lr.set_title(L["lr"])
    ax_lr.set_xlabel(xlabel)
    ax_lr.set_ylabel("lr")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ---- (3) grad norm ----
    ax_grad.plot(x, run["grad"], color=PALETTE["grad"], alpha=0.25, lw=0.8)
    ax_grad.plot(x, ema(run["grad"], smooth), color="#6B4C7A", lw=1.8)
    ax_grad.set_title(L["grad"])
    ax_grad.set_xlabel(xlabel)
    ax_grad.set_ylabel("|g|")

    # ---- (4) throughput ----
    ax_tps.plot(x, run["tps"], color=PALETTE["tps"], alpha=0.25, lw=0.8)
    ax_tps.plot(x, ema(run["tps"], smooth), color="#B35A00", lw=1.8)
    ax_tps.set_title(L["tps"])
    ax_tps.set_xlabel(xlabel)
    ax_tps.set_ylabel("tok/s")
    ax_tps.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    title = f"{L['dash']} - {run['name']}   (step {run['last_step']:.0f})"
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.985)
    _finish(fig, out / f"{run['name']}{suffix}_dashboard.png")


def plot_loss(run, out: Path, lang: str, smooth: int, xmode: str, logy: bool, suffix: str = ""):
    """单张 loss 图，去掉噪声，直接放进报告。"""
    L = LABELS[lang]
    x = run["step"] / 1e9 if xmode == "tokens" else run["step"]
    xlabel = L["token"] + " (B)" if xmode == "tokens" else L["step"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, run["loss"], color=PALETTE["train"], alpha=0.14, lw=0.8,
            label=L["train"] + " (raw)")
    ax.plot(x, ema(run["loss"], smooth), color=PALETTE["ema"], lw=2.2,
            label=L["train"] + f" (EMA {smooth})")
    if len(run["vstep"]):
        vx = run["vstep"] / 1e9 if xmode == "tokens" else run["vstep"]
        ax.plot(vx, run["vloss"], color=PALETTE["val"], lw=1.8, marker="o", ms=3.2,
                label=L["val"])
        ax.annotate(f"{L['final']} {run['vloss'][-1]:.4f}",
                    xy=(vx[-1], run["vloss"][-1]), xytext=(-10, 16),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=PALETTE["val"],
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec=PALETTE["val"], lw=0.8, alpha=0.9))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.set_title(f"Loss - {run['name']}")
    ax.legend(loc="upper right")
    _finish(fig, out / f"{run['name']}{suffix}_loss.png")


def plot_compare(runs, out: Path, lang: str, xmode: str):
    L = LABELS[lang]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(runs):
        if not len(r["vstep"]):
            continue
        vx = r["vstep"] / 1e9 if xmode == "tokens" else r["vstep"]
        ax.plot(vx, r["vloss"], color=cmap(i % 10), lw=1.9, marker="o", ms=3,
                label=f"{r['name']} ({r['vloss'][-1]:.4f})")
    ax.set_xlabel(L["token"] + " (B)" if xmode == "tokens" else L["step"])
    ax.set_ylabel(L["val"])
    ax.set_title(L["val"] + " - compare")
    ax.legend(loc="upper right")
    _finish(fig, out / "compare_loss.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="画训练曲线")
    ap.add_argument("jsonl", nargs="*", help="训练记录 jsonl；不给就自动取 runs/pretrain 下最新的")
    ap.add_argument("--out-dir", default=None, help="默认 runs/plots")
    ap.add_argument("--smooth", type=int, default=51, help="EMA 窗口（记录条数）")
    ap.add_argument("--x", default="step", choices=["step", "tokens"],
                    help="横轴用步数还是已见 token 数")
    ap.add_argument("--logy", action="store_true", help="loss 用对数轴")
    ap.add_argument("--lang", default="auto", choices=["auto", "zh", "en"])
    args = ap.parse_args()

    # 选语言
    if args.lang == "auto":
        lang = "zh"          # 交给 setup_style 里的字体探测决定要不要退回 en
    else:
        lang = args.lang
    lang = setup_style(lang)

    # 选文件
    default_dir = _REPO_ROOT / "runs" / "pretrain"
    if args.jsonl:
        paths = [Path(p) for p in args.jsonl]
    else:
        cands = sorted(default_dir.glob("train_*_rank0.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit(f"没找到训练记录：{default_dir}/train_*_rank0.jsonl")
        paths = [cands[-1]]
        print(f"[plot] 默认取最新: {paths[0].name}")

    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "runs" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 非默认渲染方式加后缀，避免和默认图互相覆盖
    suffix = ""
    if args.x != "step" or args.logy:
        suffix = f"_{args.x}" + ("_log" if args.logy else "")

    runs = []
    for p in paths:
        p = p if p.is_absolute() else (_REPO_ROOT / p)
        run = load_run(p)
        runs.append(run)
        print(f"\n[plot] {run['name']}: {len(run['step'])} 条记录, "
              f"step {run['step'][0]:.0f}→{run['last_step']:.0f}, "
              f"val 点 {len(run['vstep'])} 个")

    for run in runs:
        plot_dashboard(run, out_dir, lang, args.smooth, args.x, args.logy, suffix)
        plot_loss(run, out_dir, lang, args.smooth, args.x, args.logy, suffix)

    if len(runs) > 1:
        plot_compare(runs, out_dir, lang, args.x)

    # 摘要表
    print("\n=== 摘要 ===")
    for r in runs:
        best = r["vloss"].min() if len(r["vloss"]) else float("nan")
        final = r["vloss"][-1] if len(r["vloss"]) else float("nan")
        tok = r["last_step"] * 65536 / 1e9
        print(f"  {r['name']:28s} step={r['last_step']:6.0f}  tokens={tok:5.2f}B  "
              f"val_first={r['vloss'][0]:.4f}  best={best:.4f}  final={final:.4f}")
    print(f"\n[plot] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
