"""模型组装 + 预训练入口（支持多卡 DDP）。

运行方式
--------
单卡::

    python models/build_models.py -C configs/pretrain/pretrain_llama.yaml

多卡（本机物理卡 3 的 Health=Critical，先把它排除，用剩下 7 张）::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
    torchrun --nproc_per_node=7 models/build_models.py -C configs/pretrain/pretrain_llama.yaml

冒烟测试（20 步，确认多卡链路通）::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
    torchrun --nproc_per_node=7 models/build_models.py --max-steps 20 --log-every 1

设计约定
--------
* **设备/并行全部交给 runtime**：本文件里不出现 ``torch.npu`` / ``torch.cuda`` /
  ``nccl`` / ``hccl``，只用 ``runtime.device`` 和 ``runtime.dist`` 的接口，
  这样同一份代码在昇腾 / GPU / CPU 上都能跑。
* **结构由 yaml 决定**：attention 走 ``runtime.config.build_attention``、
  FFN 走 ``build_ffn``，所以换架构只改 ``configs/*.yaml``，不用动这个文件。
* **数据按 rank 分片**：``data/tokenized/*/train.bin`` 是 uint16 裸流，用
  ``np.memmap`` 零拷贝打开；每个 rank 从不同偏移起步、步长为 ``world_size*B*T``，
  这样各卡看到的是不同数据、且一起覆盖整个语料。
"""

# --- 让这个文件既能当脚本被 torchrun 调用，也能 python -m models.build_models ---
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import datetime
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cache.PagedKVCache import PagedKVCache
from modules.normalize.RMSNorm import RMSNorm
from runtime.config import build_attention, build_ffn, load_config
from runtime.device import (
    autocast,
    device_info,
    get_device,
    seed_all,
    synchronize,
)
from runtime.dist import (
    all_reduce_mean,
    all_reduce_sum,
    barrier,
    cleanup,
    global_batch_tokens,
    no_sync,
    rank0_print,
    setup,
    wrap_model,
)

# =====================================================================================
# 配置 / 参数
# =====================================================================================


def torch_dtype(name, default=torch.float32) -> torch.dtype:
    """把 yaml 里的 ``torch.float16`` 这类字符串转成 ``torch.dtype``。

    直接把它当 dtype 用会报错（它只是个 str），PagedKVCache 里 ``torch.empty(dtype=...)``
    也会挂。
    """
    if isinstance(name, torch.dtype):
        return name
    if not name:
        return default
    return getattr(torch, str(name).split(".")[-1], default)


def parse_args():
    parser = argparse.ArgumentParser(description="模型组装 + 预训练（支持多卡 DDP）")
    parser.add_argument(
        "-C",
        "--config",
        default="configs/pretrain/pretrain_llama.yaml",
        help="YAML 配置（相对仓库根目录解析）",
    )
    # 下面这些是"临时覆盖 yaml"，方便扫参时不用改文件
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-ckpt", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="开 PagedKVCache（只在生成/推理时需要；训练别开，会随步数一直涨）",
    )
    parser.add_argument(
        "--verbose-init",
        action="store_true",
        help="逐参数打印初始化的形状（默认只打一行汇总，多卡下别开）",
    )
    return parser.parse_args()


def resolve_config(args):
    path = Path(args.config)
    if not path.is_absolute():
        path = Path(_REPO_ROOT) / path
    if not path.exists():
        raise FileNotFoundError(f"找不到配置：{path}")
    cfg = load_config(str(path))
    # 命令行覆盖
    overrides = {
        ("train", "micro_batch_size"): args.micro_batch_size,
        ("train", "gradient_accumulation_steps"): args.grad_accum,
        ("train", "max_steps"): args.max_steps,
        ("train", "learning_rate"): args.lr,
        ("model", "context_length"): args.seq_len,
        ("run", "out_dir"): args.out_dir,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value
    cfg["run"].setdefault("out_dir", "runs/pretrain")
    cfg["run"]["config_path"] = str(path)
    return cfg


# =====================================================================================
# 模型
# =====================================================================================


class Block(nn.Module):
    """标准 pre-norm Transformer block：x + Attn(Norm(x))，再 x + FFN(Norm(x))。

    注意和原来那份的区别：**加了残差和 FFN**。原来的 ``forward`` 是
    ``x = norm(x); x = attn(x)``，既没有残差也没有前馈层、还每层都重新 norm 一遍
    残差流 —— 那样不是 Transformer，梯度也很难传。这里按主流写法组装，注意力/FFN
    的实现仍然全部来自 ``modules/``。
    """

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        n_embd = cfg["model"]["n_embd"]
        self.norm_1 = RMSNorm(n_embd)
        self.attn = build_attention(cfg, layer_idx=layer_idx)
        self.norm_2 = RMSNorm(n_embd)
        self.ffn = build_ffn(cfg)

    def forward(self, x, position_offset: int = 0, kv_cache=None):
        x = x + self.attn(self.norm_1(x), position_offset=position_offset, kv_cache=kv_cache)
        x = x + self.ffn(self.norm_2(x))
        return x


class Model(nn.Module):
    """词表嵌入 + N 层 Block + 最终 RMSNorm + 输出投影（可与嵌入层共享权重）。"""

    def __init__(self, config, use_cache: bool = False, verbose_init: bool = False):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        self.n_layer = int(model_cfg["n_layer"])
        self.vocab_size = int(model_cfg["vocab_size"])
        self.n_embd = int(model_cfg["n_embd"])

        self.embedd = nn.Embedding(self.vocab_size, self.n_embd)
        self.blocks = nn.ModuleList(
            Block(config, layer_idx=i) for i in range(self.n_layer)
        )
        self.norm = RMSNorm(self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        if model_cfg.get("tie_embeddings", False):
            # GPT-2 式的权重共享：省一份 [vocab, n_embd] 参数
            self.proj.weight = self.embedd.weight

        # KV cache 只在生成时需要；训练开着会随步数无限增长（显存泄漏式）
        self.use_cache = use_cache
        self.cache = (
            PagedKVCache(
                num_layers=self.n_layer,
                block_size=int(model_cfg["context_length"]),
                device=get_device(),
                dtype=torch_dtype(config["train"].get("precision")),
            )
            if use_cache
            else None
        )

        self._init_weights(verbose=verbose_init)

    def _init_weights(self, verbose: bool = False):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "gamma" in name:
                nn.init.ones_(param)          # RMSNorm 的 gamma 应当是 1
            else:
                nn.init.zeros_(param)
            if verbose:
                print(f"  {name}: {tuple(param.shape)}")
        n_param = sum(p.numel() for p in self.parameters())
        print(f"[model] 参数量 = {n_param:,} ({n_param / 1e6:.1f}M)", flush=True)

    # ------------------------------------------------------------------
    def forward(self, idx, targets=None, position_offset: int = 0):
        x = self.embedd(idx)
        cache = self.cache if self.use_cache else None
        for block in self.blocks:
            x = block(x, position_offset=position_offset, kv_cache=cache)
        x = self.norm(x)
        logits = self.proj(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    # ------------------------------------------------------------------
    def reset_cache(self):
        if self.cache is not None:
            self.cache.reset()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=32, temperature=1.0, top_k=None):
        """自回归采样。

        有 cache 时：第一轮喂整段 prompt，之后每轮只喂 1 个 token —— 位置偏移由
        attention 模块自己按 cache 长度推（所以这里传 ``position_offset=0``，
        显式再传一次会重复计数）。
        """
        self.eval()
        if self.cache is not None:
            self.reset_cache()
        for _ in range(max_new_tokens):
            if self.cache is not None:
                n_cached = self.cache.get_seq_len(0)
                inp = idx if n_cached == 0 else idx[:, n_cached:]
            else:
                inp = idx
            logits, _ = self.forward(inp, position_offset=0)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# =====================================================================================
# 数据：uint16 裸流 + 多卡分片
# =====================================================================================


class DataLoaderLite:
    """从 ``*.bin``（uint16 裸 token 流）上按 token 滑窗取 batch。

    * ``np.memmap`` 零拷贝打开 15GB 的文件，不会一次性读进内存；
    * 多卡分片：rank r 从 ``r*B*T`` 起步，每次前进 ``world_size*B*T``，
      各卡数据互不重叠，合起来完整覆盖语料。
    """

    def __init__(self, B: int, T: int, cfg, ctx, split: str):
        assert split in ("train", "val"), split
        self.B, self.T = B, T
        self.split = split
        self.rank = ctx.rank
        self.world_size = ctx.world_size

        key = "train_path" if split == "train" else "val_path"
        path = cfg["data"].get(key) or cfg["data"].get(split)
        if path is None:
            raise KeyError(f"配置里没有 data.{key}")
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path(_REPO_ROOT) / self.path
        if not self.path.exists():
            raise FileNotFoundError(f"找不到数据文件：{self.path}")

        # 注意：tokenize_fineweb.py 写出来的是 uint16；直接 np.load 读不了裸流
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.tokens)
        if self.n_tokens < (B * T + 1) * self.world_size:
            raise ValueError(
                f"{self.path} 只有 {self.n_tokens} 个 token，放不下 "
                f"{self.world_size} 卡 × {B} × {T}"
            )
        self.reset()

    def reset(self):
        """回到本 rank 的起始位置（每次评估前调用，保证可复现）。"""
        self.position = self.B * self.T * self.rank

    def next_batch(self):
        B, T = self.B, self.T
        stride = B * T * self.world_size
        if self.position + B * T + 1 > self.n_tokens:
            self.reset()
        buf = self.tokens[self.position : self.position + B * T + 1]
        # uint16 -> int64（embedding 需要 long），B*T 只有几千个元素，开销可忽略
        chunk = torch.from_numpy(np.asarray(buf, dtype=np.int64))
        x = chunk[:-1].view(B, T)
        y = chunk[1:].view(B, T)
        self.position += stride
        return x, y


# =====================================================================================
# 学习率 / 训练
# =====================================================================================


def lr_at(step: int, cfg) -> float:
    """warmup + 余弦衰减到 ``min_lr_ratio * lr``。"""
    train = cfg["train"]
    max_lr = float(train["learning_rate"])
    min_lr = max_lr * float(train.get("min_lr_ratio", 0.1))
    max_steps = int(train["max_steps"])
    warmup = int(train.get("warmup_steps", 0))

    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device, eval_iters: int, ctx) -> float:
    model.eval()
    loader.reset()
    # losses 必须建在加速器上：hccl/nccl 只能 all_reduce 设备张量，CPU 张量会报错
    losses = torch.zeros(eval_iters, device=device)
    for i in range(eval_iters):
        x, y = loader.next_batch()
        with autocast(device):
            _, loss = model(x.to(device), y.to(device))
        losses[i] = loss.detach().float()
    mean = all_reduce_mean(losses.mean())
    model.train()
    return float(mean)


def main():
    args = parse_args()
    cfg = resolve_config(args)
    ctx = setup()                       # 建 process group + 绑定本 rank 的设备
    device = ctx.device
    seed_all(int(cfg["train"].get("seed", 1337)) + ctx.rank)

    rank0_print(ctx, "=" * 78)
    rank0_print(ctx, f"配置: {cfg['run']['config_path']}")
    rank0_print(ctx, f"设备: {device}")
    rank0_print(ctx, device_info())
    rank0_print(ctx, "=" * 78)

    # ---------------- 数据 ----------------
    B = int(cfg["train"]["micro_batch_size"])
    T = int(cfg["model"]["context_length"])
    grad_accum = int(cfg["train"]["gradient_accumulation_steps"])
    train_loader = DataLoaderLite(B, T, cfg, ctx, split="train")
    val_loader = DataLoaderLite(B, T, cfg, ctx, split="val")
    rank0_print(
        ctx,
        f"[data] {train_loader.path.name}: {train_loader.n_tokens / 1e9:.2f}B tokens | "
        f"micro_batch={B} seq={T} grad_accum={grad_accum} world_size={ctx.world_size}\n"
        f"       每次参数更新的全局 token 数 = "
        f"{global_batch_tokens(B, T, grad_accum, ctx):,}",
    )

    # ---------------- 模型 ----------------
    rank0_print(ctx, f"[model] arch={cfg['model'].get('attention')} "
                     f"ffn={cfg['model'].get('ffn')} n_layer={cfg['model']['n_layer']}")
    model = Model(cfg, use_cache=args.use_cache, verbose_init=args.verbose_init)
    model.to(device)
    model = wrap_model(model, ctx)        # 单卡时原样返回
    raw_model = model.module if ctx.enabled else model

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )

    # ---------------- 训练循环 ----------------
    max_steps = int(cfg["train"]["max_steps"])
    eval_interval = int(cfg["train"].get("eval_interval", 100))
    ckpt_interval = int(cfg["train"].get("checkpoint_interval", 500))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    out_dir = Path(cfg["run"]["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(_REPO_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{cfg['model'].get('attention')}-{cfg['model']['n_layer']}L-"
        f"{cfg['model']['n_embd']}d"
    )
    log_path = out_dir / f"train_{tag}_rank{ctx.rank}.jsonl"
    history = []

    model.train()
    for step in range(max_steps):
        last = step == max_steps - 1
        if not args.no_eval and (step % eval_interval == 0 or last):
            val_loss = evaluate(model, val_loader, device, args.eval_iters, ctx)
            rank0_print(ctx, f"[eval] step {step}: val_loss = {val_loss:.4f}")

        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss_accum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # 梯度累积：只有最后一个 micro-step 才做梯度通信
            sync_ctx = nullcontext() if micro == grad_accum - 1 else no_sync(model)
            with sync_ctx, autocast(device):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.detach()

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        # 说明：DDP 在最后一个 micro-step 已经把所有 rank 的梯度做了 all_reduce
        # （得到的是同一个平均值），所以每个 rank 看到的梯度完全一致，
        # 这里各自按同样的阈值裁剪是等价的，不需要额外通信。
        optimizer.step()
        synchronize(device)
        dt = time.perf_counter() - t0

        step_loss = float(all_reduce_mean(loss_accum))
        tokens = global_batch_tokens(B, T, grad_accum, ctx)
        if ctx.is_main and (step % args.log_every == 0 or last):
            rank0_print(
                ctx,
                f"[train] step {step:5d}/{max_steps} loss {step_loss:.4f} "
                f"lr {lr:.2e} |g| {float(norm):.3f} {dt * 1000:.0f}ms "
                f"{tokens / dt:,.0f} tok/s",
            )
            history.append(
                {"step": step, "loss": step_loss, "lr": lr, "grad_norm": float(norm),
                 "ms": dt * 1000, "tokens_per_sec": tokens / dt}
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1], ensure_ascii=False) + "\n")

        if (step % ckpt_interval == 0 or last) and not args.no_ckpt:
            if ctx.is_main:
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg.to_dict(),
                    "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                path = out_dir / f"{tag}_step{step}.pt"
                torch.save(ckpt, path)
                rank0_print(ctx, f"[ckpt] 已保存 {path}")
            # 其他 rank 等 rank0 写完再继续（否则可能被抢先覆盖）
            barrier()

    cleanup(ctx)


if __name__ == "__main__":
    main()
