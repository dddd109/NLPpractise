"""下游评测：HellaSwag（4 选 1）与 LAMBADA（末词预测）。

设备
----
``--device`` 默认 ``auto``：有 NPU 就用 NPU、否则 CPU。

但在昇腾上用 NPU 有一个**必须处理**的前提：算子图是"按 shape 编译"的。
HellaSwag 有几千条长度各异的样本，直接跑会触发几千次图编译（每条几秒），
完全不可接受。所以这里做了 **长度分桶（右填充到 bucket 的倍数）**：

* 所有序列右填充到 ``ceil(len / bucket) * bucket``，不同的 T 值只剩个位数；
* 每个 T 只编译一次，之后全部复用；
* 因果注意力只看左边，**右填充不影响真实 token 的 logits**，所以打分不受污染；
* HellaSwag 的 4 个 ending 会**拼成一个 batch 一次前向**（4 倍加速）。

用法::

    # 自动选设备（NPU 优先）
    python scripts/eval/downstream_eval.py --ckpt runs/pretrain/xxx_step45999.pt --limit 1000

    # 强制 CPU / NPU
    python scripts/eval/downstream_eval.py --device cpu  --limit 50
    python scripts/eval/downstream_eval.py --device npu  --limit 200

指标
----
HellaSwag:
    ``acc``      按每 token 平均 logprob 选 ending（标准做法），随机基线 25%
    ``acc_norm`` 按每字符平均 logprob 选（抵消长度偏置）
LAMBADA:
    ``acc``      教师强制逐 token 取 argmax，整个末词全部猜对才算对
    ``ppl_last_word`` 末词的平均困惑度（随机初始化时 ≈ 词表大小）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 必须在 import datasets 之前：这台机器直连 huggingface.co 不通，要走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 必须在 import torch 之前先导入 runtime.device —— 它会在 torch 导入之前判断 CANN
# 是否已加载，没加载就设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0，否则 `import torch` 直接崩。
from runtime.device import get_device, npu_available, synchronize

import tiktoken
import torch
import torch.nn.functional as F

from models.build_models import Model
from runtime.config import load_config


# ---------------------------------------------------------------------------
# 打分核心：右填充 + 分桶，一次前向算一批序列的 continuation logprob
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_continuation_stats(
    model, seqs, ctx_lens, device, block_size, vocab_size, bucket=64
):
    """一批 (ctx+cont) 序列的 continuation 统计。

    Args:
        seqs:     每条序列的 token id 列表（ctx + continuation）
        ctx_lens: 每条序列的 ctx 长度（cont 从这之后开始打分）
    Returns:
        ``[(sum_logprob, n_tokens, greedy_all_match), ...]``
    """
    # 超长从左侧截断（保留结尾）；同时修正 ctx 长度
    fixed_seqs, fixed_ctx = [], []
    for ids, cl in zip(seqs, ctx_lens):
        if len(ids) > block_size:
            cut = len(ids) - block_size
            ids = ids[cut:]
            cl = max(cl - cut, 1)
        fixed_seqs.append(ids)
        fixed_ctx.append(cl)

    max_len = max(len(s) for s in fixed_seqs)
    total = ((max_len + bucket - 1) // bucket) * bucket
    ids_batch = torch.zeros((len(fixed_seqs), total), dtype=torch.long)  # 右填充 0
    for i, ids in enumerate(fixed_seqs):
        ids_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    x = ids_batch[:, :-1].to(device)
    targets = ids_batch[:, 1:].to(device)
    logits, _ = model(x)
    logits = logits.float()[..., :vocab_size]

    results = []
    for i, (ids, cl) in enumerate(zip(fixed_seqs, fixed_ctx)):
        start = max(cl - 1, 0)          # logits[i] 预测 ids[i+1]
        end = len(ids) - 1              # 真实长度内最后一个可预测位置
        lg, tg = logits[i, start:end], targets[i, start:end]
        logp = F.log_softmax(lg, dim=-1).gather(-1, tg[:, None]).squeeze(-1)
        results.append(
            (float(logp.sum()), int(logp.numel()), bool((lg.argmax(-1) == tg).all()))
        )
    return results


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------
def eval_hellaswag(model, enc, device, block_size, limit, seed, bucket, vocab):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    n = acc = acc_norm = 0
    t0 = time.perf_counter()
    for ex in ds:
        ctx = ex["ctx"].rstrip()
        ctx_ids = enc.encode(ctx)
        # 4 个 ending 拼成一批，一次前向
        ending_ids = [enc.encode(" " + e) for e in ex["endings"]]
        stats = batched_continuation_stats(
            model,
            [ctx_ids + e for e in ending_ids],
            [len(ctx_ids)] * 4,
            device,
            block_size,
            vocab,
            bucket,
        )
        per_tok = [s / max(k, 1) for s, k, _ in stats]
        per_chr = [s / max(len(e), 1) for (s, _, _), e in zip(stats, ex["endings"])]
        best = max(range(4), key=lambda i: per_tok[i])
        best_norm = max(range(4), key=lambda i: per_chr[i])
        acc += int(best == ex["label"])
        acc_norm += int(best_norm == ex["label"])
        n += 1
        if n % 100 == 0:
            print(f"  [hellaswag] {n}/{len(ds)}  acc={acc/n:.4f}  "
                  f"{n/(time.perf_counter()-t0):.1f} it/s", flush=True)
    return {"n": n, "acc": acc / max(n, 1), "acc_norm": acc_norm / max(n, 1)}


# ---------------------------------------------------------------------------
# LAMBADA
# ---------------------------------------------------------------------------
def eval_lambada(model, enc, device, block_size, limit, seed, bucket, vocab):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    n = acc = skipped = 0
    total_logp, total_tok = 0.0, 0
    t0 = time.perf_counter()
    for ex in ds:
        text = ex["text"].strip()
        cut = text.rfind(" ")
        if cut <= 0:
            skipped += 1
            continue
        ctx_ids = enc.encode(text[:cut])
        cont_ids = enc.encode(text[cut + 1 :])
        if not cont_ids:
            skipped += 1
            continue
        (s, k, greedy_ok), = batched_continuation_stats(
            model, [ctx_ids + cont_ids], [len(ctx_ids)], device, block_size, vocab, bucket
        )
        acc += int(greedy_ok)
        total_logp += s
        total_tok += k
        n += 1
        if n % 100 == 0:
            print(f"  [lambada]   {n}  acc={acc/n:.4f}  "
                  f"{n/(time.perf_counter()-t0):.1f} it/s", flush=True)
    return {
        "n": n,
        "skipped": skipped,
        "acc": acc / max(n, 1),
        "ppl_last_word": float(torch.tensor(-total_logp / max(total_tok, 1)).exp()),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HellaSwag / LAMBADA 下游评测")
    ap.add_argument("-C", "--config", default="configs/pretrain/pretrain_llama.yaml")
    ap.add_argument("--ckpt", default=None, help="训练产出的 .pt；不给就用随机初始化")
    ap.add_argument("--tasks", default="hellaswag,lambada")
    ap.add_argument("--limit", type=int, default=200, help="每个任务取多少条（0=全部）")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "npu", "cpu"],
        help="auto = 有 NPU 用 NPU",
    )
    ap.add_argument("--bucket", type=int, default=64, help="长度分桶粒度（NPU 上越小编译越多）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    block_size = args.block_size or int(cfg["model"]["context_length"])

    if args.device == "auto":
        device = get_device()                      # npu -> cuda -> cpu
    else:
        device = get_device(args.device)
    print(f"[eval] device={device}  bucket={args.bucket}  block_size={block_size}")

    model = Model(cfg, use_cache=False).to(device)
    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
        try:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(ckpt, map_location=device)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state)
        step = blob.get("step") if isinstance(blob, dict) else "?"
        print(f"[eval] 已加载 {ckpt}（step={step}）")
    model.eval()

    enc = tiktoken.get_encoding(str(cfg["data"].get("tokenizer", "gpt2")).split(":")[-1])
    vocab = enc.n_vocab
    print(f"[eval] tokenizer={enc.name} limit={args.limit}")

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n=== {task} ===", flush=True)
        if task == "hellaswag":
            r = eval_hellaswag(model, enc, device, block_size, args.limit,
                               args.seed, args.bucket, vocab)
            print(f"  HellaSwag  n={r['n']}  acc={r['acc']:.4f}  "
                  f"acc_norm={r['acc_norm']:.4f}   (随机基线 0.25)", flush=True)
        elif task == "lambada":
            r = eval_lambada(model, enc, device, block_size, args.limit,
                             args.seed, args.bucket, vocab)
            print(f"  LAMBADA    n={r['n']}  acc={r['acc']:.4f}  "
                  f"ppl(last word)={r['ppl_last_word']:.1f}", flush=True)
        else:
            print(f"  未知任务 {task}，跳过")
            continue
        results[task] = r
        if device.type != "cpu":
            synchronize(device)

    results["_meta"] = {
        "config": str(cfg_path),
        "ckpt": args.ckpt,
        "device": str(device),
        "limit": args.limit,
        "seed": args.seed,
        "bucket": args.bucket,
    }
    out = Path(args.out) if args.out else (_REPO_ROOT / "runs" / "downstream_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {out}")


if __name__ == "__main__":
    main()
