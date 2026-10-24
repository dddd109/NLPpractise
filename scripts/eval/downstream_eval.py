"""下游评测：HellaSwag（4 选 1）与 LAMBADA（末词预测）。

为什么默认跑在 CPU 上
--------------------
昇腾的算子图是"按 shape 编译"的：HellaSwag 有几千条**长度各不相同**的样本，
在 NPU 上会触发几千次图编译（实测每个新 shape 要几秒），完全不可接受。
61M 的小模型在 CPU 上跑完 1000 条大约几分钟，所以默认 ``--device cpu``。
（等 CANN 的融合算子修好后，可以试 ``--device npu``。）

用法::

    # 随机初始化跑通链路（很快）
    python scripts/eval/downstream_eval.py --limit 20

    # 用训练好的 checkpoint
    python scripts/eval/downstream_eval.py \
        --ckpt runs/pretrain/gqa-12L-512d_step20000.pt --limit 1000

    # 只跑一个任务
    python scripts/eval/downstream_eval.py --tasks lambada --limit 200

指标说明
--------
HellaSwag:
    ``acc``      按 **每 token 平均 logprob** 选 ending（标准做法）
    ``acc_norm`` 按 **每字符平均 logprob** 选（抵消长度偏置，通常略高）
    随机基线 = 25%
LAMBADA:
    ``acc``      教师强制逐 token 取 argmax，整个末词全部猜对才算对
                  （等价于"贪心生成能否复现末词"，随机基线接近 0%）
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 必须在 import datasets 之前：这台机器直连 huggingface.co 不通，要走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 必须在 import torch 之前先导入 runtime.device —— 它会在 torch 导入之前判断 CANN
# 是否已加载，没加载就设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0，否则 `import torch` 直接崩。
from runtime.device import device_info, get_device, synchronize

import tiktoken
import torch
import torch.nn.functional as F

from models.build_models import Model
from runtime.config import load_config


# ---------------------------------------------------------------------------
# 打分核心：给 (ctx, continuation) 算 continuation 的 logprob
# ---------------------------------------------------------------------------
@torch.no_grad()
def continuation_stats(model, ctx_ids, cont_ids, device, block_size, vocab_size):
    """返回 ``(sum_logprob, n_tokens, greedy_all_match)``。

    * 只统计 continuation 部分（ctx 的 token 不参与打分）；
    * 序列超过 ``block_size`` 时从**左侧**截断（保留结尾，这是我们关心的部分）；
    * ``vocab_size`` 用来把 logits 截到 tokenizer 的词表（模型可能 pad 到 50304）。
    """
    ids = list(ctx_ids) + list(cont_ids)
    ctx_kept = len(ctx_ids)
    if len(ids) > block_size:
        cut = len(ids) - block_size
        ids = ids[cut:]
        ctx_kept = max(ctx_kept - cut, 1)

    x = torch.tensor(ids[:-1], device=device, dtype=torch.long)[None]
    targets = torch.tensor(ids[1:], device=device, dtype=torch.long)[None]
    logits, _ = model(x)
    logits = logits[0].float()[..., :vocab_size]

    start = max(ctx_kept - 1, 0)          # logits[i] 预测 ids[i+1]
    logits = logits[start:]
    targets = targets[0, start:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)

    greedy_ok = bool((logits.argmax(dim=-1) == targets).all().item())
    return float(logp.sum()), int(logp.numel()), greedy_ok


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------
def eval_hellaswag(model, enc, device, block_size, limit, seed):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    vocab = enc.n_vocab
    n = acc = acc_norm = 0
    for ex in ds:
        ctx = ex["ctx"].rstrip()
        ctx_ids = enc.encode(ctx)
        scores = []
        for ending in ex["endings"]:
            cont_ids = enc.encode(" " + ending)
            s, k, _ = continuation_stats(model, ctx_ids, cont_ids, device, block_size, vocab)
            scores.append((s, max(k, 1), max(len(ending), 1)))

        best = max(range(4), key=lambda i: scores[i][0] / scores[i][1])
        best_norm = max(range(4), key=lambda i: scores[i][0] / scores[i][2])
        acc += int(best == ex["label"])
        acc_norm += int(best_norm == ex["label"])
        n += 1
        if n % 50 == 0:
            print(f"  [hellaswag] {n}/{len(ds)}  acc={acc/n:.4f}", flush=True)
        del ctx_ids, scores
    return {"n": n, "acc": acc / max(n, 1), "acc_norm": acc_norm / max(n, 1)}


# ---------------------------------------------------------------------------
# LAMBADA
# ---------------------------------------------------------------------------
def eval_lambada(model, enc, device, block_size, limit, seed):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    vocab = enc.n_vocab
    n = acc = skipped = 0
    total_logp = 0.0
    total_tok = 0
    for ex in ds:
        text = ex["text"].strip()
        cut = text.rfind(" ")
        if cut <= 0:
            skipped += 1
            continue
        ctx_ids = enc.encode(text[:cut])
        cont_ids = enc.encode(text[cut + 1:])
        if not cont_ids:
            skipped += 1
            continue
        s, k, greedy_ok = continuation_stats(
            model, ctx_ids, cont_ids, device, block_size, vocab
        )
        acc += int(greedy_ok)
        total_logp += s
        total_tok += k
        n += 1
        if n % 50 == 0:
            print(f"  [lambada]   {n}  acc={acc/n:.4f}", flush=True)
        del ctx_ids, cont_ids
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
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    block_size = args.block_size or int(cfg["model"]["context_length"])

    device = get_device(args.device) if args.device != "cpu" else torch.device("cpu")
    print(f"[eval] device={device}")
    if args.device != "cpu":
        print(device_info())

    model = Model(cfg, use_cache=False).to(device)
    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
        try:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:                      # 老版本 torch 没有 weights_only
            blob = torch.load(ckpt, map_location=device)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state)
        print(f"[eval] 已加载 {ckpt}（step={blob.get('step') if isinstance(blob, dict) else '?'}）")
    model.eval()

    enc = tiktoken.get_encoding(str(cfg["data"].get("tokenizer", "gpt2")).split(":")[-1])
    print(f"[eval] tokenizer={enc.name} block_size={block_size} limit={args.limit}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {}
    for task in tasks:
        print(f"\n=== {task} ===", flush=True)
        if task == "hellaswag":
            r = eval_hellaswag(model, enc, device, block_size, args.limit, args.seed)
            print(f"  HellaSwag  n={r['n']}  acc={r['acc']:.4f}  acc_norm={r['acc_norm']:.4f}"
                  f"   (随机基线 0.25)", flush=True)
        elif task == "lambada":
            r = eval_lambada(model, enc, device, block_size, args.limit, args.seed)
            print(f"  LAMBADA    n={r['n']}  acc={r['acc']:.4f}  "
                  f"pPL(last word)={r['ppl_last_word']:.1f}", flush=True)
        else:
            print(f"  未知任务 {task}，跳过")
            continue
        results[task] = r
        if args.device != "cpu":
            synchronize(device)

    results["_meta"] = {
        "config": str(cfg_path),
        "ckpt": args.ckpt,
        "device": str(device),
        "limit": args.limit,
        "seed": args.seed,
    }
    out = Path(args.out) if args.out else (_REPO_ROOT / "runs" / "downstream_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果已写入 {out}")


if __name__ == "__main__":
    main()
