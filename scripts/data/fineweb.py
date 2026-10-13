"""下载 HuggingFaceFW/fineweb-edu 的 sample-10BT 数据，筛掉过短文档，写成 jsonl。

本机（NPU 服务器）环境有两个坑，脚本内部已经处理，直接运行即可：

1. huggingface.co 直连超时（curl 12s 无响应），但镜像 hf-mirror.com 可用（约 31MB/s）。
   -> 用 HF_ENDPOINT 指向镜像；外部环境若已设置 HF_ENDPOINT 则以外部为准。

2. 这个 conda 环境里一旦 import torch，就会自动加载 torch_npu，而 libhccl.so 不在
   LD_LIBRARY_PATH 中，于是报 "Failed to load the backend extension: torch_npu"。
   datasets 的 streaming 内部会 import torch，所以整个下载会直接崩。
   -> 用 TORCH_DEVICE_BACKEND_AUTOLOAD=0 关掉设备后端自动加载（下载数据不需要 NPU）。

3. 用 load_dataset(..., name="sample-10BT") 走内置配置时，datasets 会去请求
   https://huggingface.co/api/datasets/.../tree/... 列文件清单（这一处不吃 HF_ENDPOINT），
   于是卡死在连接超时上。-> 这里直接给出 sample-10BT 的 parquet 分片地址，
   用 parquet 加载器流式读取：既绕开该清单请求，也不必先下载整个 2.15GB 分片。
"""

import os

# --- 必须在 import datasets 之前设置 ---
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from datasets import load_dataset  # noqa: E402

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
# sample-10BT 由 14 个 parquet 分片组成（sample/10BT/000_00000.parquet ... 013_00000.parquet），
# 每个约 2.15GB。默认读第 0 个分片，要更多数据就换 --shard。
PARQUET_SHARD = "sample/10BT/000_00000.parquet"


def shard_url(shard: str) -> str:
    return f"hf://datasets/{DATASET_ID}/{shard}"


def main():
    parser = argparse.ArgumentParser()
    # 注意：默认文件名要和 scripts/data/tokenize_fineweb.py 的 --input 保持一致
    parser.add_argument(
        "--output",
        default="data/raw/fineweb_edu/part-00000.jsonl",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=100_000_000_000_000_000_000,
        help="最多保存多少有效文档",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="最少多少字符的文档才会被保留",
    )
    parser.add_argument(
        "--shard",
        default=PARQUET_SHARD,
        help="sample-10BT 的 parquet 分片路径（例如 sample/10BT/001_00000.parquet）",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(
            f"Output file {output_path} already exists. "
            "Please remove it or choose a different path."
        )
        return

    # 先写 .partial，全部成功后再改名，避免中断时留下一个看起来正常的半成品文件
    tmp_path = output_path.with_name(output_path.name + ".partial")

    print(
        f"Dataset: {DATASET_ID}, Config: {DATASET_CONFIG}, Shard: {args.shard}\n"
        f"Source: {shard_url(args.shard)}\n"
        f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}\n"
        f"Output: {output_path}, Max Docs: {args.max_docs}, Min Chars: {args.min_chars}"
    )

    dataset = load_dataset(
        "parquet",
        data_files=shard_url(args.shard),
        split="train",
        streaming=True,
    )

    seen = 0
    saved = 0
    rejected = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            seen += 1
            text = row.get("text") or ""
            text = text.replace("\x00", "")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = text.strip()

            if len(text) < args.min_chars:
                rejected += 1
                continue

            record = {
                "text": text,
                "source": DATASET_ID,
                "id": row.get("id"),
                "url": row.get("url"),
                # 注意：sample-10BT 里没有 date 列，用 CC-MAIN 的 dump 名代替
                "dump": row.get("dump"),
                "language": row.get("language", "en"),
                "language_score": row.get("language_score"),
                "token_count": row.get("token_count"),
                "score": row.get("score"),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

            if saved % 1000 == 0:
                f.flush()
                print(f"seen: {seen}, saved: {saved}, rejected: {rejected}")

            if saved >= args.max_docs:
                print(f"Reached max docs limit of {args.max_docs}. Stopping.")
                break

        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(output_path)
    print(f"Wrote {saved} docs to {output_path}")

    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "shard": args.shard,
        "max_docs": args.max_docs,
        "seen": seen,
        "saved": saved,
        "rejected": rejected,
        "output": str(output_path),
    }

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
