
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken


def is_validation_document(text: str) -> bool:
    # 基于文档内容的稳定划分，约 1% 进入验证集
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/fineweb_edu/part-000.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tokenized/fineweb_gpt2",
    )
    parser.add_argument(
        "--flush-tokens",
        type=int,
        default=1_000_000,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    stats_path = output_dir / "stats.json"

    for path in (train_path, val_path, stats_path):
        if path.exists():
            raise FileExistsError(
                f"{path} 已存在。请使用新的输出目录，避免覆盖。"
            )

    # GPT-2 BPE：词表大小 50257，EOS ID 为 50256
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token

    train_buffer = []
    val_buffer = []

    stats = {
        "input": str(input_path),
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": enc.n_vocab,
        "eos_token_id": eos_id,
        "documents": 0,
        "train_documents": 0,
        "val_documents": 0,
        "train_tokens": 0,
        "val_tokens": 0,
        "skipped_empty": 0,
    }

    def flush(buffer, file_obj):
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(file_obj)
            buffer.clear()

    with (
        input_path.open("r", encoding="utf-8") as src,
        train_path.open("wb") as train_file,
        val_path.open("wb") as val_file,
    ):
        for line in src:
            row = json.loads(line)
            text = (row.get("text") or "").strip()

            if not text:
                stats["skipped_empty"] += 1
                continue

            # 普通文本编码；文档边界由 EOS 明确标记
            token_ids = enc.encode_ordinary(text)
            token_ids.append(eos_id)

            stats["documents"] += 1

            if is_validation_document(text):
                target = val_buffer
                file_obj = val_file
                stats["val_documents"] += 1
                stats["val_tokens"] += len(token_ids)
            else:
                target = train_buffer
                file_obj = train_file
                stats["train_documents"] += 1
                stats["train_tokens"] += len(token_ids)

            target.extend(token_ids)

            if len(target) >= args.flush_tokens:
                flush(target, file_obj)

            if stats["documents"] % 1000 == 0:
                print(
                    f"docs={stats['documents']:,} "
                    f"train_tokens={stats['train_tokens']:,} "
                    f"val_tokens={stats['val_tokens']:,}"
                )

        flush(train_buffer, train_file)
        flush(val_buffer, val_file)

    stats["storage_dtype"] = "uint16"
    stats["train_file"] = str(train_path)
    stats["val_file"] = str(val_path)
    stats["train_bytes"] = train_path.stat().st_size
    stats["val_bytes"] = val_path.stat().st_size

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
