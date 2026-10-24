# 150M 实验方案（配置 + 代码改动清单）

> 目标：把当前 61M 的对照实验放大到 **~152M**，用于架构对比（GQA 基线 → MLA / SWA / MoE）。
> 本文只写"改哪里、怎么改、为什么"；代码由你自己落，改完按 §5 验证。
> 所有数字都标了【核算】= 按公式手算，可复现。

---

## 1. 先给结论

| 项 | 61M（现在） | **152M（目标）** |
|---|---|---|
| n_layer | 12 | **18** |
| n_embd | 512 | **768** |
| n_head | 8 | **12**（head_dim 仍 64） |
| n_kv_head | 4（G=2） | **4**（G=3） |
| intermediate_size | 1408 | **2048**（≈ 2.67×n_embd，LLaMA 惯例） |
| block_size | 1024 | 1024（不变） |
| micro_batch × seq × grad_accum × world | 1×1024×8×8 | 不变 → **65,536 tokens/步** |
| 参数量 | 61.15M | **151.9M**【核算】 |
| 每卡状态内存（DDP, 16B/param） | 0.98 GB | **2.43 GB** |
| 每卡总占用（含激活，估算） | ~3 GB | **~5–6 GB**（32GB 很宽松） |
| 1 epoch | 150,372 步 | 同样 **150,372 步**（语料不变） |
| Chinchilla 最优（20×params） | 1.22B tokens ≈ 18,662 步 | **3.04B tokens ≈ 46,357 步** |

**关键取舍**：
- 想和 61M **同 token 预算对比** → 仍然跑 **20,000 步**（1.31B tokens，等价 8.6 tokens/param，欠训练但可比）。
- 想跑出 152M 的"正常"质量 → 跑 **46,000 步**（3.04B tokens，Chinchilla 最优），时间约为 20k 步的 2.3 倍。

建议：**先跑 20k 步**（和基线可比、省时间），确认曲线正常后再决定要不要延长到 46k。

---

## 2. 配置怎么改

**不要改现有的 `pretrain_llama.yaml`**（那是 61M 基线的复现依据）。新建一份：

`configs/pretrain/pretrain_150m.yaml`

```yaml
model:
  name: llama_150m
  vocab_size: 50257
  n_layer: 18            # 12 -> 18
  n_embd: 768            # 512 -> 768
  n_head: 12             # 8 -> 12（768/12 = 64，head_dim 不变）
  n_kv_head: 4           # 4 -> 4（G = 12/4 = 3，保持 3:1 的 GQA 比例）
  intermediate_size: 2048   # 1408 -> 2048（≈2.67×n_embd）
  context_length: 1024
  tie_embeddings: true

  position_embedding: rope
  normalization: rmsnorm
  ffn: swiglu
  attention: gqa

data:
  train_path: data/tokenized/fineweb_gpt2/train.bin
  val_path: data/tokenized/fineweb_gpt2/val.bin
  tokenizer: tiktoken:gpt2
  eos_token_id: 50256
  dtype: uint16

training:
  seed: 42
  precision: torch.float16
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.1
  warmup_steps: 1000        # 20k 步的 5%；若跑 46k 步就设 2000
  grad_clip: 1.0

  micro_batch_size: 1
  gradient_accumulation_steps: 8
  max_steps: 20000          # 见 §1 的取舍
  eval_interval: 100
  checkpoint_interval: 2000 # 见 §4.3（152M 的 ckpt 约 1.5~2.5GB/个）
```

**参数联动提醒**：`tokens/步 = micro_batch × context_length × grad_accum × world_size`。
如果你把 `micro_batch_size` 从 1 改成 2（吞吐更好），tokens/步 翻倍 → **`max_steps` 要减半**（20k → 10k）才能保持同样的 token 预算，否则就不是同 token 对比了。

---

## 3. 预算核算（自己复核一遍，别信我）

### 3.1 参数量【核算】

```
embed  : 50257 × 768                      = 38,597,376
每层 attn : q  768×768+768                =    590,592
            kv 768×(2×64×4)+512           =    393,728
            proj 768×768+768              =    590,592     → 1,574,912
每层 ffn  : SwiGLU(768,2048) 3×768×2048   =  4,718,592
每层 norm : 2 × 768                       =      1,536
每层合计                                  =  6,295,040
× 18 层                                  = 113,310,720
final norm: 768                           =        768
--------------------------------------------------------
总计                                      = 151,908,864  ≈ 152.0M
```

注意：`embed` 占了 **25%**（38.6M）。这是 `tie_embeddings: true` 的好处（只算一次）。

### 3.2 显存【核算】

- 状态：152M × 16 B = **2.43 GB/卡**（DDP 每卡一份完整副本）
- 激活（B=1, T=1024）：注意力分数 12 头 × 1024² × 2B ≈ 25 MB/层，
  反向要留 2~3 份 → ~1.4 GB（18 层）；FFN 激活 ~0.1 GB；合计 **~1.5–2.5 GB**
- 总计 **~5–6 GB/卡**，32 GB 用不到 20%

→ **B=1/T=1024 不需要 activation checkpointing**。但如果把 `context_length` 提到 2048 或 `micro_batch` 提到 4，先按 §4.4 加 checkpointing 再试。

### 3.3 算力【核算】

20k 步 = 1.31B tokens，FLOPs ≈ `6 × N × D = 6 × 1.519e8 × 1.31e9 ≈ 1.19e18`。

时间**只能实测**：日志每步都打 `tok/s`，总时间 ≈ 总 tokens ÷ tok/s。
参考：你 61M 的 2 万步如果实测约 18 h，则 152M 大约 **1.5~2.5 倍**（大模型 MFU 更高，不会线性放大 2.5 倍）。**别用小模型的 tok/s 乘参数倍率**。

---

## 4. 代码要改什么

先给个好消息：**模型代码本身是 shape-generic 的**，`Block` / `Model` / `build_attention` / `build_ffn` 全都没写死维度，
所以 §2 的配置改完理论上就能直接跑。下面按"必须 / 建议 / 可选"分。

### 4.1 【必须先验证，不用改代码】配置键能不能被吃到

`intermediate_size` → 内部 `ffn_hidden_dim`、`attention: gqa` → `arch` 这两条别名**已经在 `runtime/config.py` 里处理了**
（`normalize_model_cfg` + `select_arch`），但要**确认没有被静默忽略**：

```bash
python - <<'PY'
import sys; sys.path.insert(0, ".")
from runtime.config import load_config, build_attention, build_ffn
cfg = load_config("configs/pretrain/pretrain_150m.yaml")
a = build_attention(cfg, layer_idx=0)
f = build_ffn(cfg)
print("attn:", type(a).__name__, "n_head", a.n_head, "kv_head", a.kv_head)
print("ffn :", type(f).__name__, "w1.out_features =", f.w1.out_features)   # 应该是 2048
PY
```

`w1.out_features` 打印 2048 才算对；如果是 3072（=4×768）说明别名没生效，需要检查 yaml 键名。

### 4.2 【建议】fp16 加 GradScaler —— 18 层更容易 NaN

现在训练循环是：

```python
with sync_ctx, autocast(device):
    _, loss = model(x, y)
    loss = loss / grad_accum
loss.backward()
...
optimizer.step()
```

**问题**：昇腾的 autocast 只能用 fp16（`runtime.device.autocast_dtype` 已经处理），而 fp16 的梯度下溢
在层数变多、loss 变小时更容易出现 Inf/NaN。61M 能跑不代表 152M 能跑。

**改法**（`models/build_models.py`，训练循环）：

```python
# 循环外
scaler = torch.npu.amp.GradScaler(enabled=(device.type == "npu"))

# 循环内
with sync_ctx, autocast(device):
    _, loss = model(x, y)
    loss = loss / grad_accum
scaler.scale(loss).backward()

scaler.unscale_(optimizer)                       # 先还原再裁剪，否则裁剪的是放大后的梯度
norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```

判据：如果日志里出现 `loss nan` 或梯度范数爆炸，先加这个；加了还不行就 **关掉 autocast 跑 fp32**（慢 1.5~2×，但稳）。

### 4.3 【建议】checkpoint 保留策略 —— 否则磁盘会被吃光

`checkpoint_interval: 2000` + 152M 的 ckpt（模型 fp16 0.3 GB + Adam m/v 1.2 GB + config）≈ **1.5~2.5 GB/个**。
20k 步 = 10 个 ≈ 15~25 GB；如果还是 500 步一个就是 40 个 ≈ 60~100 GB（你只剩 194 GB）。

**改法**（`models/build_models.py`，保存完 ckpt 之后）：

```python
keep_last = 3
old = sorted(out_dir.glob(f"{tag}_step*.pt"), key=lambda p: p.stat().st_mtime)
for p in old[:-keep_last]:
    p.unlink()
```

### 4.4 【可选】activation checkpointing（只在 OOM 时加）

T=1024 / B=1 大概率用不上。真 OOM 了再在 `Block.forward` 外面套：

```python
from torch.utils.checkpoint import checkpoint
x = checkpoint(blk, x, use_reentrant=False)
```

注意：**开了 KV cache（`--use-cache`）时不能用**，两者语义冲突。

### 4.5 【建议】残差缩放初始化 —— 层数从 12 到 18 后更明显

现在 `Model._init_weights` 用的是 `xavier_uniform_`，没有对残差分支做缩放。
层数一大，残差流方差会累积（这就是 GPT-2/nanoGPT 里 `1/sqrt(2L)` 那个技巧）。

**改法**（`models/build_models.py: Model._init_weights`）：

```python
import math
for name, p in self.named_parameters():
    if p.dim() > 1:
        nn.init.normal_(p, mean=0.0, std=0.02)
    elif "gamma" in name:
        nn.init.ones_(p)          # RMSNorm 的 gamma
    else:
        nn.init.zeros_(p)

scale = 1.0 / math.sqrt(2 * self.n_layer)
with torch.no_grad():
    for blk in self.blocks:
        blk.attn.proj.weight.mul_(scale)   # 注意：GQA 里叫 proj
        blk.ffn.w2.weight.mul_(scale)      # SwiGLU 的输出投影
```

注意 `blk.ffn` 是 `SwiGLU`，输出投影叫 **`w2`**（不是 `c_proj`）。如果以后换成 `ffn: moe`，MoEFFN 的专家也是 SwiGLU，要把每个专家的 `w2` 都缩放。

### 4.6 【可选】学习率

`3e-4` 对 152M + 65k tokens/步是可以的（GPT-2 124M 量级常用 3e-4~6e-4）。
如果看到 loss 在前期震荡，先降 warmup 或降到 `2e-4`；**不要一次改多个变量**。

---

## 5. 验证步骤（按顺序，别跳）

```bash
cd /home/gdut/projects/NLPpractise && conda activate nlp-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ① 配置键真的生效（§4.1）：attn 头数 12/4、ffn 2048
# ② 参数总量：应该打印 ~151.9M
python models/build_models.py -C configs/pretrain/pretrain_150m.yaml \
  --dry-run --seq-len 128 --micro-batch-size 1 --grad-accum 1

# ③ 8 卡链路 + 显存（不更新权重）
torchrun --nproc_per_node=8 models/build_models.py \
  -C configs/pretrain/pretrain_150m.yaml \
  --dry-run --seq-len 128 --micro-batch-size 1 --grad-accum 1

# ④ 正式训练（挂后台，看前 50 步的 loss 是否正常下降）
nohup setsid bash -c 'cd /home/gdut/projects/NLPpractise && source ~/miniconda3/etc/profile.d/conda.sh && conda activate nlp-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh && torchrun --nproc_per_node=8 --master_port=29701 models/build_models.py -C configs/pretrain/pretrain_150m.yaml' > logs/train150m.log 2>&1 < /dev/null & sleep 3; tail -f logs/train150m.log
```

**四个必须看的数**（全在启动/前几步日志里）：
1. `[model] 参数量 = 151,908,864`（±几千以内）
2. `[data] ... 每次参数更新 = 65,536 tokens` / `1 个 epoch = 150,372 步`
3. 第 1 步 `loss ≈ 10.8`（= ln(50257)），50 步内应降到 ~9.x
4. `tok/s` → 用 `3.04e9 / tok/s` 估算 46k 步要多久，决定跑 20k 还是 46k

> ⚠️ 小提示：`[budget]` 那行只在**正式训练**时打印（`--dry-run` 在它之前就 return 了）。
> 想让 dry-run 也打，把 `out_dir`/`budget` 那段挪到 `if args.dry_run` 之前即可。

---

## 6. 实验协议（规范，别省）

| 项 | 要求 |
|---|---|
| 对照变量 | 一次只改一个（先只放大到 152M，别同时上 MLA/SWA） |
| token 预算 | 所有配置**相同**（都用 20k 步 或 都用 46k 步） |
| seed | ≥ 2 个（小模型 acc 波动 ±3 个点，单 seed 会盖过架构差异） |
| 指标 | val_loss / **BPB**、HellaSwag acc/acc_norm、tok/s、峰值显存 |
| 评测样本数 | 固定（如 HellaSwag 1000 条、LAMBADA 1000 条） |
| 记录 | 每次 run 存 `config` + git commit + 硬件 + 时长（日志里已有 config，建议在 commit 里带上 run id） |

**训练完评测**：

```bash
python scripts/eval/downstream_eval.py \
  --config configs/pretrain/pretrain_150m.yaml \
  --ckpt runs/pretrain/gqa-18L-768d_step19999.pt \
  --limit 1000
```

（ckpt 文件名里的 `18L-768d` 是 `tag` 自动生成的，不用手改。）

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| OOM | B/T 太大 | 降 `micro_batch`，或加 §4.4 checkpointing |
| `loss nan` | fp16 下溢 | 加 §4.2 GradScaler；还不行就 fp32 |
| 启动后几分钟没动静 | 昇腾按 shape 编译算子图 | **正常**，第一个 step 慢，之后恢复 |
| 生成样例还是慢 | 生成在 NPU 上每个新长度都重编译 | 已经改成 CPU 副本（`print_sample`），如果慢说明没生效 |
| 参数量对不上 | 别名没生效 / vocab 不是 50257 | 按 §4.1 打印 attn/ffn 的形状确认 |
| 磁盘不够 | ckpt 太多 | §4.3 保留最近 3 个 |
