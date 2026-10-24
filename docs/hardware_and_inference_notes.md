# 硬件能力 / SWA / 投机解码 / 架构定位 笔记

> 记录于 2026-10-24。回答四个问题：
> ① 8 卡 910ProB 能训多大模型；② 这套设施能做多大的项目；
> ③ SWA 是什么；④ 投机解码在 DeepInfer 里在哪、怎么理解；⑤ 当前架构老不老。
>
> ⚠️ 文中所有数字都标了来源性质：**【实测】**= 这台机器上跑出来的；**【估算】**= 按公式推的；
> **【官方】**= DeepSeek 自报。估算值只用于判量级，别当结论引用。

---

## 1. 你这 8 张卡能训多大的模型

### 1.1 先把显存账算清楚【估算】

每张卡 **HBM 32 GB**（`npu-smi` 实测 32768 MB），8 卡共 256 GB。

混合精度 + AdamW、不分片（纯 DDP）时，**每 1 个参数**要占：

| 项 | 字节 |
|---|---|
| bf16/fp16 权重 | 2 |
| bf16/fp16 梯度 | 2 |
| fp32 master 权重（可选） | 4 |
| Adam 一阶/二阶动量（fp32） | 8 |
| **合计** | **12 ~ 16** |

单卡 32 GB，扣掉激活值 / 碎片 / workspace（按 25~40% 余量算），能放状态的空间约 **20~24 GB**：

| 模型规模 | 状态内存（16 B/param） | 纯 DDP 可行性 |
|---|---|---|
| **61M**（你现在） | 0.98 GB | 轻松，每卡只用了 ~3% |
| 150M | 2.4 GB | 轻松 |
| 300M | 4.9 GB | 可行 |
| **500M** | 8.2 GB | 可行（还要留激活，建议开 gradient checkpointing） |
| 1B | 16.4 GB | **紧张**，必须开 checkpointing，序列别太长 |
| 3B | 49 GB | DDP 放不下 → 要 ZeRO-2/3 或 FSDP |
| 7B | 115 GB | 必须全分片 + checkpointing，速度会很惨 |

**结论：纯 DDP（最稳、昇腾支持最好）的安全上限 ≈ 500M~1B。**
想上 3B+ 就得引入 ZeRO/FSDP —— 但昇腾上 DeepSpeed/FSDP 的成熟度低于 CUDA，
踩坑成本高，**不建议为课程级实验付这个代价**。

### 1.2 数据账：真正卡住你的是数据，不是显存【估算】

Chinchilla 的经验法则：**tokens ≈ 20 × 参数**。你现在的语料：

```
train.bin ≈ 9.86B tokens   (14 个 fineweb-edu 分片全下完)
9.86e9 / 20 ≈ 4.93e8
```

→ **Chinchilla 最优点大约在 500M 参数**。也就是说：
- 训 **≤500M**：数据够，甚至有余（可以跑多个 epoch）
- 训 **1B**：需要 20B tokens，你只有一半 → 数据饥饿，再加参数主要是在"记住"而不是"泛化"
- 训 60~150M：相对算力是"训练不足"（undertrained），但**做架构对比完全够用**，而且快

### 1.3 时间账：别拿小模型的 tok/s 线性外推【估算】

正确算法只有一个 —— **拿日志里的实测 tok/s 去除**：

```
总训练时间 ≈ 总 tokens / 实测 tok/s          # 日志每步都会打 tok/s
```

参照系（量级感）：8×910ProB 的 bf16 稠密算力与 8×A100 大致同一档；
社区里"nanoGPT 用 8×A100 训 GPT-2 124M / 10B tokens ≈ 4 天"是个可用的标尺：

| 目标 | 数据量 | 粗估时间 |
|---|---|---|
| 61M | 1.3B（= 你现在的 2 万步） | **≈ 18 h**【实测推算】 |
| 150M | 3B | ~1.5 天 |
| 500M | 10B（1 个 epoch） | **~2 周** |
| 1B | 20B | ~2 个月 |

⚠️ 注意：**小模型的 tok/s 会严重低估大模型的效率**（小模型是 kernel launch 受限，
不是算力受限）。你现在 61M 单卡 0.25 ms/token 的实测值不能直接乘参数倍率去推 500M。

---

## 2. 这套设施能做到多庞大的项目

**能做的（建议范围）**

| 项目 | 可行性 |
|---|---|
| 60~150M 模型的**架构消融**（GQA / MLA / MoE / SWA × 2~3 seed） | ✅ 最合适，几天级 |
| **推理侧优化**研究（KV cache 策略、投机解码、窗口注意力） | ✅ 不需要大模型，正合适 |
| 数据流水线（下载 → 分词 → 打包 → 评测） | ✅ 已跑通 |
| 500M 单次预训练（1 epoch，~10B tokens） | ✅ 但要 2 周，且只够跑 1~2 组 |
| 多机 / 多节点实验 | ❌ 单机 HCCL，没跨节点 |
| 7B 级有意义预训练 | ❌ 显存和时间都不够 |
| 在线 serving 压测（真实 QPS/SLA） | ❌ 没有线上流量，只能做离线吞吐 |

**一个能一周内交差的实验矩阵**

| # | 模型 | tokens | 目的 | 指标 |
|---|---|---|---|---|
| A | GQA 60M（基线） | 1.3B | 参考点 | val loss / BPB / HellaSwag / tok/s / 峰值显存 |
| B | GQA + SWA(W=256) 60M | 1.3B | 窗口注意力对质量/长上下文的影响 | 同上 + KV bytes/token |
| C | MLA 60M | 1.3B | 低秩压缩换 cache | 同上 |
| D | GQA + MoE(E=8, top2) 150M | 3B | 激活参数 vs 质量 | 同上 + 激活参数量 |
| E | 投机解码（不改模型） | — | 推理加速 | 接受率、tokens/s |

**规范要求**（否则结论不可信）：同 token 预算、固定 eval limit、≥2 个 seed、报告 BPB 而不只是 loss。

---

## 3. SWA 是什么

### 3.1 定义

**SWA = Sliding Window Attention（滑动窗口注意力）**。
标准因果注意力里，位置 t 的 token 能看到 0..t 的全部历史；SWA 里它**只看最近 W 个**：

```
允许的 key j 满足：  t - W + 1  ≤  j  ≤  t
```

| | 标准因果注意力 | SWA（窗口 W） |
|---|---|---|
| 计算量 | O(T²) | **O(T·W)** |
| KV cache 大小 | **随 T 线性增长** | **上界 W，不涨** |
| 长文本 | 贵到不可行 | 可行（代价是只看局部） |

Mistral 7B（W=4096）、Longformer、Gemma 2 的"局部-全局交替层"都是这个思路。
实践中常**混合**：大部分层用 SWA，少数层用全注意力，兼顾局部细节和长程依赖。

> ⚠️ 术语撞车：机器学习里 "SWA" 也常指 **Stochastic Weight Averaging**（权重平均）。
> 这里说的都是 **Sliding Window Attention**，看上下文区分。

### 3.2 DeepInfer 的 "SWA Bounded Replay" 省的是什么【官方】

它不是新的注意力形式，而是**针对已经用了 SWA 层的模型做推理优化**：

- 朴素做法：解码时为了算一个 token，需要把整段前缀重新前向（或重算整个窗口）
- 他们的做法：因为 SWA 层只关心最后 W 个 token，**解码时只需重放窗口内的部分**
- 官方数字：KV 重算从"最后 2560 个 token"降到"最后 ~128 个"（有报道写 140）
  → **prefill 总计算量几乎减半**

一句话：**SWA 让 cache 有上界，Bounded Replay 就是"既然只看窗口，就别重算窗口外的东西"。**

### 3.3 在你的代码里怎么加（很小）

你的掩码集中在 `modules/attention/masking.py::CausalMaskCache`，现在是：

```python
mask = torch.ones(q_len, kv_len, device=device, dtype=torch.bool).tril(diagonal=offset)
```

加窗口只需要多一个 `triu`：

```python
mask = mask & torch.ones(q_len, kv_len, device=device, dtype=torch.bool).triu(
    diagonal=offset - window + 1
)
```

然后把 KV cache 改成**环形缓冲**（只保留最近 W 个），就能拿到"cache 有上界"的收益。
这个实验和你的 GQA/MLA 对比天然契合 —— 指标直接看 **KV bytes/token** 和长文本下的峰值显存。

---

## 4. 投机解码（Speculative Decoding）

### 4.1 原理：草稿 → 一次验证 → 接受前缀

自回归解码的瓶颈是"每生成 1 个 token 就要跑一次完整前向"（memory-bound）。
投机解码换成：

1. **草稿**：用一个很便宜的方式先猜出 `k` 个后续 token（小模型 / n-gram / MTP 头）
2. **验证**：把 `[前缀 + 这 k 个草稿]` 一次性喂给大模型 —— **1 次前向就能同时校验 k 个位置**
3. **接受**：从前往后，逐个用大模型的分布判定是否接受；接受 `m` 个（`m ≤ k`），
   剩余的回滚，再从大模型分布采 1 个
4. 净收益：一次前向产出 `m+1` 个 token

只要用标准的拒绝采样（rejection sampling），**输出分布与原模型完全一致（无损）**；
`m` 越大收益越高。理论上限接近 `k+1` 倍，实际常见 1.5~2.5×。

### 4.2 在 DeepInfer 里的位置（仓库文件树，2026-10-20 首版）

官方新闻里叫 **DSpark**；代码里的命名是 **`dflash` / `mtp`（multi-token prediction）** 这一套：

| 文件 | 作用（按命名推断） |
|---|---|
| `deepinfer/model_executor/models/mtp/dflash.py` | **草稿/投机解码的头**（核心） |
| `deepinfer/model_executor/models/v4_modules/heads.py` | 各类预测头（含置信度头） |
| `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` | 变长 MTP 调度（决定验证哪些草稿） |
| `deepinfer/model_executor/layers/tilelang_ops/store_mtp_info.py` | 草稿信息写回 |
| `deepinfer/worker/dflash_cuda_graph_runner.py` | 草稿阶段的图执行 |
| `deepinfer/worker/model_runner.py` | 前向 / 验证主流程 |
| `deepinfer/model_executor/layers/tilelang_ops/sampler.py` | 采样与接受判定 |
| `deepinfer/sequence.py` | 序列状态（草稿 token 挂在这） |
| `deepinfer/model_executor/layers/tilelang_ops/swa_index.py` | SWA 的索引/重放（配合 Bounded Replay） |

> 说明：路径来自 GitHub 文件树接口（`api.github.com/repos/deepseek-ai/DeepInfer/git/trees/main?recursive=1`），
> 我没有逐个文件精读，"作用"列是按文件名 + 官方描述推断的。要精读建议从 `mtp/dflash.py` 开始。

官方数字【官方】：草稿一次提 5 个，置信度头决定哪些值得验证 → 待验证草稿量 **−30%**，
生产负载下吞吐 **+4%~7%**。

### 4.3 在你这台机器上怎么做一个最小版（推荐做，性价比最高）

不需要 V4.1，也不需要 SM100：用你 61M 的模型 + 一个**免费草稿**就能复现机制。

```
草稿：从当前前缀里做 n-gram 查找（prompt / 已生成文本里找最长的重复后缀，接上它的后续）
      → 提 3~5 个 token，零模型成本
验证：把 [prefix + 草稿] 送进 model 一次前向，逐位置比对 argmax / 用拒绝采样
接受：记下接受长度 m，回滚多余部分
```

指标：**接受率（m/k 的分布）**、**tokens/s 相对提升**、**每 token 的验证次数**。
好处：完全不动模型结构，和你的 `benchmark/` 框架能直接对接，也是简历上很亮的实验。

---

## 5. 你现在的架构"老"吗

### 5.1 已经有的（其实很新）

| 组件 | 你的实现 | 状态 |
|---|---|---|
| 归一化 | RMSNorm（pre-norm） | ✅ 现代（LLaMA 系） |
| 位置编码 | RoPE | ✅ 现代 |
| 激活/FFN | SwiGLU | ✅ 现代（LLaMA 系） |
| 注意力 | MHA / MQA / GQA / MLA | ✅ 覆盖了从基线到 DeepSeek 的谱系 |
| MoE | MoEFFN（top-k + aux + 共享专家） | ✅ 现代 |
| 其他 | 无 bias、tied embeddings、fp16 autocast | ✅ 现代 |

**结论：这不是"老架构"，这是标准 LLaMA-2/3 配方**，比很多教学项目完整得多。
准确的说法是：**模块很新，但组装起来训练的还是 LLaMA-2 那一版**。

### 5.2 相对 2026 前沿（DeepSeek V4.x 这一档）缺什么

| 缺的 | 有没有现成模块 | 值不值得现在补 |
|---|---|---|
| **SWA / 局部-全局交替** | ❌ 需要加（mask 一行 + 环形 cache） | **值得**，直接省 cache、支持长文本 |
| **MLA 接进训练** | ✅ 已有 `modules/attention/MLA.py` | 值得（配置改 `arch: mla` 即可对比） |
| **MoE 接进训练** | ✅ 已有 `modules/moe/MoEFFN.py` | 60M 上没意义，300M+ 再说 |
| **MTP（多 token 预测）** | ❌ | 训练侧正则 + 推理侧草稿头，进阶再做 |
| QK-Norm / attention sink | ❌ | 稳定性优化，小模型看不出差别 |
| 长上下文外推（YaRN/NTK） | ❌ | 只有做长文本实验才需要 |
| 推理侧：投机解码 / PagedAttn / graph | 部分有（PagedKVCache） | **值得**，见 §4.3 |

**在 60M 这个规模上**：MoE 和 MLA 的收益基本被噪声淹没（它们是为几十 B 设计的分摊/压缩手段）；
真正有意义的只有两件事 —— **SWA（改推理成本）** 和 **投机解码（改推理速度）**，
因为它们优化的是"每 token 的代价"，在什么规模都成立。

### 5.3 建议的升级顺序

1. **SWA**：`masking.py` 加 `window` 参数 + 环形 KV cache，和 GQA 基线做对照（1 天）
2. **MLA 接进训练**：只改 yaml `model.arch: mla`，跑 1.3B tokens 对照（已具备条件）
3. **投机解码最小版**：n-gram 草稿 + 一次验证（1~2 天）
4. 放大到 300M 后再考虑 **MoE**；**MTP** 放最后

---

## 6. 参考

- DeepInfer 仓库：`github.com/deepseek-ai/DeepInfer`（Apache-2.0，2026-10-20 首版）
- 配套：`github.com/deepseek-ai/deepseek-recipe`（MIT，token 编解码）
- 官方文档：`doc/inference_deployment_qa.md`、`eval/README.md`（仓库内）
- SWA: Mistral 7B (arXiv:2310.06825)、Longformer (arXiv:2004.05150)
- 投机解码：Leviathan et al. 2023 (arXiv:2211.17192)、Chen et al. 2023 (arXiv:2302.01318)
- MoE 负载均衡：Switch Transformer (arXiv:2101.03961)、DeepSeekMoE (arXiv:2401.06066)
- Chinchilla 缩放律：arXiv:2203.15556
