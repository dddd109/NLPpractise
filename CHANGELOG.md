# NLPpractise 改动日志（CHANGELOG）

> 记录 2026-10-13 ~ 2026-10-16 在 `refactor/llm-system` 分支上对
> `runtime/`、`modules/`、`models/build_models.py`、`configs/` 的改动。
> 原则：**只改自己负责的文件；用户已暂存（staged）的改动一律不动**。
> 环境：昇腾 910ProB × 8，CANN 9.1.1，torch 2.10.0+cpu，torch_npu 2.10.0.post6。

---

## 提交总览

| commit | 概要 |
|---|---|
| `0e8a211` | 存底：补充 `.gitignore` 数据/日志规则；新增 fineweb-edu 下载与 GPT-2 分词脚本 |
| `b961273` | 新增 `runtime/`（device + config）；`GQA`、`MLA`、`MLAKVCache`；配置样例 |
| `e8e8910` | 新增可训练的 MoE 前馈层 `MoEFFN`；配置入口支持 `ffn: moe` |
| `88cd025` | `GQA` 去掉 `repeat_interleave` 拷贝（多后端可选 + 掩码缓存）；新增 `runtime/dist.py` 多卡并行；`build_models` 支持 DDP |
| `061009b` | `build_models` 增加 `--dry-run`（8 卡链路自检，不更新权重）；更正卡 3 说明 |

---

## 改动明细

### `runtime/device.py`（新增）

**目的**：屏蔽"开发机是 CUDA、训练机是昇腾 NPU"的差异。

- **在 `import torch` 之前**判断 CANN 环境：没加载则先 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，
  避免 `import torch` 因 torch_npu 加载失败而直接崩；要用 NPU 时 `require_npu()` 给出带修复步骤的报错。
- 统一函数：`get_device / set_device / synchronize / manual_seed / seed_all / empty_cache /
  memory_allocated_gb / max_memory_allocated_gb / device_info / visible_devices / summary`。
- `autocast_dtype(device)`：**昇腾只支持 fp16**，传 bf16 会被 torch **静默关掉**；
  所以 NPU→`torch.float16`、CUDA→`torch.bfloat16`。`autocast()` 用这个结果。
- 实测：不 source CANN 时 `import runtime.device` 本身不崩、`npu_available()==False`、
  `require_npu()` 抛"请先 source set_env.sh"；source 后识别 8 张 `Ascend910ProB`。

### `runtime/config.py`（新增 + 后续兼容）

**目的**：单一配置入口（`dataclass 默认值 < YAML(支持 extends) < 命令行 --set`）。

- `Config`：`model / train / data / run` 四个 section；`dumps()/dump()/to_dict()/get(dotted)`。
- 为了和你的 yaml/代码兼容，额外支持了：
  - **字典式取值**：`cfg["model"]["n_embd"]`（加了 `__getitem__/__setitem__/__contains__/keys/items`）。
  - **段名别名**：`training:` ↔ `train:`（`SECTION_ALIASES`）。
  - **字段名别名**：`hidden_size→n_embd`、`context_length→block_size`、
    `num_attention_heads→n_head`、`num_key_value_heads→n_kv_head`、
    `intermediate_size→ffn_hidden_dim`、`num_hidden_layers→n_layer`、`max_position_embeddings→block_size`。
  - `position_embedding: rope` → `rope: True`（`normalize_model_cfg`）。
  - 架构名：`arch` 或 `attention` 都认（`select_arch`）。
- 注册表：`build_attention`（`mha/mqa/gqa/mla`）、`build_ffn`（`gelu/swiglu/moe`）。
  **加新架构 = 加一个类 + 注册一行 + 加一个 yaml**，训练/benchmark 代码不动。

### `runtime/dist.py`（新增，250 行）

**目的**：多卡 DDP 封装（和 `device.py` 一个思路，把环境差异关进 runtime）。

- 后端自动：NPU→`hccl`、CUDA→`nccl`、CPU→`gloo`（`RUNTIME_DIST_BACKEND` 可覆盖）。
- `setup()`：**先 `torch.npu.set_device(local_rank)` 再 `init_process_group`**（昇腾硬要求）；
  `device_id=` 消 barrier 警告并带 `TypeError` 回退；没 torchrun 变量时退化为单进程上下文。
- `wrap_model()`：DDP；`broadcast_buffers=False`（RoPE 的 `theta_arr`、因果掩码是推导常量，
  每步广播纯浪费一次通信）；只给 CUDA 传 `device_ids`。
- `all_reduce_mean()`：用 **SUM÷world_size**（部分 hccl 不支持 `ReduceOp.AVG`）；返回 detach 副本。
- `no_sync()`（梯度累积不通信）、`barrier()`、`rank0_print()`、`global_batch_tokens()`、`cleanup()`。

### `modules/attention/GQA.py`（重写，`88cd025`）

**目的**：解决"`repeat_interleave` 复制 K/V 浪费显存"。

- 新增 `GQAConfig.attn_impl`：`auto | sdpa | manual | manual_repeat`。
- `sdpa_gqa_available(device_type)`：按设备类型**真跑一次**探测（CPU 过、该昇腾环境抛错），缓存结果。
- `forward` 拆分三个注意力实现：
  - `_attn_sdpa`：`F.scaled_dot_product_attention(..., enable_gqa=True)`（GQA 核内原生，
    不复制 K/V、不物化 T×T）；预填充用 `is_causal`，带 cache 时传显式 mask。
  - `_attn_manual`：**默认实现，去掉 repeat_interleave** ——
    `(q*scale).view(B,Hkv,G,T,d) @ k.unsqueeze(2).transpose(-1,-2)` 广播，算完 view 回去。
  - `_attn_manual_repeat`：保留旧实现用于 A/B 对比。
- 小优化：`scale` 折进 q（省一次 T×T 逐元素乘）；掩码改用 `CausalMaskCache`。
- 接口不变（`forward(x, position_offset=0, kv_cache=None)`）、参数名仍 `q/kv/proj`，
  因此仍能与 `MQA` 互载权重做等价性对拍。

### `modules/attention/masking.py`（新增）

- `CausalMaskCache.get(q_len, kv_len, offset, device)`：只缓存最近一份掩码，
  不再每层每步 `ones().tril()`；注释写明 `tril(diagonal=offset)` 的来历（cache 解码时偏移）。
- `neg_inf(dtype)`：返回 `finfo(dtype).min`。**不要用 `-inf`** —— 昇腾 `masked_fill(-inf)`
  会触发不支持的 `isneginf` 算子 → 每次回退 CPU 并同步。

### `modules/attention/MLA.py`（小幅）

- 掩码改用 `CausalMaskCache`；`-inf` 改用 `neg_inf(att.dtype)`。

### `modules/cache/MLAKVCache.py`（`b961273`）

- MLA 专用**单槽** latent cache（只缓存 `[c_kv, k_pe]`），带 `memory_bytes()` 计账；
  说明为什么和 `BaseKVCache` 的双槽签名不同（MLA 没有独立的 V）。

### `modules/moe/MoEFFN.py`（`e8e8910`）

**目的**：原 `modules/moe/MoE.py` 的 `BasicMoE` 有 bug（`ModuleList` 用 tensor 索引 + `argmax` 不可导，
router 不训练），故新增可训练的实现。

- top-k 路由 + 只在选中专家上重新归一化；负载均衡 aux loss（Switch 式）；
  router z-loss（可选）；共享专家（DeepSeekMoE 式，可选）；`router_jitter`（可选）。
- **发现的一个性质**（写在 `__main__` 里）：`aux = E·Σ f_i·P_i` 只衡量 f 对 P 的偏离，
  所以 router 输出均匀 + top-k 打平时 **aux 仍是 1.0 但只有一个专家在干活**（f 退化成 one-hot）。
  必须看 `last_usage`，不能只看 aux。
- 参数/激活量/FLOPs 计账：`describe()`、`active_param_count()`、`flops_per_token()`。
- 文件头注明 `BasicMoE` 为何不可用。

### `modules/__init__.py` 与各子包 `__init__.py`

- `modules/attention/__init__.py`：补导出（`Attention/MHA/MQA/GQA/MLA`）。
- `modules/normalize/__init__.py`、`modules/positional/__init__.py`：**原来没有**，补上。
- `modules/cache/__init__.py`：补导出 `MLAKVCache`。
- `modules/moe/__init__.py`：添加 `MoEFFN`，并在 docstring 里 warn `BasicMoE` 不可用。

### `models/build_models.py`（你已有的文件，改动较大）

- 顶部加 `sys.path` 修正（脚本方式 / torchrun / `-m` 都能跑）；相对导入改绝对导入。
- `torch_dtype()`：`"torch.float16"` 字符串 → `torch.dtype`。
- `parse_args()`：`-C` + 一批临时覆盖参数 + `--use-cache / --verbose-init / --dry-run`。
- `resolve_config()`：配置路径按 `__file__` 解析到仓库根，不再依赖 cwd。
- 新增 `Block`（pre-norm + attn + pre-norm + FFN + 残差）。**注意**：原 `Model.forward`
  没有残差、没有 FFN、还在每个 block 前重复 norm 残差流；这里按主流写法组装。
- `Model`：`embedd/blocks/norm/proj`，支持 `tie_embeddings`；KV cache 改 `--use-cache` 显式开启；
  `_init_weights` 默认只打一行汇总（`--verbose-init` 才逐条），RMSNorm 的 `gamma` 初始化为 1。
- `forward(idx, targets=None, position_offset=0)` → 返回 `(logits, loss)`。
- `generate`：cache 路径只喂新 token、`position_offset=0`（偏移由 cache 长度推，避免重复计数）。
- `DataLoaderLite`：`np.memmap(dtype=uint16)` 读你的 `.bin`（原实现 `np.load` 读 `.npy`，对不上）；
  多卡分片 `start = r*B*T`、`stride = world_size*B*T`；`reset()` 保证评估可复现。
- `lr_at()` warmup + cosine；`evaluate()` 多卡求平均（**loss 建在设备上**，hccl 只能同步设备张量）。
- 训练循环：梯度累积 + `no_sync`、loss 多卡求平均、只 rank0 打印/写盘、`barrier()` 保护 ckpt。
- 新增 `dry_run_check()`：建组→建模型→DDP→**一次前向+反向**→校验各 rank 梯度已同步；
  不调用 `optimizer.step()`，权重不变（对应 `--dry-run`）。

### `configs/`

- `configs/train/base.yaml`：训练/数据/运行默认值；`dtype: auto`（由 `runtime.device` 按后端选）。
- `configs/model/gqa_small.yaml`、`mla_small.yaml`、`moe_small.yaml`：架构样例（`extends: ../train/base.yaml`）。
- 你的 `configs/pretrain/pretrain_llama.yaml`：**未改动、仍在暂存区**。

---

## 实测数据（昇腾 910ProB，fp16）

### GQA：`repeat_interleave` vs 不复制（注意力本身，稳态中位数）

| 形状 | 前向 | 前向+反向 | 峰值显存 |
|---|---|---|---|
| B=1 H=8 Hkv=4 T=1024 | **−7.8%** | −0.7% | **−4.5%** |
| B=8 H=8 Hkv=4 T=1024 | +0.7% | −1.6% | −2.5% |
| B=1 H=16 Hkv=2 T=2048 | **−4.7%** | −4.5% | −1.4% |

（数值与 `repeat_interleave` 完全等价：整层前向 `max|Δ|=0.000e+00`。）

**端到端 12 层 block（B=2 T=256，warmup 后）**：前向 45.3→44.8 ms（−1.1%），前+反 128.8→128.8 ms（±0%）。

> 结论：`repeat_interleave` 只值 `2*B*H*T*D*2` 字节（该配置下 <2 MB），端到端基本看不出。
> 真正的大头是**没被融合的 T×T 分数矩阵**（B=8 时 128 MB/份）。要解决它只能靠融合注意力，
> 而本机 `aclnnFlashAttentionScore / PromptFlashAttention` **因 inferShape 未注册而不可用**（环境问题，见下）。

### `masked_fill(-inf)` vs `finfo.min`

- 数值等价（`max|Δ|=0.000e+00`）；延迟 0.510→0.500 ms（**−2%**）。
- 意义：少一次 `isneginf` 的 CPU 回退，但不是性能主因（已实测）。

### 8 卡链路自检（`--dry-run`，未训练）

```
[dry-run][rank 0] device=npu:0  数据起点 token[0:128]    x.sum()=689089
[dry-run][rank 3] device=npu:3  ...                     ← 卡3 Health=Critical 但计算正常
[dry-run][rank 6] device=npu:6  数据起点 token[768:896]  x.sum()=571144
[dry-run][rank 0..7] loss≈10.82~10.88  loss(全局平均)=10.8498
                     |grad - 全局平均grad| = 0.000e+00  OK（DDP 同步生效）
[dry-run] 链路检查完成，未做任何参数更新（没有 optimizer.step）
```

- 卡 3 单独实测：`matmul ok, sum=25251.50` + `GQA 前向 ok` → Health 的 `Critical` 只是状态标志，计算正常。
- **8 卡可用，无需排除任何卡**（之前"卡 3 异常先排掉"的判断已更正）。

---

## 已知问题 / 待办

1. **融合注意力不可用（环境问题，非代码）**：
   `npu_fusion_attention` / `npu_prompt_flash_attention` / `F.scaled_dot_product_attention`(enable_gqa)
   在本机全部失败，报 `The inferShape function does not exist` / `can not cast format when output is input`。
   属 CANN 算子包缺失或 torch_npu 与 CANN 版本不配套。**需管理员处理**；修好后 `attn_impl: auto` 自动走 SDPA，
   显存/延迟都会再降。
2. `runs/`（训练日志/ckpt 输出目录）**未加入 `.gitignore`**（我建议加 `/runs/`，暂未改）。 ← 待确认
3. `modules/moe/MoE.py` 的 `BasicMoE` 仍不可用（有 bug，`MoEFFN` 是正确实现）。
4. `modules/attention/MQA.py` 的 kv_cache 路径有两个隐患：
   - 掩码 `tril(T,T)` 只覆盖本次 chunk → 带 cache 单步解码（T=1,kv_len>1）形状/语义错；
   - RoPE 只用了 `position_offset`，没算上已缓存长度 → 解码位置错。
   （GQA/MLA 已修，MQA 未动。）
5. `models/build_models.py` 里 `Block` 是我按"标准 pre-norm + 残差 + FFN"补的，**改动了原架构设计**；
   若有自己的设计，请回退这一段。
6. `import modules.attention.GQA as g` 会拿到 **类**而不是子模块（因为 `__init__.py` 里 `from .GQA import GQA`
   把子模块名覆盖了）；一般用 `from modules.attention import GQA` / `from modules.attention.GQA import GQA` 即可。

---

## 复现 / 自检命令

```bash
# 硬件/环境
python -m runtime.device
python -m runtime.dist

# 配置入口
python -m runtime.config --config configs/model/gqa_small.yaml

# GQA 三种实现一致性 + 与 MQA 等价性
python -m modules.attention.GQA

# 8 卡链路自检（不训练，权重不变）
conda activate nlp-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
torchrun --nproc_per_node=8 models/build_models.py -C configs/pretrain/pretrain_llama.yaml \
  --dry-run --seq-len 128 --micro-batch-size 1 --grad-accum 1
```

---

## 外部无关目录说明

`/home/gdut/projects/npu_portability`（在仓库外、非 git）：独立的"NPU portability kit"，
用来自证某个审计/论文的 C1（CRPS 估计偏置）与 C6（输出分布反转排序）结论。
与 `NLPpractise`、本日志记录的任何改动**无关**，也未被我修改。
