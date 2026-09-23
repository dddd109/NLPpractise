# NLP Practice

这是一个面向 Transformer / LLM 结构化实现与性能基准研究的练习项目，聚焦在注意力机制、KV Cache、以及小规模 GPT2 训练实验。

当前仓库中，核心内容包括：

- 多头注意力（MHA）
- 多查询注意力（MQA）
- 变体的注意力模块与基准封装
- 简单的 KV cache 与性能测量脚本
- 小规模 GPT2 / 字符级语言模型训练代码
- 基础 benchmark 框架，用于统计 latency / throughput / FLOPs / memory

---

## 项目结构

```text
nlp/
├── README.md
├── requirements.txt
├── __init__.py
├── src.ipynb
├── benchmark/
│   ├── __init__.py
│   ├── case.py
│   ├── result.py
│   ├── runner.py
│   ├── test/
│   │   ├── __init__.py
│   │   └── test_attention.py
│   └── utils/
│       ├── flop.py
│       ├── memory.py
│       ├── profiler.py
│       └── timer.py
├── models/
│   ├── buildGPT2/
│   │   ├── Shakespeare.txt
│   │   ├── train.py
│   │   └── model/
│   │       ├── __init__.py
│   │       └── module/
│   │           └── __init__.py
│   ├── buildnanogpt/
│   │   ├── fineweb.py
│   │   ├── hellaswag.py
│   │   └── train_gpt2.py
│   └── loopedtransformer/
│       ├── train_baseline.ipynb
│       └── train_looped.ipynb
└── modules/
    ├── __init__.py
    ├── attention/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── MHA.py
    │   └── MQA.py
    ├── cache/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── dynamic.py
    │   ├── dynamicv1.py
    │   └── paged.py
    ├── ffn/
    │   ├── __init__.py
    │   ├── GeLU.py
    │   └── SWiGLU.py
    ├── moe/
    │   └── MoE.py
    ├── normalize/
    │   └── RMSNorm.py
    └── positional/
        └── RoPE.py
```

---

## 主要模块说明

### 1. 模块实现

`modules/` 是核心实现层，包含：

- `modules/attention/`
  - `MHA.py`: 多头注意力实现
  - `MQA.py`: 多查询注意力实现，减少 KV 头数
  - `base.py`: 注意力抽象基类
- `modules/cache/`
  - `base.py`, `dynamic.py`, `paged.py`: KV cache 及其实现
- `modules/positional/RoPE.py`
  - 旋转位置编码实现
- `modules/normalize/RMSNorm.py`
  - RMSNorm 归一化层
- `modules/ffn/GeLU.py`, `SWiGLU.py`
  - 前馈网络变体

这些实现适合用于理解从基础 Transformer 块到更高效注意力设计的演进过程。

### 2. Benchmark 框架

`benchmark/` 目录中包含一个轻量 benchmark 工具，用于衡量：

- 延迟（latency）
- 吞吐（throughput）
- 理论/估算 FLOPs
- 内存占用和 KV cache 估计
- profiling 相关信息

关键文件：

- `benchmark/case.py`: 基准用例定义
- `benchmark/result.py`: 结果数据结构
- `benchmark/runner.py`: 运行基准测试
- `benchmark/utils/`: flops, memory, timer, profiler 工具

### 3. 训练与实验

- `models/buildGPT2/train.py`: 使用 Shakespeare 文本做字符级 GPT2 风格训练
- `models/buildnanogpt/`: 更接近 nanogpt 风格的训练脚本与数据处理
- `models/loopedtransformer/`: looped transformer 实验笔记本

---

## 环境依赖

项目依赖在 `requirements.txt` 中，主要包括：

```bash
pip install -r requirements.txt
```

当前依赖：

- torch==2.6.0
- numpy
- tqdm
- tiktoken
- datasets
- transformers
- requests

如果你使用 CUDA 环境，注意力 benchmark 和训练脚本会优先在 GPU 上执行；若没有 GPU，则会自动回退到 CPU。

---

## 快速开始

### 1) 运行注意力 benchmark 测试

```bash
python -m benchmark.test.test_attention
```

该测试会构造一个简单的 MQA 模型，并跑一组 benchmark 相关逻辑，验证功能和计算路径是否正常。

### 2) 直接运行 MQA 模块

```bash
python -m modules.attention.MQA
```

这个脚本中内置了几个简单性能对比示例（例如 MQA with rope / without rope / MHA），可直接用于观察不同注意力形式的时间开销。

### 3) 训练字符级小模型

```bash
cd models/buildGPT2
python train.py
```

该脚本会读取 `Shakespeare.txt`，训练一个小型字符级语言模型并生成文本。

---

## 设计目标

这个项目的目标不是一个完整的大型框架，而是一个“学习型”实现库，适合用于：

- 理解 Transformer 的结构与实现细节
- 对比 MHA / MQA / 其他注意力设计的计算差异
- 学习 KV cache 的实现和其对长上下文场景的影响
- 研究 benchmark / profiling / FLOPs 分析的方法
- 作为从零实现 LLM 相关组件的实验模板

---

## 适合的研究方向

如果你继续扩展这个项目，可以重点探索：

- MQA / GQA / MLA 与标准 MHA 的性能差异
- KV cache 的动态与分页实现
- RoPE、RMSNorm、SwiGLU 的组合效果
- 更贴近 GPT2 / NanoGPT 的训练框架封装
- 不同 attention 形态下的 throughput、memory、算力利用率对比

---

## 说明

本项目更偏向“研究与实验型代码”，而不是生产级框架。代码中包含大量说明性实现、实验脚本和学习笔记风格的结构，适合用于：

- 自学 LLM 相关工程实现
- 复现经典 Transformer 组件
- 做注意力机制研究和性能对比

