# NLP

此库用于复现学习NLP以及相关技术
目前 'Andrej Karpathy' 复现的GPT2，包括一些实验的notebook，和直接写为脚本的py文件

```
NLPpractise/
│
├── configs/
│   ├── gpt2.yaml
│   ├── gqa.yaml
│   ├── mqa.yaml
│   ├── mla.yaml
│   ├── moe.yaml
│   └── benchmark/
│
├── modules/
│   │
│   ├── attention/
│   │   ├── base.py
│   │   ├── mha.py
│   │   ├── mqa.py
│   │   ├── gqa.py
│   │   └── mla.py
│   │
│   ├── cache/
│   │   ├── base.py
│   │   ├── dynamic.py
│   │   ├── static.py
│   │   └── paged.py
│   │
│   ├── positional/
│   │   └── rope.py
│   │
│   ├── normalization/
│   │   ├── layernorm.py
│   │   └── rmsnorm.py
│   │
│   ├── feedforward/
│   │   ├── gelu.py
│   │   ├── swiglu.py
│   │   └── moe.py
│   │
│   └── kernels/
│       ├── torch_attention.py
│       ├── flash_attention.py
│       ├── triton_attention.py
│       ├── cuda_attention/
│       └── tilelang_attention.py
│
├── models/
│   ├── gpt2.py
│   ├── llama_like.py
│   └── deepseek_like.py
│
├── data/
│   ├── datasets.py
│   ├── fineweb.py
│   └── ...
│
├── training/
│   ├── train.py
│   ├── optimizer.py
│   └── distributed.py
│
├── evaluation/
│   ├── perplexity.py
│   └── hellaswag.py
│
├── experiments/
│   ├── attention/
│   │   ├── mha_vs_gqa.py
│   │   └── mha_vs_mqa.py
│   │
│   ├── cache/
│   │   ├── dynamic_vs_static.py
│   │   └── paged.py
│   │
│   ├── rope/
│   ├── swiglu/
│   ├── moe/
│   └── mla/
│
├── benchmarks/
│   ├── runner.py
│   ├── latency.py
│   ├── memory.py
│   └── throughput.py
│
├── tests/
│   ├── test_attention.py
│   ├── test_rope.py
│   ├── test_cache.py
│   └── test_models.py
│
└── README.md
```
