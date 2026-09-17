import torch

from modules.attention.MQA import MQA,Config

from benchmark.case import BenchmarkCase
from benchmark.runner import BenchmarkRunner
from benchmark.utils.flop import (
    linear_flops,
    attention_flops,
)
from benchmark.result import WorkloadInfo

device = "cuda" if torch.cuda.is_available() else 'cpu'

B,T,C,H,H_KV = 1,1024,512,16,4
D = C//H
config = Config(
    n_embd=C,
    block_size=T,
    n_head=H,
    kv_head=H_KV,
    rope=False,
)
model = MQA(config=config).to(device=device)

x = torch.randn((B,T,C),device=device)

def calculate_flops():
    qkv_flops = linear_flops(B*T,C,C)+linear_flops(B*T,C,2*H_KV*D)

    attention = attention_flops(B,H,T,T,D)

    output = linear_flops(B*T,C,C)

    return qkv_flops+attention+output

workload = WorkloadInfo(
    name="MQA_B1_T1024",
    batch_size=B,
    input_shapes={
        "x":tuple(x.shape),
    },
    dtype=str(x.dtype),
    device=str(x.device),
    metadata={
        "q_len": T,
        "kv_len": T,
        "hidden_size": C,
        "num_q_heads": H,
        "num_kv_heads": H_KV,
        "head_dim": D,
        "mode": "prefill",
        "num_tokens": B * T,
    }
)

case = BenchmarkCase(
    name="MQA_B1_T1024",
     fn=lambda:model(x),
    #  num_tokens=B*T,
    workload=workload,
    flop_fn=calculate_flops,
    throughput_fn=lambda timing, workload:
    workload.metadata["num_tokens"]
    / (timing.mean_ms / 1000),
)
runner = BenchmarkRunner()
result = runner.run(case=case)
print(result)