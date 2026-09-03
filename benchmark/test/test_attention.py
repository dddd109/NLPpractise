import torch

from modules.attention.MQA import MQA,Config

from benchmark.case import BenchmarkCase
from benchmark.runner import BenchmarkRunner
from benchmark.utils.flop import (
    linear_flop,
    attention_flops,
)

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
    qkv_flops = linear_flop(B*T,C,C)+linear_flop(B*T,C,2*H_KV*D)

    attention = attention_flops(B,H,T,T,D)

    output = linear_flop(B*T,C,C)

    return qkv_flops+attention+output
case = BenchmarkCase("MQA_B1_T1024",
                     fn=lambda:model(x),
                     num_tokens=B*T,
                     flop_fn=calculate_flops,
                     )
runner = BenchmarkRunner()
result = runner.run(case=case)
print(result)