import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional,Tuple
from ..ffn.SWiGLU import SwiGLU
from ..normalize.RMSNorm import RMSNorm
from ..positional.RoPE import RoPE

class BasicExpert(nn.Module):
    def __init__(self,
                 featrue_in:int,
                 featrue_out:int):
        super().__init__()
        self.c_fc = nn.Linear(featrue_in,featrue_out)
        self.gelu = nn.GELU('tanh')

    def forward(self,x):
        return self.gelu(self.c_fc(x))

class BasicMoE(nn.Module):
    def __init__(self,featrue_in,featrue_out,num_experts):
        super().__init__()
        self.gate = nn.Linear(featrue_in,num_experts) # ...,C ->...,N
        self.experts = nn.ModuleList([BasicExpert(featrue_in,featrue_out)for _ in range(num_experts)])

    def forward(self,x):
        w = self.gate(x) #...,C -> ...,N
        w = F.softmax(w,dim=-1)
        return self.experts[w.argmax(dim=-1)](x)

