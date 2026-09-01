import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from ..ffn.SWiGLU import SwiGLU
from ..normalize.RMSNorm import RMSNorm
from ..positional.RoPE import RoPE

class MoE(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,x):
        