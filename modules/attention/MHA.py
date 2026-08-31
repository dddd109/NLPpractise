import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..positional.RoPE import RoPE

class MultiHeadAttention(nn.Module):
    """
    实现的多头注意力
    n_embd : int 每个token的嵌入维度 ,通道数C
    n_head : int 多头的头数
    block_size : int 总的上下文时间步 T
    """
    def __init__(self,n_embd,n_head,block_size,rope:bool=False):
        super().__init__()
        assert n_embd%n_head ==0
        self.c_attn = nn.Linear(n_embd,3*n_embd)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size))
                             .view(1,1,block_size,block_size))
        self.proj = nn.Linear(n_embd,n_embd)
        self.n_embd =n_embd
        self.n_head =n_head
        if rope is not None:
            self.rope = RoPE(n_embd//n_head)
        else:
            self.rope = None
    def forward(self,x:torch.Tensor)->torch.Tensor: # B,T,n_embd
        B,T,C = x.size()
        qkv = self.c_attn(x)# B,T,3*n_embd
        q,k,v = qkv.split(self.n_embd,dim = 2) #B,T,C
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,T,nh,head_size --> B,nh,T,C/nh
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,C//nh
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) 

        if self.rope:
            cos,sin = self.rope.get_cos_sin(T,q.device,q.dtype)
            q,k = RoPE.apply_rotary_emb(q,k,cos,sin)

        weight = (q@k.transpose(-1,-2)) *(1.0/math.sqrt(k.size(-1))) # B,nh,T,T
        weight = weight.masked_fill(self.mask[:,:,:T,:T] == 0 ,float('-inf'))
        weight = F.softmax(weight,dim=-1)
        weight = weight@v # B,nh,T,C//nh

        # wegiht = F.scaled_dot_product_attention(q,k,v,is_causal=True) 等同于上述四行
        weight = weight.transpose(1,2).contiguous().view(B,T,C) # B，T,C
        return self.proj(weight)

