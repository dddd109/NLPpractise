
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from ffn.SWiGLU import SwiGLU
from normalize.RMSNorm import RMSNorm
from positional.RoPE import RoPE

class A(nn.Module):
    r"""
    Multi-Qeury-Attention
    通过减少K/V头数来降低计算量
    
    Args:
        n_embd :int 嵌入维度大小 注释中为C或n_embd
        block_size :int 时间步T
        n_heads,kv_heads : int 分别表明Q和KV的头数,如果kvhead为None就是普通MHA
        rope : bool = True 是否使用rope 默认True
    """
    def __init__(self,n_embd:int,block_size:int,
                 n_heads:int,kv_heads:int =None,
                 rope:bool=True):
        super().__init__()
        assert n_embd%n_heads==0
        self.n_embd = n_embd
        self.rope = RoPE(dim=n_embd//n_heads) if rope is not None else None
        
        if kv_heads is None:
            self.n_head = n_heads
            self.qkv = nn.Linear(n_embd,3*n_embd)

        else: # q_C//n_head == kv_C//self.kv_head
            self.n_head = n_heads
            self.kvhead =kv_heads
            self.q = nn.Linear(n_embd,n_embd)
            self.kv = nn.Linear(n_embd,2*n_embd//self.n_head*self.kvhead)
            self.kv_ch = n_embd//self.n_head*self.kvhead
            
        self.proj = nn.Linear(n_embd,n_embd)
        self.register_buffer('mask',torch.tril(torch.ones(block_size,block_size)
                                               .view(1,1,block_size,block_size)))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C = x.size()# B,T,C
        if self.kvhead is not None:
            q = self.q(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
             # B,T,C -> B,T,n_head,C//n_head -> B,n_head,T,C//n_head
            k,v = self.kv(x).split(self.kv_ch,dim =-1) # B,T,C*kvhead//n_head 
            # 要兼容GQA和MQA

            k = k.view(B,T,self.kvhead,self.n_embd//self.n_head).transpose(1,2)\
                # .repeat_interleave(self.n_head//self.kvhead,dim=1)
                #  # B,kvhead,T,C//n_head
            v = v.view(B,T,self.kvhead,self.n_embd//self.n_head).transpose(1,2)\
                # .repeat_interleave(self.n_head//self.kvhead,dim=1)
                #  # B,kvhead,T,C//n_head
            
        else:
            qkv = self.qkv(x)# B,T,3*n_embd
            q,k,v = qkv.split(self.n_embd,dim = 2) #B,T,C
            q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,T,nh,head_size --> B,nh,T,C/nh
            k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,C//nh
            v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        if self.rope:
            cos,sin = self.rope.get_cos_sin(T,q.device,q.dtype)
            q,k = RoPE.apply_rotary_emb(q,k,cos,sin)

        attn = (q@k.transpose(-1,-2))*(1.0/math.sqrt(k.size(-1))) # B,nh,T,T
        attn = attn.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        attn = F.softmax(attn,dim=-1) 
        weight = weight@v # ...,T,T@..,T,C//nh -> ...,T,C//nh

        weight = weight.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(weight)