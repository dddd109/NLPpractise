
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from ..ffn.SWiGLU import SwiGLU
from ..normalize.RMSNorm import RMSNorm
from ..positional.RoPE import RoPE
from ..cache.base import BaseKVCache
from .base import Attention
from dataclasses import dataclass


@dataclass
class Config:
        n_embd:int
        block_size:int
        n_head:int
        kv_head:int =None
        rope:bool=True

class MQA(Attention):
    r"""
    Multi-Qeury-Attention
    通过减少K/V头数来降低计算量
    
    Args:
        n_embd :int 嵌入维度大小 注释中为C或n_embd
        block_size :int 时间步T
        n_head,kv_head : int 分别表明Q和KV的头数,如果kvhead为None就是普通MHA
        rope : bool = True 是否使用rope 默认True
    """
    def __init__(self,
                 config:Config,
                 layer_idx:Optional[int] = None
                 ):
        super().__init__()
        assert config.n_embd%config.n_head==0 ,"Q(KV)头数必须能整除通道数"
        self.n_embd = config.n_embd
        self.rope = RoPE(dim=self.n_embd//config.n_head) if config.rope  else None
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        if config.kv_head is None or config.kv_head==0:
            self.kv_head = config.n_head

        else: # q_C//n_head == kv_C//self.kv_head
            assert self.n_embd%config.kv_head==0,"KV头数必须能整除通道数"
            self.kv_head = config.kv_head
        
        self.q = nn.Linear(self.n_embd,self.n_embd)
        self.kv = nn.Linear(self.n_embd,2*self.n_embd//self.n_head*self.kv_head)
        self.kv_ch = self.n_embd//self.n_head*self.kv_head

        self.proj = nn.Linear(self.n_embd,self.n_embd)

    def forward(self,
                x:torch.Tensor,
                position_offset:int =0,
                kv_cache:Optional[BaseKVCache] = None
                )->torch.Tensor:
        B,T,C = x.size()# B,T,C


        q = self.q(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
            # B,T,C -> B,T,n_head,C//n_head -> B,n_head,T,C//n_head
        k,v = self.kv(x).split(self.kv_ch,dim =-1) # B,T,C*kv_head//n_head 
        # 要兼容MQA MHA GQA
        
        # k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,C//nh
        # v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.kv_head,self.n_embd//self.n_head).transpose(1,2)
        v = v.view(B,T,self.kv_head,self.n_embd//self.n_head).transpose(1,2)

        if self.rope is not None:
            cos,sin = self.rope.get_cos_sin(T,q.device,q.dtype,position_offset)
            q,k = RoPE.apply_rotary_emb(q,k,cos,sin)

        if kv_cache is not None:
            assert self.layer_idx is not None,"使用kvcache要传入层号"
            k,v = kv_cache.update(self.layer_idx,k,v)

        if self.kv_head!=self.n_head:
#-----------------------------------------------------------------------------------------------------------
            k = k.repeat_interleave(self.n_head//self.kv_head,dim=1)  # B,kv_head,T,C//n_head 这是直接物理复制，可优化
            v = v.repeat_interleave(self.n_head//self.kv_head,dim=1) # B,kv_head,T,C//n_head
#-----------------------------------------------------------------------------------------------------------

        attn = (q@k.transpose(-1,-2))*(1.0/math.sqrt(k.size(-1))) # B,nh,T,T
        attn_mask = torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        attn = attn.masked_fill(~attn_mask,float("-inf"))
        attn = F.softmax(attn,dim=-1) 
        attn = attn@v # ...,T,T@..,T,C//nh -> ...,T,C//nh

        attn=attn.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(attn) 


if __name__ =="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaltimes = 50
    cfg_mqa = Config(
        n_embd=512,
        block_size=1024,
        n_head=16,
        kv_head=4,
        rope=True
    )
    cfg_mqa_no_rope = Config(
        n_embd=512,
        block_size=1024,
        n_head=16,
        kv_head=4,
        rope=False
    )
    cfg_mha = Config(
        n_embd=512,
        block_size=1024,
        n_head=16,
        kv_head=None,
        rope=True
    )

    m1 = MQA(config=cfg_mqa, layer_idx=0).to(device)
    m2 = MQA(config=cfg_mqa_no_rope, layer_idx=0).to(device)
    m3 = MQA(config=cfg_mha, layer_idx=0).to(device)
    x = torch.randn((8,1024,512)).to(device)
    with torch.no_grad():
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        t6 = torch.cuda.Event(enable_timing=True)
        for _ in range(3): #预热
            _ = m1(x)

        t1.record()
        for _ in range(evaltimes):
            y1 = m1(x)
        t2.record()

        t2.synchronize()
        t3.record()
        for _ in range(evaltimes):
            y2 = m2(x)
        t4.record()
        t4.synchronize()
        t5.record()
        for _ in range(evaltimes):
            y3 = m3(x)
        t6.record()
        t6.synchronize()

        print("use torch.no_grad")
        print('-'*50)
        print(f"x.shape:{x.shape},y1.shape：{y1.shape}")
        print(f"MQA with rope time cost{t1.elapsed_time(t2):.2f}ms")
        print('-'*50)
        print(f"x.shape:{x.shape},y1.shape：{y2.shape}")
        print(f"MQA without rope time cost{t3.elapsed_time(t4):.2f}ms")
        print('-'*50)
        print(f"x.shape:{x.shape},y1.shape：{y3.shape}")
        print(f"MHA with rope time cost{t5.elapsed_time(t6):.2f}ms ")

# (torch) PS D:\Users\AD\vscode\code\Deeplearning\NLP> python -m modules.attention.MQA                                                                             
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MQA with rope time cost30651.03030204773ms
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MQA without rope time cost32687.052249908447ms
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MHA with rope time cost67622.21646308899ms
# 

# 实际上开销在self.qkv,计算图，torch.compile 
# - MHA qkv 投影：`3*512*512 = 786432`权重
# - MQA q+kv 投影：`512*512 + 512*256 = 393216`权重

# (torch) PS D:\Users\AD\vscode\code\Deeplearning\NLP> python -m modules.attention.MQA                                                                             
# use torch.no_grad
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MQA with rope time cost6968.07ms
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MQA without rope time cost6946.19ms
# --------------------------------------------------
# x.shape:torch.Size([32, 1024, 512]),y1.shape：torch.Size([32, 1024, 512])
# MHA with rope time cost7012.01ms 
# (torch) PS D:\Users\AD\vscode\code\Deeplearning\NLP> 