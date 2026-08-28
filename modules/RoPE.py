import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    r"""
    旋转位置编码 RoPE (Rotary Position Embedding)，实现 default 版本
    数学原理：二维向量旋转矩阵

    .. math::
        \begin{bmatrix}x' \\ y'\end{bmatrix}
        =
        \begin{bmatrix}
        \cos\theta_m & -\sin\theta_m \\
        \sin\theta_m & \cos\theta_m
        \end{bmatrix}
        \begin{bmatrix}x \\ y\end{bmatrix}
    
    Args:
        dim(int) 隐藏特征维度,下同注释中的C
        max_sqlen(int) 支持的最大序列长度
    
    利用复数等价实现：(x + iy) * e^{i\cdot m\theta}，复数相乘等价旋转，计算更快
    Shape:
        - input:  :math:`[B, H, T, C]`  B=batch,H=head,T=seq_len,C=per_head_dim
        - output: :math:`[B, H, T, C]` 和输入同shape
    Note:
        ✅ 只给 Q / K 使用，**不要给V使用**
    """
    def __init__(self,dim,max_sqlen,theta =10000.0):
        super().__init__()
        self.dim=dim
        self.max_sqlen=max_sqlen
        self.theta = theta

        theta_arr = 1.0/(self.theta **(torch.arange(0,self.dim,2).float()/self.dim)) # [dim//2]
        m = torch.arange(self.max_sqlen) # 原序列索引 [max_sqlen]
        freqs = torch.outer(m,theta_arr)# 外积 m*theta_i-->旋转角 θ [max_sqlen,dim//2] 
        freqs = torch.polar(torch.ones_like(freqs),freqs) # cosθ + isinθ [max_sqlen,dim//2]

        self.register_buffer("freqs",freqs)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        assert x.size(-1)%2==0 ,"RoPE 最后一维度必须为偶数"#B,H,T,C
        B,H,T,C = x.size()
        freqs = self.freqs[None,None,:T,:] # 1,1,T,dim//2
        x = x.reshape(B,H,T,C//2,2) # B,H,T,C//2,2
        x = torch.view_as_complex(x.contiguous()) # B,H,T,C//2
        x = freqs *x # B,H,T,C//2
        x = torch.view_as_real(x).reshape(B,H,T,C) # B,H,T,C
        return x
# 当前版本的RoPE没有自适应 max_sqlen

if __name__ =="__main__":
    dim = 8
    max_sqlen = 8
    rope = RoPE(dim,max_sqlen)
    x = torch.randn(3,2,4,dim)
    y = rope(x)
    print(f"input:{x.shape} output:{y.shape}")
    print(f"x:{x}")
    print("-"*50)
    print(f"y:{y}")