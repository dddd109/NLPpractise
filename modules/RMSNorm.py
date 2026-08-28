import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    r"""
    RMSNorm 均方根归一化 (Root‑Mean‑Square Normalization)

    LayerNorm 的简化变体，移除均值中心化步骤，无 beta 偏移参数，
    仅保留可学习缩放参数 gamma，计算开销更低，广泛用于 LLaMA / Mistral 等大模型。

    .. math::
        y = \gamma \cdot x \cdot \mathrm{rsqrt}\big(\mathrm{mean}(x^2) + \epsilon\big)

    Args:
        dim (int): 隐藏特征维度，输入张量最后一维的大小
        eps (float, optional): 数值稳定项，防止分母为 0。默认: ``1e-5``

    Shape:
        - Input: :math:`[..., C]`，任意形状张量，最后一维为特征维度 ``C``
        - Output: :math:`[..., C]`，输出与输入张量形状完全相同

    Example::
        >>> dim = 128
        >>> norm = RMSNorm(dim)
        >>> x = torch.randn(2, 3, dim)
        >>> y = norm(x)
        >>> print(y.shape)
        torch.Size([2, 3, 128])
    """
    def __init__(self,dim,eps = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        C = x.size(-1)
        rms = x.pow(2).mean(-1,keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        return self.gamma*x * rms

if __name__ == "__main__":
    dim = 128
    norm = RMSNorm(dim)
    x= torch.randn(2,3,dim)
    y = norm(x)
    print(f"input {x.shape} output:{y.shape},output y mean:{y.sum(dim=-1)}")