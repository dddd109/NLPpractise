"""MoE（Mixture-of-Experts）前馈层。

为什么另写一个（而不是改本目录的 ``BasicMoE``）
-----------------------------------------------
``modules/moe/MoE.py`` 里的 ``BasicMoE`` 是早期草稿，有**两处会直接出错**：

1. ``return self.experts[w.argmax(dim=-1)](x)`` —— ``self.experts`` 是
   ``nn.ModuleList``，``ModuleList.__getitem__`` 只接受 ``int``；这里传的是
   ``argmax`` 出来的 tensor，会直接抛 ``TypeError``。（``__getitem__`` 只对
   ``ParameterList`` 做过 tensor 索引的支持。）
   *注：在 *Python 语义* 上 ``ModuleList`` 还有一个坑——历史版本里对 list 的
   整型索引会被当作"第几个模块"，所以即使能索引也无法实现"按 token 取不同专家"；
   正确的写法要用 ``nonzero`` 找出 token 下标后逐个专家 ``index_add_``；*
2. 就算索引修好了，``argmax`` **不可导**，router（``self.gate``）拿不到任何梯度，
   等于从头到尾没被训练过 —— 而 MoE 的核心恰恰是让 router 学会分流。

所以这里按主流做法重写：

* **top-k 路由**：每个 token 取 router logits 最大的 k 个专家，只在选中的 k 个上
  做 softmax 并重新归一化（保证权重和为 1）。k=2 即 Mixtral 的配置。
* **负载均衡辅助损失**（Switch Transformer 式）：
  ``aux = E * Σ_i f_i · P_i``，其中 ``f_i`` 是分给专家 i 的 token 比例（按 top-1 名次
  统计）、``P_i`` 是 router 给专家 i 的平均概率。这一项会惩罚"所有 token 都挤到
  少数几个专家上"的塌缩。完美均匀时 ``aux = 1``。
* **router 抖动**（可选，Switch 的做法）：训练时给 logits 加一点高斯噪声。
  这一项不只是"随机探索"——它解决 aux loss 的一个盲区：
  ``aux`` 只衡量 ``f`` 对 ``P`` 的偏离，所以**当 router 输出均匀（P 均匀）时，
  即使 top-k 因为打分完全相同而每次都挑中同一批专家（f 退化成 one-hot），
  aux 依然是 1.0**。加噪声可以打破这种确定性打平。（实测：见本文件 __main__）
* **router z-loss**（可选，ST-MoE 提出）：抑制 router logits 的数值增长，提升稳定性。
* **共享专家**（DeepSeekMoE 式，可选）：所有 token 无条件过一遍共享专家，
  让"公共知识"不必占用路由专家的容量。

实现说明（为什么是 for 循环）
----------------------------
这里用"逐专家掩码 + ``index_add_``"的朴素写法，**优先保证可读性和可验证性**。
生产实现会用 grouped GEMM / scatter-gather 把专家并行起来；本仓库以"讲清楚结构 +
能做 benchmark"为主，所以不做过早优化（代价：专家数大时明显偏慢）。

参考：
  * Switch Transformer, arXiv:2101.03961, §2.2（aux loss）
  * Mixtral of Experts,  arXiv:2401.04088（top-2 路由）
  * DeepSeekMoE,        arXiv:2401.06066（共享专家）
  * ST-MoE,             arXiv:2202.08906（router z-loss）
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ffn.SWiGLU import SwiGLU


@dataclass
class MoEFFNConfig:
    """MoE 前馈层参数。

    Args:
        dim:            输入/输出维度（= 模型的 n_embd）
        hidden_dim:     单个专家内部的隐层维度；None 表示 4 * dim（GPT-2 风格）
        num_experts:    路由专家个数 E
        top_k:          每个 token 激活几个专家
        shared_experts: 共享专家个数（0 = 不启用）
        aux_loss_coef:  负载均衡损失的系数
        z_loss_coef:    router z-loss 的系数（0 = 关闭）
        router_jitter:  训练时给 router logits 加的高斯噪声标准差（0 = 关闭）。
                        Switch 里用 0.01 量级；作用是打破 top-k 的确定性打平，
                        详见模块 docstring 里关于 aux loss 盲区的说明。
    """

    dim: int
    hidden_dim: Optional[int] = None
    num_experts: int = 8
    top_k: int = 2
    shared_experts: int = 0
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.0
    router_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.dim
        assert self.num_experts >= 1, "num_experts 至少为 1"
        assert 1 <= self.top_k <= self.num_experts, "top_k 必须落在 [1, num_experts]"
        assert self.shared_experts >= 0


class MoEFFN(nn.Module):
    r"""top-k 路由的 MoE 前馈层。

    Shape:
        - Input:  ``[..., C]``（任意 batch 形状，最后一维是特征维）
        - Output: ``[..., C]``

    额外输出（训练循环里用）::

        aux = model.ffn.last_aux_loss      # 加进总 loss：loss = ce + aux
        print(model.ffn.last_usage)        # 各专家被选中的比例（监控负载均衡）

    注意：导出与 ``BasicMoE`` 的对比 —— ``BasicMoE`` 是"每 token 只走 1 个专家、
    且 router 不训练"；这里是标准要能训练的实现。
    """

    def __init__(self, config: MoEFFNConfig):
        super().__init__()
        self.config = config

        # router：把 token 的隐状态映射成 E 个专家分数（不带 bias）
        self.router = nn.Linear(config.dim, config.num_experts, bias=False)

        # 路由专家：本仓库统一用 SwiGLU 作为专家内部结构
        self.experts = nn.ModuleList(
            SwiGLU(config.dim, config.hidden_dim) for _ in range(config.num_experts)
        )

        # 共享专家（DeepSeekMoE 风格）：所有 token 都必须过
        if config.shared_experts > 0:
            self.shared_experts: Optional[nn.ModuleList] = nn.ModuleList(
                SwiGLU(config.dim, config.hidden_dim)
                for _ in range(config.shared_experts)
            )
        else:
            self.shared_experts = None

        # 供训练循环读取的统计量（不是 buffer，不参与 state_dict，避免污染 checkpoint）
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        self.last_usage: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lead_shape = x.shape[:-1]
        x_flat = x.reshape(-1, cfg.dim)          # [N, C]
        n_tokens = x_flat.size(0)

        # ---- 1) router 打分 --------------------------------------------------
        logits = self.router(x_flat)             # [N, E]
        if self.training and cfg.router_jitter > 0:
            # 只在训练时加噪声：打破 top-k 的确定性打平（否则 logits 相同时
            # 每个 token 都会挑中同一批专家，f 退化成 one-hot，而 aux 仍可能是 1.0）
            logits = logits + torch.randn_like(logits) * cfg.router_jitter
        probs = F.softmax(logits, dim=-1)        # [N, E]

        # ---- 2) top-k 选择 + 归一化 -----------------------------------------
        topk_probs, topk_idx = probs.topk(cfg.top_k, dim=-1)      # [N, k], [N, k]
        # 只在被选中的 k 个专家上重新归一化，保证"每个 token 的权重和为 1"
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ---- 3) 逐专家聚合 --------------------------------------------------
        out = torch.zeros_like(x_flat)
        for expert_id in range(cfg.num_experts):
            hit = topk_idx == expert_id              # [N, k] bool
            if not bool(hit.any()):
                continue
            token_ids, slot_ids = hit.nonzero(as_tuple=True)
            weights = topk_probs[token_ids, slot_ids].unsqueeze(-1)   # [M, 1]
            expert_out = self.experts[expert_id](x_flat[token_ids])   # [M, C]
            # 同一个 token 可能在 top-k 里多次命中同一专家？不会（topk 互异），
            # 但不同 token 会命中同一专家，所以用 index_add_ 累加
            out.index_add_(0, token_ids, weights * expert_out)

        # ---- 4) 共享专家（无条件参与）--------------------------------------
        if self.shared_experts is not None:
            shared = sum(expert(x_flat) for expert in self.shared_experts)
            out = out + shared

        # ---- 5) 负载均衡 / z-loss（只影响训练，不改变前向输出）-------------
        self._update_router_stats(probs, logits, topk_idx, n_tokens)

        return out.view(*lead_shape, cfg.dim)

    # ------------------------------------------------------------------
    def _update_router_stats(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        n_tokens: int,
    ) -> None:
        """算出 aux loss 和专家使用率，存到 ``last_aux_loss`` / ``last_usage``。"""
        cfg = self.config
        n_experts = cfg.num_experts

        # f_i：分给专家 i 的 token 比例（按 top-1 名次统计，Switch/Mixtral 的做法）
        one_hot = F.one_hot(topk_idx[:, 0], num_classes=n_experts).to(probs.dtype)
        f = one_hot.mean(dim=0)                       # [E]
        # P_i：router 给专家 i 的平均概率
        p_mean = probs.mean(dim=0)                    # [E]

        aux = n_experts * torch.sum(f * p_mean)       # 完美均匀时 = 1.0
        loss = cfg.aux_loss_coef * aux

        if cfg.z_loss_coef > 0:
            # router z-loss：压制 logits 的数值规模（logsumexp 的平方均值）
            z = torch.logsumexp(logits, dim=-1).pow(2).mean()
            loss = loss + cfg.z_loss_coef * z

        self.last_aux_loss = loss
        self.last_usage = f.detach()

    # ------------------------------------------------------------------
    # 便于报告 / benchmark 的账目统计
    # ------------------------------------------------------------------
    def expert_param_count(self) -> int:
        """所有路由专家的参数量之和。"""
        return sum(p.numel() for p in self.experts.parameters())

    def total_param_count(self) -> int:
        """含 router 与共享专家在内的全部参数。"""
        return sum(p.numel() for p in self.parameters())

    def active_param_count(self) -> int:
        """**每个 token 实际激活**的参数量（MoE 的关键卖点）。"""
        active = self.router.weight.numel()                      # router 总是要算
        active += self.config.top_k * sum(
            p.numel() for p in self.experts[0].parameters()
        )
        if self.shared_experts is not None:
            active += sum(p.numel() for p in self.shared_experts.parameters())
        return active

    def flops_per_token(self) -> float:
        """每个 token 的乘加次数（近似：一次 Linear 记为 2 * in * out）。"""
        cfg = self.config
        per_expert = 3 * 2 * cfg.dim * cfg.hidden_dim            # SwiGLU: w1/w3 升维 + w2 降维
        flops = 2 * cfg.dim * cfg.num_experts                    # router
        flops += cfg.top_k * per_expert
        flops += cfg.shared_experts * per_expert
        return float(flops)

    def describe(self) -> str:
        cfg = self.config
        return (
            f"MoEFFN(E={cfg.num_experts}, top_k={cfg.top_k}, "
            f"shared={cfg.shared_experts}, hidden={cfg.hidden_dim}) | "
            f"总参数={self.total_param_count():,} "
            f"每 token 激活={self.active_param_count():,} "
            f"激活率={self.active_param_count() / self.total_param_count():.1%}"
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 64, 64)

    print("--- 正常 router ---")
    moe = MoEFFN(MoEFFNConfig(dim=64, hidden_dim=128, num_experts=8, top_k=2, shared_experts=1))
    y = moe(x)
    print("  out:", tuple(y.shape))
    print(" ", moe.describe())
    print("  aux loss:", round(float(moe.last_aux_loss.detach()), 5))
    print("  专家使用率:", [round(float(v), 3) for v in moe.last_usage])
    print("  FLOPs/token:", moe.flops_per_token())

    # aux loss 的盲区：logits 打平时 aux 仍是 1.0，但负载其实退化
    print("\n--- aux loss 盲区演示（把 router 权重清零，让所有 logits 相同）---")
    for jitter in (0.0, 0.05):
        m = MoEFFN(
            MoEFFNConfig(
                dim=64, hidden_dim=128, num_experts=8, top_k=2,
                aux_loss_coef=1.0, router_jitter=jitter,
            )
        )
        with torch.no_grad():
            m.router.weight.zero_()
        m.train()
        m(x)
        used = sum(1 for v in m.last_usage if float(v) > 0)
        print(f"  router_jitter={jitter:<5} aux={float(m.last_aux_loss.detach()):.4f} "
              f"使用率={[round(float(v), 3) for v in m.last_usage]} 用到的专家={used}/8")
    print("  -> jitter=0 时 aux=1.0 看着「完美均衡」，其实只有一个专家在干活；")
    print("     加噪声后负载才真正分散开。所以监控一定要看 last_usage，不能只看 aux。")
