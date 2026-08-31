import torch
import torch.nn as nn


class RoPE(nn.Module):
    r"""
    RoPE 旋转位置编码
    核心修复：
    1. 移除易引发设备不匹配的 cos/sin 缓存，改为实时计算（RoPE 计算量极小，性能可忽略）
    2. 新增 position offset 支持，原生兼容 KV Cache 自回归推理
    3. 自动对齐输入张量 dtype，适配 fp16/bf16 混合精度训练/推理
    4. 移除序列长度上限限制，支持动态变长

    Args:
        dim: per-head dimension 单头注意力的维度
        theta: 频率基数，默认 10000.0（与 Llama 原版一致）
    Shape:
        forward(x): x [B, H, T, C] → [B, H, T, C]
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # 频率基数注册为 buffer，随模型 .to(device) 自动迁移设备
        theta_arr = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("theta_arr", theta_arr)  # shape: [dim // 2]

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32, offset: int = 0):
        """
        生成对应长度的 cos、sin 相位矩阵
        Args:
            seq_len: 当前序列长度
            device: 目标计算设备
            dtype: 输出数据类型，建议与 Q/K 张量保持一致
            offset: 位置偏移（KV Cache 推理时使用，训练阶段保持 0）
        Returns:
            cos, sin: shape [seq_len, dim // 2]
        """
        # 生成位置索引：[offset, offset+1, ..., offset+seq_len-1]
        m = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        # 外积得到每个位置、每个维度对的相位
        freqs = torch.outer(m, self.theta_arr.to(device))
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos, sin

    @staticmethod
    def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Q、K 同时应用 RoPE
        Args:
            q: 查询张量，shape [B, H, T, C]
            k: 键张量，shape [B, H, T, C]
            cos: 余弦相位，shape [T, C // 2]
            sin: 正弦相位，shape [T, C // 2]
        Returns:
            q_rot, k_rot: 旋转后的 Q、K，形状与输入一致
        """
        B, H, T, C = q.shape
        # 维度广播：[T, C//2] → [1, 1, T, C//2]，适配 [B, H, T, C] 形状
        cos = cos[None, None, :T, :]
        sin = sin[None, None, :T, :]

        # 拆分奇偶维度（对应复数的实部、虚部）
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2] # B,H,T,C//2
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        # 旋转矩阵运算
        q_out_even = q_even * cos - q_odd * sin
        q_out_odd = q_even * sin + q_odd * cos
        k_out_even = k_even * cos - k_odd * sin
        k_out_odd = k_even * sin + k_odd * cos

        # 交错合并回原始维度顺序
        q_rot = torch.stack([q_out_even, q_out_odd], dim=-1).flatten(-2)
        k_rot = torch.stack([k_out_even, k_out_odd], dim=-1).flatten(-2)
        return q_rot, k_rot

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        单个张量旋转接口
        生产环境请优先使用 apply_rotary_emb 同时处理 Q、K
        Args:
            x: 输入张量，shape [B, H, T, C]
            offset: 位置偏移
        Returns:
            x_rot: 旋转后的张量，形状与输入一致
        """
        assert x.size(-1) % 2 == 0, "RoPE 要求最后一维维度为偶数"
        B, H, T, C = x.shape
        cos, sin = self.get_cos_sin(seq_len=T, device=x.device, dtype=x.dtype, offset=offset)

        cos = cos[None, None, :T, :]
        sin = sin[None, None, :T, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out


# ---------------- 测试与使用示例 ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 8  # 单头维度
    rope = RoPE(dim=dim).to(device)

    B, H, T = 2, 4, 6  # batch, 注意力头数, 序列长度
    q = torch.randn(B, H, T, dim, device=device)
    k = torch.randn(B, H, T, dim, device=device)

    # 用法1：单张量 forward 接口（测试用）
    q1 = rope(q)
    # 用法2：生产标准用法：获取 cos/sin 后同时旋转 Q、K
    cos, sin = rope.get_cos_sin(T, device, q.dtype)
    q2, k2 = RoPE.apply_rotary_emb(q, k, cos, sin)

    print(f"单张量接口与双张量接口误差：{(q1 - q2).abs().max():.8f}")
    print(f"q_rot 形状: {q2.shape}, k_rot 形状: {k2.shape}")

    # 测试 KV Cache 场景（offset 位置偏移）
    # 模拟：已有 4 个 token 的 KV 缓存，新进来 2 个 token，位置从 4 开始
    offset = 4
    new_seq_len = 2
    cos_offset, sin_offset = rope.get_cos_sin(new_seq_len, device, q.dtype, offset=offset)
    print(f"\nKV Cache 偏移 cos 形状: {cos_offset.shape}，起始位置: {offset}")

    # 验证偏移正确性：offset=4 的第0个位置，等价于 offset=0 的第4个位置
    cos_full, _ = rope.get_cos_sin(offset + new_seq_len, device, q.dtype, offset=0)
    print(f"偏移逻辑验证误差：{(cos_offset - cos_full[offset:offset+new_seq_len]).abs().max():.8f}")
