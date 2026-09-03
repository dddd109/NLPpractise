# benchmark/utils/flop.py


def matmul_flops(
    m: int,
    k: int,
    n: int,
) -> int:
    return 2 * m * k * n


def linear_flops(
    tokens: int,
    in_features: int,
    out_features: int,
) -> int:
    return matmul_flops(
        tokens,
        in_features,
        out_features,
    )


def attention_qk_flops(
    batch_size: int,
    num_q_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
) -> int:
    return (
        2
        * batch_size
        * num_q_heads
        * q_len
        * kv_len
        * head_dim
    )


def attention_av_flops(
    batch_size: int,
    num_q_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
) -> int:
    return (
        2
        * batch_size
        * num_q_heads
        * q_len
        * kv_len
        * head_dim
    )


def attention_flops(
    batch_size: int,
    num_q_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
) -> int:

    return (
        attention_qk_flops(
            batch_size,
            num_q_heads,
            q_len,
            kv_len,
            head_dim,
        )
        +
        attention_av_flops(
            batch_size,
            num_q_heads,
            q_len,
            kv_len,
            head_dim,
        )
    )