import torch

from ...modules.attention.MQA import MQA, MQAConfig

from benchmark.case import BenchmarkCase
from benchmark.result import WorkloadInfo
from benchmark.runner import BenchmarkRunner

from benchmark.utils.flop import (
    linear_flops,
    attention_flops,
)


# ============================================================
# 1. Basic configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

B = 1
T = 1024
C = 512
H = 16
H_KV = 4

D = C // H


# ============================================================
# 2. Build model
# ============================================================

config = MQAConfig(
    n_embd=C,
    block_size=T,
    n_head=H,
    kv_head=H_KV,
    rope=False,
)

model = MQA(config=config).to(device)
model.eval()


# ============================================================
# 3. Input
# ============================================================

x = torch.randn(
    (B, T, C),
    device=device,
)


# ============================================================
# 4. Workload information
# ============================================================

workload = WorkloadInfo(
    name="MQA_B1_T1024",
    batch_size=B,
    input_shapes={
        "x": tuple(x.shape),
    },
    output_shapes=None,
    dtype=str(x.dtype),
    device=str(x.device),
    metadata={
        "q_len": T,
        "kv_len": T,
        "hidden_size": C,
        "num_q_heads": H,
        "num_kv_heads": H_KV,
        "head_dim": D,
        "mode": "prefill",
        "num_tokens": B * T,
    },
)


# ============================================================
# 5. Analytical FLOPs
# ============================================================

def calculate_flops() -> int:
    # Q projection
    q_flops = linear_flops(
        B * T,
        C,
        C,
    )

    # K/V projection
    kv_flops = linear_flops(
        B * T,
        C,
        2 * H_KV * D,
    )

    # QK^T + AV
    attention = attention_flops(
        batch_size=B,
        num_q_heads=H,
        q_len=T,
        kv_len=T,
        head_dim=D,
    )

    # Output projection
    output_flops = linear_flops(
        B * T,
        C,
        C,
    )

    return (
        q_flops
        + kv_flops
        + attention
        + output_flops
    )


# ============================================================
# 6. Estimated memory traffic
#
# NOTE:
# This is analytical tensor traffic, NOT real HBM traffic.
# It is only used for an estimated arithmetic intensity.
# ============================================================

def calculate_bytes() -> int:
    bytes_per_element = x.element_size()

    def tensor_bytes(numel: int) -> int:
        return numel * bytes_per_element

    total_bytes = 0

    # --------------------------------------------------------
    # Q projection
    #
    # X + Wq + Q
    # --------------------------------------------------------

    total_bytes += tensor_bytes(B * T * C)
    total_bytes += tensor_bytes(C * C)
    total_bytes += tensor_bytes(B * T * C)

    # --------------------------------------------------------
    # K/V projection
    #
    # X + Wkv + KV
    # --------------------------------------------------------

    total_bytes += tensor_bytes(B * T * C)
    total_bytes += tensor_bytes(C * (2 * H_KV * D))
    total_bytes += tensor_bytes(B * T * 2 * H_KV * D)

    # --------------------------------------------------------
    # Attention QK^T
    #
    # Q + K + score
    #
    # NOTE:
    # This assumes the tensors are materialized.
    # It does not model cache reuse or fusion.
    # --------------------------------------------------------

    q_bytes = tensor_bytes(
        B * H * T * D
    )

    k_bytes = tensor_bytes(
        B * H * T * D
    )

    score_bytes = tensor_bytes(
        B * H * T * T
    )

    total_bytes += q_bytes
    total_bytes += k_bytes
    total_bytes += score_bytes

    # --------------------------------------------------------
    # Attention AV
    #
    # score + V + output
    # --------------------------------------------------------

    v_bytes = tensor_bytes(
        B * H * T * D
    )

    total_bytes += score_bytes
    total_bytes += v_bytes
    total_bytes += q_bytes

    # --------------------------------------------------------
    # Output projection
    #
    # attention output + Wo + final output
    # --------------------------------------------------------

    total_bytes += tensor_bytes(B * T * C)
    total_bytes += tensor_bytes(C * C)
    total_bytes += tensor_bytes(B * T * C)

    return total_bytes


# ============================================================
# 7. Throughput
# ============================================================

def calculate_throughput(timing, workload) -> float:
    num_tokens = workload.metadata["num_tokens"]

    return num_tokens / (
        timing.mean_ms / 1000.0
    )


# ============================================================
# 8. Benchmark case
# ============================================================

case = BenchmarkCase(
    name="MQA_B1_T1024",

    fn=lambda: model(x),

    workload=workload,

    flop_fn=calculate_flops,

    byte_fn=calculate_bytes,

    throughput_fn=calculate_throughput,
)


# ============================================================
# 9. Run benchmark
# ============================================================

runner = BenchmarkRunner(
    warmup=10,
    iters=100,
)


result = runner.run(
    case=case,
    enable_profile=True,
)


# ============================================================
# 10. Display result
# ============================================================

print("\n" + "=" * 70)
print("Benchmark Result")
print("=" * 70)

print(f"Name: {result.name}")

print("\n[Workload]")
print(f"  Input shape : {result.workload.input_shapes}")
print(f"  dtype       : {result.workload.dtype}")
print(f"  device      : {result.workload.device}")

print("\n[Timing]")
print(f"  mean        : {result.timing.mean_ms:.4f} ms")
print(f"  median      : {result.timing.median_ms:.4f} ms")
print(f"  p50         : {result.timing.p50_ms:.4f} ms")
print(f"  p90         : {result.timing.p90_ms:.4f} ms")
print(f"  p99         : {result.timing.p99_ms:.4f} ms")
print(f"  std         : {result.timing.std_ms:.4f} ms")

print("\n[Throughput]")
print(f"  tokens/s    : {result.throughput:.2f}")

print("\n[Memory]")
print(f"  allocated   : {result.memory.allocated_mb:.2f} MB")
print(f"  reserved    : {result.memory.reserved_mb:.2f} MB")
print(f"  peak alloc  : {result.memory.peak_allocated_mb:.2f} MB")
print(f"  peak reserv  : {result.memory.peak_reserved_mb:.2f} MB")
print(f"  KV cache    : {result.memory.kv_cache_mb}")

print("\n[Compute]")
print(
    f"  theoretical FLOPs : "
    f"{result.compute.theoretical_flops / 1e9:.4f} GFLOPs"
)

print(
    f"  profiler FLOPs    : "
    f"{result.compute.profiler_flops / 1e9:.4f} GFLOPs"
    if result.compute.profiler_flops is not None
    else "  profiler FLOPs    : N/A"
)

print(
    f"  achieved TFLOPS   : "
    f"{result.compute.achieved_tflops:.4f}"
    if result.compute.achieved_tflops is not None
    else "  achieved TFLOPS   : N/A"
)

print(
    f"  arithmetic intensity : "
    f"{result.compute.arithmetic_intensity:.4f} FLOP/Byte"
    if result.compute.arithmetic_intensity is not None
    else "  arithmetic intensity : N/A"
)

if result.compute.estimated_memory_bytes is not None:
    print(
        f"  estimated bytes   : "
        f"{result.compute.estimated_memory_bytes / 1e9:.4f} GB"
    )


# ============================================================
# 11. Profiler result
# ============================================================

if result.profile is not None:

    print("\n" + "=" * 70)
    print("Profiler Result")
    print("=" * 70)

    print(
        f"CUDA self time : "
        f"{result.profile.total_cuda_time_us / 1000:.4f} ms"
    )

    print(
        f"CPU self time  : "
        f"{result.profile.total_cpu_time_us / 1000:.4f} ms"
    )

    print("\n[Operator Table]")
    print(result.profile.operator_table)

    if result.profile.trace_path is not None:
        print(
            f"\nChrome trace: "
            f"{result.profile.trace_path}"
        )