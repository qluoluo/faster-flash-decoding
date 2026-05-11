from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the fast decode kernel.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--value-dim", type=int, default=128)
    p.add_argument("--bs", type=int, default=128, help="Quantized block size.")
    p.add_argument("--sbs", type=int, default=None, help="Sub-block size. Defaults to --bs.")
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--num-splits", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Activation dtype for Q/K/V.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp8-residual", action="store_true")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def quantize_k_2bit_symmetric(
    k: torch.Tensor,
    *,
    fp8_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pack K as 2-bit symmetric values plus optional FP8 residual."""
    bsz, seq_len, num_kv_heads, head_dim = k.shape
    qmax = 3
    qzero = qmax / 2.0
    vals_per_byte = 4
    k_packed = (head_dim + vals_per_byte - 1) // vals_per_byte

    scale = (k.abs().amax(dim=1) / qzero).clamp_min(1e-6).contiguous()
    k_q = torch.round(k / scale[:, None, :, :] + qzero).clamp(0, qmax).to(torch.uint8)

    k_dequant = (k_q.to(torch.float32) - qzero) * scale[:, None, :, :].to(torch.float32)
    residual = None
    if fp8_residual:
        residual = (k.to(torch.float32) - k_dequant).to(torch.float8_e4m3fn).contiguous()

    pad = k_packed * vals_per_byte - head_dim
    if pad:
        padding = torch.zeros(
            (bsz, seq_len, num_kv_heads, pad),
            device=k.device,
            dtype=k_q.dtype,
        )
        k_q = torch.cat((k_q, padding), dim=-1)

    k_q = k_q.view(bsz, seq_len, num_kv_heads, k_packed, vals_per_byte)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()

    return k_q_packed, scale, residual


def cuda_benchmark(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads.")

    from ffd_core import attn_forward_decode_quantized

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = _dtype(args.dtype)
    q = torch.randn(
        args.batch,
        1,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch,
        args.seq_len,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        args.batch,
        args.seq_len,
        args.num_kv_heads,
        args.value_dim,
        device=device,
        dtype=dtype,
    )
    k_q, k_scale, k_residual = quantize_k_2bit_symmetric(
        k,
        fp8_residual=not args.no_fp8_residual,
    )

    def run_kernel():
        return attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_residual=k_residual,
            k_bits=2,
            scale=1.0 / math.sqrt(args.head_dim),
            BS=args.bs,
            SBS=args.sbs,
            delta=args.delta,
            use_fp8_residual=not args.no_fp8_residual,
            num_splits=args.num_splits,
        )

    out = run_kernel()
    torch.cuda.synchronize(device)
    elapsed_ms = cuda_benchmark(
        run_kernel,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    print(f"kernel: {attn_forward_decode_quantized.__module__}")
    print(
        "shape: "
        f"B={args.batch}, T={args.seq_len}, HQ={args.num_heads}, "
        f"HKV={args.num_kv_heads}, D={args.head_dim}, V={args.value_dim}"
    )
    print(f"output: shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"latency: {elapsed_ms:.4f} ms/iter ({args.iters} iters)")


if __name__ == "__main__":
    main()
