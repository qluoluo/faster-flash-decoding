"""Kernel correctness test: compare FFD output against FlashAttention (SDPA)."""

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
    p = argparse.ArgumentParser(description="Test FFD kernel correctness.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for output comparison")
    p.add_argument("--rtol", type=float, default=1e-1, help="Relative tolerance for output comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp8-residual", action="store_true")
    return p.parse_args()


def quantize_k_2bit_symmetric(
    k: torch.Tensor,
    *,
    fp8_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def flash_attn_reference(q, k, v, scale, n_rep):
    """Reference FlashAttention via SDPA."""
    k_expanded = repeat_kv(k, n_rep)
    v_expanded = repeat_kv(v, n_rep)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k_expanded, v_expanded, attn_mask=None, dropout_p=0.0, is_causal=True,
    )
    return out


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")
    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("--num-heads must be divisible by --num-kv-heads.")

    from ffd_core import attn_forward_decode_quantized

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float16
    n_rep = args.num_heads // args.num_kv_heads

    # Generate random K, V cache (long context) and a single query (decode step)
    k_cache = torch.randn(
        args.batch, args.seq_len, args.num_kv_heads, args.head_dim,
        device=device, dtype=dtype,
    )
    v_cache = torch.randn(
        args.batch, args.seq_len, args.num_kv_heads, args.head_dim,
        device=device, dtype=dtype,
    )
    q = torch.randn(
        args.batch, 1, args.num_heads, args.head_dim,
        device=device, dtype=dtype,
    )

    # Quantize K cache
    k_q, k_scale, k_residual = quantize_k_2bit_symmetric(
        k_cache, fp8_residual=not args.no_fp8_residual,
    )

    # Run FFD kernel
    scale = 1.0 / math.sqrt(args.head_dim)
    out_ffd = attn_forward_decode_quantized(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        v=v_cache,
        k_residual=k_residual,
        k_bits=2,
        scale=scale,
        BS=args.bs,
        delta=args.delta,
        use_fp8_residual=not args.no_fp8_residual,
    )
    # FFD kernel returns [B, HQ, V]; SDPA expects [B, HQ, 1, V]
    out_ffd_4d = out_ffd.unsqueeze(2)  # [B, HQ, 1, V]

    # Run FlashAttention reference
    q_ref = q.transpose(1, 2)  # [B, HQ, 1, D]
    k_ref = k_cache.transpose(1, 2)  # [B, HKV, T, D]
    v_ref = v_cache.transpose(1, 2)  # [B, HKV, T, D]
    out_ref = flash_attn_reference(q_ref, k_ref, v_ref, scale, n_rep)
    # out_ref is [B, HQ, 1, D]

    # Compare
    diff = (out_ffd_4d - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ffd_4d.flatten(), out_ref.flatten(), dim=0
    ).item()

    passed = torch.allclose(out_ffd_4d, out_ref, atol=args.atol, rtol=args.rtol)

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
