from __future__ import annotations

import math

import torch

RCP_LN2 = 1.4426950408889634


def normalize_selector_type(selector_type: str | None) -> str:
    normalized = (selector_type or "top_delta").strip().lower().replace("-", "_")
    if normalized in ("delta", "topdelta"):
        normalized = "top_delta"
    if normalized in ("k", "topk"):
        normalized = "top_k"
    if normalized not in ("top_delta", "top_k"):
        raise ValueError(f"Unsupported selector_type={selector_type!r}")
    return normalized


def resolve_selector_value(
    selector_type: str,
    selector_value: float | int | None,
    *,
    delta: float,
) -> float | int:
    if selector_value is None:
        return float(delta) if selector_type == "top_delta" else 1
    return float(selector_value) if selector_type == "top_delta" else int(selector_value)


@torch.no_grad()
def build_q2_topk_threshold(
    *,
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    block_size: int,
    k_bits: int = 2,
) -> torch.Tensor | None:
    if k_scale is None or k_scale.ndim != 4:
        raise ValueError(
            "Q2 top-k threshold expects per-block k_scale with shape [B, NTB, HKV, K]."
        )

    batch_size, _, num_heads, head_dim = q.shape
    _, token_capacity, num_kv_heads, packed_dim = k_q.shape
    _, num_blocks, scale_hkv, scale_dim = k_scale.shape
    if num_blocks <= 0:
        return None
    if scale_hkv != num_kv_heads or scale_dim != head_dim:
        raise ValueError(
            f"Incompatible Q2 scale layout: k_scale={tuple(k_scale.shape)}, "
            f"expected HKV={num_kv_heads}, K={head_dim}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )

    vals_per_byte = 8 // k_bits
    expected_packed_dim = (head_dim + vals_per_byte - 1) // vals_per_byte
    if packed_dim != expected_packed_dim:
        raise ValueError(
            f"k_q packed dim mismatch: got {packed_dim}, expected {expected_packed_dim}"
        )

    token_count = min(token_capacity, num_blocks * block_size)
    if token_count <= 0:
        return None

    qmax = (1 << k_bits) - 1
    qzero = qmax / 2.0
    groups = num_heads // num_kv_heads

    shifts = torch.arange(vals_per_byte, device=q.device, dtype=torch.int32) * k_bits
    k_packed = k_q[:, :token_count].to(torch.int32)
    unpacked = ((k_packed.unsqueeze(-1) >> shifts) & qmax).reshape(
        batch_size,
        token_count,
        num_kv_heads,
        -1,
    )[..., :head_dim]
    unpacked = unpacked.to(torch.float32)

    scale_per_token = (
        k_scale.to(torch.float32)
        .unsqueeze(2)
        .expand(batch_size, num_blocks, block_size, num_kv_heads, head_dim)
        .reshape(batch_size, token_count, num_kv_heads, head_dim)
    )
    k_dequant = (unpacked - qzero) * scale_per_token

    q_grouped = q.squeeze(1).to(torch.float32).view(
        batch_size, num_kv_heads, groups, head_dim
    )
    k_blocks = k_dequant.view(
        batch_size, num_blocks, block_size, num_kv_heads, head_dim
    ).permute(0, 3, 1, 2, 4)

    scores = torch.einsum("bhgk,bhnsk->bhgns", q_grouped, k_blocks)
    scores = scores * ((1.0 / math.sqrt(head_dim)) * RCP_LN2)
    block_scores = scores.amax(dim=-1)
    group_scores = block_scores.amax(dim=2)
    return group_scores


def topk_group_scores_to_threshold(group_threshold: torch.Tensor, num_heads: int) -> torch.Tensor:
    if group_threshold.ndim != 2:
        raise ValueError(
            f"group_threshold must have shape [B, HKV], got {tuple(group_threshold.shape)}"
        )
    _, num_kv_heads = group_threshold.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    groups = num_heads // num_kv_heads
    threshold = group_threshold.repeat_interleave(groups, dim=1)
    return threshold.contiguous()


def build_q2_selector_threshold(
    *,
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    block_size: int,
    selector_type: str,
    selector_value: float | int,
    k_bits: int = 2,
) -> torch.Tensor | None:
    selector_type = normalize_selector_type(selector_type)
    if selector_type != "top_k":
        return None

    group_scores = build_q2_topk_threshold(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        block_size=block_size,
        k_bits=k_bits,
    )
    if group_scores is None:
        return None

    num_blocks = group_scores.shape[-1]
    top_k = max(1, min(int(selector_value), num_blocks))
    kth_scores = torch.topk(group_scores, k=top_k, dim=-1, sorted=True).values[..., -1]
    return topk_group_scores_to_threshold(kth_scores, q.shape[2])
