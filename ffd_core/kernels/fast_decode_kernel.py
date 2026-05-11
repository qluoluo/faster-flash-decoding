# Persistent Accumulation optimization:
# Core idea:
# 1. Eliminate Stage 2 (Aggregation) and Compact Buffer writes entirely.
# 2. Stage 1 becomes "Persistent" mode:
#    - Each Thread Block handles a range of the T dimension (Split-K).
#    - Accumulators (m, l, acc) are kept in registers.
#    - Iterate over blocks in the assigned range, compute scores, compare with global threshold.
#    - If a block is kept, online-update the register accumulators.
#    - Finally output only partial results (Split-K) to global memory.
# 3. Add a tiny Stage 3 (Reduction) kernel to merge Split-K results.
#
# Expected benefit: reduced DRAM bandwidth (no intermediate writes), fewer kernel launches.

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym_persistent_bk128"


# Reuse the existing threshold kernel — it is already fast and needs no changes
@triton.jit
def attn_compute_threshold_qbits(
    q,
    k_q,
    k_scale,
    th_out,
    scale,
    T,
    NTB,
    delta,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr = 128,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    BK: tl.constexpr = 128,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    tb0 = 0
    if USE_PERBLOCK_SCALE:
        scale_base0 = pid_b * (NTB * HKV * K) + tb0 * (HKV * K) + pid_hkv * K
    else:
        scale_base0 = pid_b * (HKV * K) + pid_hkv * K

    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    base_tok0_q = (
        pid_b * (T * HKV * K_PACKED) + offs_t0 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    )
    tl.multiple_of(base_tok0_q, K_PACKED)

    tb1 = NTB - 1
    if USE_PERBLOCK_SCALE:
        scale_base1 = pid_b * (NTB * HKV * K) + tb1 * (HKV * K) + pid_hkv * K
    else:
        scale_base1 = pid_b * (HKV * K) + pid_hkv * K

    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    base_tok1_q = (
        pid_b * (T * HKV * K_PACKED) + offs_t1 * (HKV * K_PACKED) + (pid_hkv * K_PACKED)
    )
    tl.multiple_of(base_tok1_q, K_PACKED)

    b_s0 = tl.zeros([BM_DOT, T_BS], tl.float32)
    b_s1 = tl.zeros([BM_DOT, T_BS], tl.float32)
    q_zero_sum0 = tl.zeros([BM_DOT], tl.float32)
    q_zero_sum1 = tl.zeros([BM_DOT], tl.float32)

    offs_k_base = tl.arange(0, BK)
    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        pack_idx = offs_k // VALS_PER_BYTE
        pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float16
        )

        scale_sub0 = tl.load(k_scale + scale_base0 + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        q_scaled_sub0 = q_sub * scale_sub0[None, :].to(tl.float16)
        q_zero_sum0 += tl.sum(q_scaled_sub0.to(tl.float32), axis=1)

        kq_ptrs0 = k_q + base_tok0_q[None, :] + pack_idx[:, None]
        kq_tile0 = tl.load(
            kq_ptrs0, mask=k_mask[:, None] & t_mask0[None, :], other=0
        ).to(tl.int32)
        kq_tile0 = ((kq_tile0 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s0 += tl.dot(q_scaled_sub0, kq_tile0, out_dtype=tl.float32)

        scale_sub1 = tl.load(k_scale + scale_base1 + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        q_scaled_sub1 = q_sub * scale_sub1[None, :].to(tl.float16)
        q_zero_sum1 += tl.sum(q_scaled_sub1.to(tl.float16), axis=1)

        kq_ptrs1 = k_q + base_tok1_q[None, :] + pack_idx[:, None]
        kq_tile1 = tl.load(
            kq_ptrs1, mask=k_mask[:, None] & t_mask1[None, :], other=0
        ).to(tl.int32)
        kq_tile1 = ((kq_tile1 >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s1 += tl.dot(q_scaled_sub1, kq_tile1, out_dtype=tl.float32)

    q_zero_sum0 *= -QZERO
    q_zero_sum1 *= -QZERO
    b_s0 = (b_s0 + q_zero_sum0[:, None]) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    b_s1 = (b_s1 + q_zero_sum1[:, None]) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_persistent(
    q,
    k_q,
    k_scale,
    k_res,
    v,
    partial_out,  # [NUM_SPLITS, B, HQ, V]
    partial_lse,  # [NUM_SPLITS, B, HQ] - storing m and l
    scale,
    T,
    NTB,
    NTBS,
    th_in,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr,
    SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    USE_EXT_TH: tl.constexpr = False,
    USE_FP8_RESIDUAL: tl.constexpr = False,
    BK: tl.constexpr = 128,
    USE_PERBLOCK_SCALE: tl.constexpr = False,
    NUM_SPLITS: tl.constexpr = 1,
):
    # Grid: (NUM_SPLITS, B, HKV)
    pid_split = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    QMAX = (1 << K_BITS) - 1
    QZERO = QMAX / 2
    VALS_PER_BYTE: tl.constexpr = 8 // K_BITS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    BK_PACKED: tl.constexpr = BK // 4
    offs_k_packed = tl.arange(0, BK_PACKED)

    # Accumulators
    acc = tl.zeros([BM_DOT, V], tl.float32)
    m_i = tl.zeros([BM_DOT], tl.float32) + NEG_INF
    l_i = tl.zeros([BM_DOT], tl.float32)

    # Load Threshold
    if USE_EXT_TH:
        th_rows = tl.load(
            th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0
        )
    else:
        th_rows = tl.zeros([BM_DOT], tl.float32)  # Should not happen

    # Load Q and Q_zero_sum
    if USE_PERBLOCK_SCALE:
        scale_base_ptr_base = pid_b * (NTB * HKV * K) + pid_hkv * K
    else:
        scale_base = pid_b * (HKV * K) + pid_hkv * K

    q_zero_sum = tl.zeros([BM_DOT], tl.float32)
    offs_k_base = tl.arange(0, BK)

    # Pre-load Q (keep in register if possible, but BM_DOT*BK=16*128=2048 float16 is ok)
    # Actually we load Q per K-block loop to save registers for acc

    # Determine block range for this split
    # Total BS blocks = NTB. Split among NUM_SPLITS
    blocks_per_split = (NTB + NUM_SPLITS - 1) // NUM_SPLITS
    start_tb = pid_split * blocks_per_split
    end_tb = min(NTB, start_tb + blocks_per_split)

    for tb in range(start_tb, end_tb):
        # Current block index tb
        if USE_PERBLOCK_SCALE:
            scale_base = scale_base_ptr_base + tb * (HKV * K)

        # Calculate Q_zero_sum for this block (if per-block scale) or global
        # If per-block scale, q_zero_sum changes per block. If not, it's constant.
        # To avoid branching, just recompute or load?
        # Recomputing is safer for register pressure than caching.

        # But wait, Q is small. Let's pre-load Q if possible.
        # But scale_sub changes.
        # Optimized: Just loop K inside.

        s0 = tb * BS

        # Iterate over Sub-Blocks (SBS) within the Block (BS)
        for sb in tl.static_range(NSB):
            offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
            t_mask_sb = offs_t_sb < T

            # Compute Score for this sub-block
            base_toksb_q = (
                pid_b * (T * HKV * K_PACKED)
                + offs_t_sb * (HKV * K_PACKED)
                + (pid_hkv * K_PACKED)
            )
            tl.multiple_of(base_toksb_q, K_PACKED)

            b_s_q = tl.zeros([BM_DOT, SBS], tl.float32)
            cur_q_zero_sum = tl.zeros([BM_DOT], tl.float32)

            for k_start in tl.static_range(0, K, BK):
                # Construct packed K pointers
                # Base offset for K packed: base_toksb_q is [SBS]
                kq_ptrs_packed = (
                    k_q
                    + base_toksb_q[None, :]
                    + (k_start // 4 + offs_k_packed[:, None])
                )
                kq_packed = tl.load(
                    kq_ptrs_packed, mask=t_mask_sb[None, :], other=0
                ).to(tl.int32)

                # Unpack and Dot - Slice 0
                offs_k_0 = k_start + offs_k_packed * 4 + 0
                k_mask_0 = offs_k_0 < K
                q_ptrs_0 = (
                    q
                    + pid_b * (HQ * K)
                    + (base_hq + rows)[:, None] * K
                    + offs_k_0[None, :]
                )
                q_sub_0 = tl.load(
                    q_ptrs_0, mask=row_mask[:, None] & k_mask_0[None, :], other=0.0
                ).to(tl.float16)
                scale_sub_0 = tl.load(
                    k_scale + scale_base + offs_k_0, mask=k_mask_0, other=0.0
                ).to(tl.float32)
                q_scaled_0 = q_sub_0 * scale_sub_0[None, :].to(tl.float16)
                cur_q_zero_sum += tl.sum(q_scaled_0.to(tl.float32), axis=1)
                k0 = (kq_packed & 3).to(tl.float16)
                b_s_q += tl.dot(q_scaled_0, k0, out_dtype=tl.float32)

                # Unpack and Dot - Slice 1
                offs_k_1 = k_start + offs_k_packed * 4 + 1
                k_mask_1 = offs_k_1 < K
                q_ptrs_1 = (
                    q
                    + pid_b * (HQ * K)
                    + (base_hq + rows)[:, None] * K
                    + offs_k_1[None, :]
                )
                q_sub_1 = tl.load(
                    q_ptrs_1, mask=row_mask[:, None] & k_mask_1[None, :], other=0.0
                ).to(tl.float16)
                scale_sub_1 = tl.load(
                    k_scale + scale_base + offs_k_1, mask=k_mask_1, other=0.0
                ).to(tl.float32)
                q_scaled_1 = q_sub_1 * scale_sub_1[None, :].to(tl.float16)
                cur_q_zero_sum += tl.sum(q_scaled_1.to(tl.float32), axis=1)
                k1 = ((kq_packed >> 2) & 3).to(tl.float16)
                b_s_q += tl.dot(q_scaled_1, k1, out_dtype=tl.float32)

                # Unpack and Dot - Slice 2
                offs_k_2 = k_start + offs_k_packed * 4 + 2
                k_mask_2 = offs_k_2 < K
                q_ptrs_2 = (
                    q
                    + pid_b * (HQ * K)
                    + (base_hq + rows)[:, None] * K
                    + offs_k_2[None, :]
                )
                q_sub_2 = tl.load(
                    q_ptrs_2, mask=row_mask[:, None] & k_mask_2[None, :], other=0.0
                ).to(tl.float16)
                scale_sub_2 = tl.load(
                    k_scale + scale_base + offs_k_2, mask=k_mask_2, other=0.0
                ).to(tl.float32)
                q_scaled_2 = q_sub_2 * scale_sub_2[None, :].to(tl.float16)
                cur_q_zero_sum += tl.sum(q_scaled_2.to(tl.float32), axis=1)
                k2 = ((kq_packed >> 4) & 3).to(tl.float16)
                b_s_q += tl.dot(q_scaled_2, k2, out_dtype=tl.float32)

                # Unpack and Dot - Slice 3
                offs_k_3 = k_start + offs_k_packed * 4 + 3
                k_mask_3 = offs_k_3 < K
                q_ptrs_3 = (
                    q
                    + pid_b * (HQ * K)
                    + (base_hq + rows)[:, None] * K
                    + offs_k_3[None, :]
                )
                q_sub_3 = tl.load(
                    q_ptrs_3, mask=row_mask[:, None] & k_mask_3[None, :], other=0.0
                ).to(tl.float16)
                scale_sub_3 = tl.load(
                    k_scale + scale_base + offs_k_3, mask=k_mask_3, other=0.0
                ).to(tl.float32)
                q_scaled_3 = q_sub_3 * scale_sub_3[None, :].to(tl.float16)
                cur_q_zero_sum += tl.sum(q_scaled_3.to(tl.float32), axis=1)
                k3 = ((kq_packed >> 6) & 3).to(tl.float16)
                b_s_q += tl.dot(q_scaled_3, k3, out_dtype=tl.float32)

            cur_q_zero_sum *= -QZERO
            b_s_q = b_s_q + cur_q_zero_sum[:, None]
            b_s_q_scaled = b_s_q * scale * RCP_LN2
            b_s_act = tl.where(t_mask_sb[None, :], b_s_q_scaled, NEG_INF)

            # Threshold Check
            m_rows_blk = tl.max(b_s_act, axis=1)

            # Pruning logic
            # Keep if ANY head in the group exceeds threshold
            # Actually, per-head check: mask out heads that don't pass
            # But we load V for the whole group (G heads sharing HKV).
            # Wait, G is usually small (e.g. 3).
            # If G=1, simple. If G>1, we process them together.

            keep_mask = (m_rows_blk >= th_rows) & row_mask
            need_keep = tl.sum(keep_mask.to(tl.int32)) > 0

            if need_keep:
                # Recompute with FP8 Residual if enabled
                if USE_FP8_RESIDUAL:
                    base_toksb_k = (
                        pid_b * (T * HKV * K) + offs_t_sb * (HKV * K) + (pid_hkv * K)
                    )
                    b_s_res = tl.zeros([BM_DOT, SBS], tl.float32)
                    for k_start in tl.static_range(0, K, BK):
                        offs_k = k_start + offs_k_base
                        k_mask = offs_k < K
                        q_ptrs = (
                            q
                            + pid_b * (HQ * K)
                            + (base_hq + rows)[:, None] * K
                            + offs_k[None, :]
                        )
                        q_sub = tl.load(
                            q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
                        ).to(tl.float16)
                        k_res_ptrssb = k_res + base_toksb_k[None, :] + offs_k[:, None]
                        k_res_tile = tl.load(
                            k_res_ptrssb,
                            mask=k_mask[:, None] & t_mask_sb[None, :],
                            other=0.0,
                        ).to(tl.float16)
                        b_s_res += tl.dot(q_sub, k_res_tile, out_dtype=tl.float32)

                    b_s = (b_s_q + b_s_res) * scale * RCP_LN2
                    b_s = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
                else:
                    b_s = b_s_act

                # Online Softmax Update
                m_curr = tl.max(b_s, axis=1)
                m_new = tl.maximum(m_i, m_curr)

                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(b_s - m_new[:, None])

                # Mask out pruned heads? No, math handles it if score is low (-inf)
                # But we did thresholding on Q-only score.
                # If we skipped loading V, we treat P as 0?
                # Ideally, if keep_mask is false for a specific row, we shouldn't update its accumulators with garbage.
                # However, b_s_act should be accurate enough.
                # If a head was pruned (keep_mask=0), its m_rows_blk < th.
                # Is it possible m_curr > m_i (which is -inf initially)? Yes.
                # So we will accumulate small probabilities. This is harmless for correctness (just tiny values).

                acc = acc * alpha[:, None]
                l_i = l_i * alpha

                # Load V
                v_offs = tl.arange(0, V)
                v_ptrs = (
                    v
                    + pid_b * (T * HKV * V)
                    + (offs_t_sb[:, None] * (HKV * V))
                    + (pid_hkv * V)
                    + v_offs[None, :]
                )
                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)

                acc += tl.dot(p.to(tl.float16), b_v, out_dtype=tl.float32)
                l_i += tl.sum(p, axis=1)
                m_i = m_new

    # Store partial results
    # partial_out: [NUM_SPLITS, B, HQ, V]
    # partial_lse: [NUM_SPLITS, B, HQ, 2] (store m_i and l_i)
    # We can pack m_i and l_i together or separate.
    # Let's use separate pointers passed in.

    # Store acc
    v_offs = tl.arange(0, V)
    # New Layout: [B, HQ, NUM_SPLITS, V]
    off_out = (
        pid_b * (HQ * NUM_SPLITS * V)
        + (base_hq + rows)[:, None] * (NUM_SPLITS * V)
        + pid_split * V
        + v_offs[None, :]
    )
    tl.store(partial_out + off_out, acc, mask=row_mask[:, None])

    # Store LSE info: m_i and l_i
    # New Layout: [B, HQ, NUM_SPLITS, 2]
    # 0 -> m, 1 -> l

    off_lse_base = (
        pid_b * (HQ * NUM_SPLITS * 2)
        + (base_hq + rows) * (NUM_SPLITS * 2)
        + pid_split * 2
    )
    off_lse_m = off_lse_base + 0
    off_lse_l = off_lse_base + 1

    tl.store(partial_lse + off_lse_m, m_i, mask=row_mask)
    tl.store(partial_lse + off_lse_l, l_i, mask=row_mask)


@triton.jit
def attn_reduce_persistent(
    partial_out,  # [B, HQ, NUM_SPLITS, V]
    partial_lse,  # [B, HQ, NUM_SPLITS, 2]
    final_out,  # [B, HQ, V]
    B: tl.constexpr,
    HQ: tl.constexpr,
    V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # One block per (B, HQ)
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    m_global = float("-inf")
    l_global = 0.0
    acc_global = tl.zeros([V], tl.float32)

    v_offs = tl.arange(0, V)

    # Base offsets for this (B, HQ)
    base_lse = pid_b * (HQ * NUM_SPLITS * 2) + pid_hq * (NUM_SPLITS * 2)
    base_out = pid_b * (HQ * NUM_SPLITS * V) + pid_hq * (NUM_SPLITS * V)

    for s in range(NUM_SPLITS):
        # Load m, l
        # Contiguous in memory: [s*2, s*2+1]
        # But we load scalar m, l.
        off_lse = s * 2
        m_s = tl.load(partial_lse + base_lse + off_lse)
        l_s = tl.load(partial_lse + base_lse + off_lse + 1)

        # Load acc
        # Contiguous in memory: [s*V : (s+1)*V]
        off_out = s * V + v_offs
        acc_s = tl.load(partial_out + base_out + off_out)

        # Merge
        m_new = tl.maximum(m_global, m_s)
        alpha_global = tl.exp2(m_global - m_new)
        alpha_s = tl.exp2(m_s - m_new)

        acc_global = acc_global * alpha_global + acc_s * alpha_s
        l_global = l_global * alpha_global + l_s * alpha_s
        m_global = m_new

    # Finalize
    # Avoid div by zero
    out = acc_global / l_global
    out = tl.where(l_global > 0, out, 0.0)

    off_final = pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(final_out + off_final, out.to(final_out.dtype.element_ty))


def _normalize_scale(
    k_scale: torch.Tensor, expect_shape, allow_perblock: bool = False, NTB: int = None
):
    if k_scale.ndim == 4:
        if k_scale.shape[1] == 1:
            k_scale = k_scale.squeeze(1)
        elif allow_perblock and NTB is not None and k_scale.shape[1] == NTB:
            B, _, HKV, K = k_scale.shape
            expected_perblock = (B, NTB, HKV, K)
            if k_scale.shape != expected_perblock:
                raise ValueError(
                    f"Per-block k_scale shape mismatch: {k_scale.shape=}, expected {expected_perblock}"
                )
            return k_scale.contiguous(), True

    if k_scale.shape != expect_shape:
        raise ValueError(
            f"Unsupported k_scale shape: {k_scale.shape=}, expected {expect_shape}"
        )

    return k_scale.contiguous(), False


def _kernel_kwargs(num_warps: int | None, num_stages: int | None) -> dict:
    kwargs = {}
    if num_warps is not None:
        if num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {num_warps}")
        kwargs["num_warps"] = int(num_warps)
    if num_stages is not None:
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        kwargs["num_stages"] = int(num_stages)
    return kwargs


def _record_kernel_time(kernel_times: dict | None, name: str, fn, device) -> None:
    if kernel_times is None:
        fn()
        return
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    kernel_times[name] = start.elapsed_time(end)


def attn_forward_decode_quantized(
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    k_residual: torch.Tensor | None = None,
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    use_fp8_residual: bool = True,
    num_warps_th: int | None = None,
    num_stages_th: int | None = None,
    num_warps_s1: int | None = None,
    num_stages_s1: int | None = None,
    # New args for persistent mode
    num_splits: int = 128,  # Tunable
    return_kernel_timings: bool = False,
    **kwargs,
):
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_residual is not None and not k_residual.is_cuda:
        raise ValueError("k_residual must be a CUDA tensor when provided")
    if k_bits != 2:
        raise ValueError(
            f"attn_forward_decode_quantized currently supports 2-bit keys, got k_bits={k_bits}"
        )
    assert k_scale.is_cuda, "k_scale must be a CUDA tensor"

    B, Tq, HQ, K = q.shape
    Bk, T, HKV, K_packed = k_q.shape
    Bv, Tv, HKVv, V = v.shape
    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    assert K_packed == expected_k_packed
    assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV
    G = HQ // HKV

    expect_shape = (B, HKV, K)
    k_scale, use_perblock_scale = _normalize_scale(
        k_scale, expect_shape, allow_perblock=True, NTB=triton.cdiv(T, BS)
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    q = q.contiguous()
    k_q = k_q.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_q
    v = v.contiguous()
    kernel_times = {} if return_kernel_timings else None

    # Threshold
    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)

        def _launch_threshold():
            attn_compute_threshold_qbits[(B, HKV)](
                q,
                k_q,
                k_scale,
                threshold_buf,
                scale,
                T,
                NTB,
                delta,
                B=B,
                HKV=HKV,
                HQ=HQ,
                K=K,
                K_PACKED=K_packed,
                G=G,
                BS=BS,
                K_BITS=k_bits,
                USE_PERBLOCK_SCALE=use_perblock_scale,
                **th_kwargs,
            )

        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)
        use_ext_th = True

    # Persistent Accumulation
    # Buffers for Split-K
    # Adjust NUM_SPLITS based on NTB
    # If NTB is small, reduce NUM_SPLITS
    eff_splits = min(num_splits, NTB)

    partial_out = torch.empty(
        (B, HQ, eff_splits, V), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (B, HQ, eff_splits, 2), device=q.device, dtype=torch.float32
    )

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)

    def _launch_persistent():
        attn_forward_stage1_persistent[(eff_splits, B, HKV)](
            q,
            k_q,
            k_scale,
            k_res,
            v,
            partial_out,
            partial_lse,
            scale,
            T,
            NTB,
            NTBS,
            threshold_buf,
            B=B,
            HKV=HKV,
            HQ=HQ,
            K=K,
            K_PACKED=K_packed,
            V=V,
            G=G,
            BS=BS,
            SBS=SBS,
            K_BITS=k_bits,
            USE_EXT_TH=use_ext_th,
            USE_FP8_RESIDUAL=use_fp8_residual,
            USE_PERBLOCK_SCALE=use_perblock_scale,
            NUM_SPLITS=eff_splits,
            **s1_kwargs,
        )

    _record_kernel_time(kernel_times, "stage1_persistent", _launch_persistent, q.device)

    # Reduction
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)

    def _launch_reduce():
        attn_reduce_persistent[(B, HQ)](
            partial_out, partial_lse, o, B=B, HQ=HQ, V=V, NUM_SPLITS=eff_splits
        )

    _record_kernel_time(kernel_times, "stage2_reduce", _launch_reduce, q.device)

    # Note: Skip ratio is hard to calculate in persistent kernel without atomic counters.
    # We return 0.0 or a dummy value.
    skip_ratio = 0.0

    if return_skip_ratio:
        if return_kernel_timings:
            return o, skip_ratio, kernel_times
        return o, skip_ratio
    if return_kernel_timings:
        return o, kernel_times
    return o


class CUDAGraphDecodeRunnerQ2FP8PersistentBK128:
    def __init__(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        num_splits: int = 128,
        use_fp8_residual: bool = True,
        warmup: int = 2,
        num_warps_th: Optional[int] = None,
        num_stages_th: Optional[int] = None,
        num_warps_s1: Optional[int] = None,
        num_stages_s1: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDAGraph capture.")

        self._device = q.device
        self._k_bits = k_bits
        self._scale = scale
        self._BS = BS
        self._SBS = SBS
        self._delta = delta
        self._num_splits = num_splits
        self._use_fp8_residual = use_fp8_residual
        self._use_ext_th = precomputed_threshold is not None
        self._num_warps_th = num_warps_th
        self._num_stages_th = num_stages_th
        self._num_warps_s1 = num_warps_s1
        self._num_stages_s1 = num_stages_s1

        if self._use_fp8_residual and k_residual is None:
            raise ValueError("use_fp8_residual=True requires k_residual")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required when use_ext_th=True")

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_k_q = torch.empty_like(k_q, device=self._device)
        self._static_k_scale = torch.empty_like(k_scale, device=self._device)
        self._static_v = torch.empty_like(v, device=self._device)
        self._static_k_residual = None
        if self._use_fp8_residual:
            self._static_k_residual = torch.empty_like(k_residual, device=self._device)

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(
                precomputed_threshold, device=self._device
            )

        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                num_splits=self._num_splits,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
            )
        torch.cuda.synchronize(self._device)

        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            self._static_out = attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                k_residual=self._static_k_residual,
                v=self._static_v,
                k_bits=self._k_bits,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                num_splits=self._num_splits,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                num_warps_th=self._num_warps_th,
                num_stages_th=self._num_stages_th,
                num_warps_s1=self._num_warps_s1,
                num_stages_s1=self._num_stages_s1,
            )

    @property
    def output(self) -> torch.Tensor:
        return self._static_out

    def replay(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")

        self._static_q.copy_(q)
        self._static_k_q.copy_(k_q)
        self._static_k_scale.copy_(k_scale)
        self._static_v.copy_(v)
        if self._use_fp8_residual:
            self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        return self._static_out, 0.0

    def replay_only(self) -> torch.Tensor:
        self._graph.replay()
        return self._static_out

    __call__ = replay
