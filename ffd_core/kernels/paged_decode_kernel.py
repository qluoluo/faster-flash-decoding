from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QUANT_MODE = "sym_persistent_bk128_unified"


# Reuse existing threshold kernel
@triton.jit
def attn_compute_threshold_qbits(
    q,
    k_q,
    k_scale,
    th_out,
    scale,
    T,
    T_BUFFER,  # Stride parameter
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

    # Initialize global max scores
    m_global = tl.zeros([BM_DOT], tl.float32) + NEG_INF

    # Pre-calculate scale base pointers if needed
    scale_base_ptr_base = 0
    if USE_PERBLOCK_SCALE:
        # Assume k_scale covers buffer
        NTB_BUFFER = tl.cdiv(T_BUFFER, BS)
        scale_base_ptr_base = pid_b * (NTB_BUFFER * HKV * K) + pid_hkv * K
    else:
        scale_base_common = pid_b * (HKV * K) + pid_hkv * K

    offs_k_base = tl.arange(0, BK)

    # --- BLOCK 1: First Block (tb=0) ---
    tb = 0
    if USE_PERBLOCK_SCALE:
        scale_base = scale_base_ptr_base  # tb=0
    else:
        scale_base = scale_base_common

    offs_t = tb * BS + tl.arange(0, BS)
    t_mask = offs_t < T

    base_tok_q = (
        pid_b * (T_BUFFER * HKV * K_PACKED)
        + offs_t * (HKV * K_PACKED)
        + (pid_hkv * K_PACKED)
    )

    tl.multiple_of(base_tok_q, K_PACKED)

    b_s = tl.zeros([BM_DOT, BS], tl.float32)
    q_zero_sum = tl.zeros([BM_DOT], tl.float32)

    for k_start in tl.static_range(0, K, BK):
        offs_k = k_start + offs_k_base
        k_mask = offs_k < K
        pack_idx = offs_k // VALS_PER_BYTE
        pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

        q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_sub = tl.load(q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float16
        )

        scale_sub = tl.load(k_scale + scale_base + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
        q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)

        kq_ptrs = k_q + base_tok_q[None, :] + pack_idx[:, None]
        kq_tile = tl.load(kq_ptrs, mask=k_mask[:, None] & t_mask[None, :], other=0).to(
            tl.int32
        )
        kq_tile = ((kq_tile >> pack_shifts[:, None]) & QMAX).to(tl.float16)
        b_s += tl.dot(q_scaled_sub, kq_tile, out_dtype=tl.float32)

    q_zero_sum *= -QZERO
    b_s = (b_s + q_zero_sum[:, None]) * scale * RCP_LN2
    b_s = tl.where(t_mask[None, :], b_s, NEG_INF)
    m_global = tl.max(b_s, axis=1)

    # --- BLOCK 2: Last Block (tb=NTB-1) ---
    # We only check if NTB > 1. If NTB == 1, we already checked it.
    # GEMINI FIX: Re-enabled Last Block check to match isolated test behavior (high sparsity).
    if NTB > 1:
        tb = NTB - 1
        if USE_PERBLOCK_SCALE:
            scale_base = scale_base_ptr_base + tb * (HKV * K)
        else:
            scale_base = scale_base_common

        offs_t = tb * BS + tl.arange(0, BS)
        t_mask = offs_t < T

        base_tok_q = (
            pid_b * (T_BUFFER * HKV * K_PACKED)
            + offs_t * (HKV * K_PACKED)
            + (pid_hkv * K_PACKED)
        )
        tl.multiple_of(base_tok_q, K_PACKED)

        b_s = tl.zeros([BM_DOT, BS], tl.float32)
        q_zero_sum = tl.zeros([BM_DOT], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            pack_idx = offs_k // VALS_PER_BYTE
            pack_shifts = (offs_k % VALS_PER_BYTE) * K_BITS

            q_ptrs = (
                q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            )
            q_sub = tl.load(
                q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
            ).to(tl.float16)

            scale_sub = tl.load(
                k_scale + scale_base + offs_k, mask=k_mask, other=0.0
            ).to(tl.float32)
            q_scaled_sub = q_sub * scale_sub[None, :].to(tl.float16)
            q_zero_sum += tl.sum(q_scaled_sub.to(tl.float32), axis=1)

            kq_ptrs = k_q + base_tok_q[None, :] + pack_idx[:, None]
            kq_tile = tl.load(
                kq_ptrs, mask=k_mask[:, None] & t_mask[None, :], other=0
            ).to(tl.int32)
            kq_tile = ((kq_tile >> pack_shifts[:, None]) & QMAX).to(tl.float16)
            b_s += tl.dot(q_scaled_sub, kq_tile, out_dtype=tl.float32)

        q_zero_sum *= -QZERO
        b_s = (b_s + q_zero_sum[:, None]) * scale * RCP_LN2
        b_s = tl.where(t_mask[None, :], b_s, NEG_INF)
        m_curr = tl.max(b_s, axis=1)
        m_global = tl.maximum(m_global, m_curr)

    # Compute threshold
    th_rows = m_global - delta
    th_ptrs = th_out + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


@triton.jit
def attn_forward_stage1_persistent_unified(
    q,
    k_q,
    k_scale,
    k_res,
    v,
    partial_out,  # [NUM_SPLITS, B, HQ, V]
    partial_lse,  # [NUM_SPLITS, B, HQ] - storing m and l
    scale,
    T,
    T_BUFFER,  # Stride parameter
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
    counters=None,  # [2] int32: [total, kept]
    ENABLE_COUNT: tl.constexpr = False,
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
        th_rows = tl.zeros([BM_DOT], tl.float32)

    # Load Q and Q_zero_sum
    scale_base_ptr_base = 0
    if USE_PERBLOCK_SCALE:
        NTB_BUFFER = tl.cdiv(T_BUFFER, BS)
        scale_base_ptr_base = pid_b * (NTB_BUFFER * HKV * K) + pid_hkv * K
    else:
        scale_base = pid_b * (HKV * K) + pid_hkv * K

    q_zero_sum = tl.zeros([BM_DOT], tl.float32)
    offs_k_base = tl.arange(0, BK)

    blocks_per_split = (NTB + NUM_SPLITS - 1) // NUM_SPLITS
    start_tb = pid_split * blocks_per_split
    end_tb = min(NTB, start_tb + blocks_per_split)

    total_subblocks = 0
    kept_subblocks = 0

    for tb in range(start_tb, end_tb):
        if USE_PERBLOCK_SCALE:
            scale_base = scale_base_ptr_base + tb * (HKV * K)

        s0 = tb * BS

        for sb in tl.static_range(NSB):
            offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
            t_mask_sb = offs_t_sb < T

            if ENABLE_COUNT:
                total_subblocks += 1

            # Compute Score for this sub-block
            base_toksb_q = (
                pid_b * (T_BUFFER * HKV * K_PACKED)
                + offs_t_sb * (HKV * K_PACKED)
                + (pid_hkv * K_PACKED)
            )

            tl.multiple_of(base_toksb_q, K_PACKED)

            b_s_q = tl.zeros([BM_DOT, SBS], tl.float32)
            cur_q_zero_sum = tl.zeros([BM_DOT], tl.float32)

            for k_start in tl.static_range(0, K, BK):
                # Construct packed K pointers
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

            keep_mask = (m_rows_blk >= th_rows) & row_mask
            need_keep = tl.sum(keep_mask.to(tl.int32)) > 0

            if ENABLE_COUNT:
                if need_keep:
                    kept_subblocks += 1

            if need_keep:
                # Recompute with FP8 Residual if enabled
                if USE_FP8_RESIDUAL:
                    base_toksb_k = (
                        pid_b * (T_BUFFER * HKV * K)
                        + offs_t_sb * (HKV * K)
                        + (pid_hkv * K)
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

                # Safe online softmax update
                mask = m_new > NEG_INF
                alpha = tl.where(mask, tl.exp2(m_i - m_new), 0.0)
                p = tl.where(mask[:, None], tl.exp2(b_s - m_new[:, None]), 0.0)

                acc = tl.where(mask[:, None], acc * alpha[:, None], acc)
                l_i = tl.where(mask, l_i * alpha, l_i)

                v_offs = tl.arange(0, V)
                v_ptrs = (
                    v
                    + pid_b * (T_BUFFER * HKV * V)
                    + (offs_t_sb[:, None] * (HKV * V))
                    + (pid_hkv * V)
                    + v_offs[None, :]
                )

                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)

                acc += tl.dot(p.to(tl.float16), b_v, out_dtype=tl.float32)
                l_i += tl.sum(p, axis=1)
                m_i = tl.where(mask, m_new, m_i)

    v_offs = tl.arange(0, V)
    # New Layout: [B, HQ, NUM_SPLITS, V]
    off_out = (
        pid_b * (HQ * NUM_SPLITS * V)
        + (base_hq + rows)[:, None] * (NUM_SPLITS * V)
        + pid_split * V
        + v_offs[None, :]
    )
    tl.store(partial_out + off_out, acc, mask=row_mask[:, None])

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

    if ENABLE_COUNT:
        tl.atomic_add(counters + 0, total_subblocks)
        tl.atomic_add(counters + 1, kept_subblocks)


@triton.jit
def attn_forward_stage1_tail(
    q,
    k_new,
    v_new,
    partial_out,
    partial_lse,
    scale,
    T_new,
    current_len,
    B: tl.constexpr,
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    BK: tl.constexpr = 128,
    SPLIT_IDX: tl.constexpr = 0,
    NUM_SPLITS: tl.constexpr = 1,
    IS_CURRENT_LEN_PTR: tl.constexpr = False,
):
    # Grid: (B, HKV)
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    if IS_CURRENT_LEN_PTR:
        cur_len_val = tl.load(current_len)
    else:
        cur_len_val = current_len

    # Initialize acc, m, l
    v_offs = tl.arange(0, V)
    acc = tl.zeros([BM_DOT, V], tl.float32)
    m_i = tl.zeros([BM_DOT], tl.float32) + NEG_INF
    l_i = tl.zeros([BM_DOT], tl.float32)

    # Compute attention for T_new
    # Loop over T_new in chunks of BK (reuse BK as time-chunk size for simplicity)
    # Actually, let's use 128 as block size for T_new
    T_BLOCK: tl.constexpr = 64
    num_new_blocks = tl.cdiv(T_new, T_BLOCK)

    offs_k_base = tl.arange(0, BK)

    for nb in range(num_new_blocks):
        t_start = nb * T_BLOCK
        offs_t_current = t_start + tl.arange(0, T_BLOCK)
        t_mask_new = (offs_t_current < T_new) & (offs_t_current < cur_len_val)

        b_s_new = tl.zeros([BM_DOT, T_BLOCK], tl.float32)

        for k_start in tl.static_range(0, K, BK):
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K

            q_ptrs = (
                q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
            )
            q_sub = tl.load(
                q_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
            ).to(tl.float16)

            k_new_ptrs = (
                k_new
                + pid_b * (T_new * HKV * K)
                + offs_t_current[None, :] * (HKV * K)
                + pid_hkv * K
                + offs_k[:, None]
            )

            k_new_tile = tl.load(
                k_new_ptrs, mask=k_mask[:, None] & t_mask_new[None, :], other=0.0
            ).to(tl.float16)

            b_s_new += tl.dot(q_sub, k_new_tile, out_dtype=tl.float32)

        b_s_new = b_s_new * scale * RCP_LN2
        b_s_new = tl.where(t_mask_new[None, :], b_s_new, NEG_INF)

        m_curr = tl.max(b_s_new, axis=1)
        m_new_val = tl.maximum(m_i, m_curr)

        # Safe online softmax update
        mask = m_new_val > NEG_INF
        alpha = tl.where(mask, tl.exp2(m_i - m_new_val), 0.0)
        p = tl.where(mask[:, None], tl.exp2(b_s_new - m_new_val[:, None]), 0.0)

        acc = tl.where(mask[:, None], acc * alpha[:, None], acc)
        l_i = tl.where(mask, l_i * alpha, l_i)

        v_new_ptrs = (
            v_new
            + pid_b * (T_new * HKV * V)
            + offs_t_current[:, None] * (HKV * V)
            + pid_hkv * V
            + v_offs[None, :]
        )

        b_v_new = tl.load(v_new_ptrs, mask=t_mask_new[:, None], other=0.0).to(
            tl.float16
        )

        acc += tl.dot(p.to(tl.float16), b_v_new, out_dtype=tl.float32)
        l_i += tl.sum(p, axis=1)
        m_i = tl.where(mask, m_new_val, m_i)

    # Store back
    off_out = (
        pid_b * (HQ * NUM_SPLITS * V)
        + (base_hq + rows)[:, None] * (NUM_SPLITS * V)
        + SPLIT_IDX * V
        + v_offs[None, :]
    )
    off_lse_base = (
        pid_b * (HQ * NUM_SPLITS * 2)
        + (base_hq + rows) * (NUM_SPLITS * 2)
        + SPLIT_IDX * 2
    )
    tl.store(partial_out + off_out, acc, mask=row_mask[:, None])
    tl.store(partial_lse + off_lse_base + 0, m_i, mask=row_mask)
    tl.store(partial_lse + off_lse_base + 1, l_i, mask=row_mask)


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
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    NEG_INF = float("-inf")
    m_global = float("-inf")
    l_global = 0.0
    acc_global = tl.zeros([V], tl.float32)

    v_offs = tl.arange(0, V)

    base_lse = pid_b * (HQ * NUM_SPLITS * 2) + pid_hq * (NUM_SPLITS * 2)
    base_out = pid_b * (HQ * NUM_SPLITS * V) + pid_hq * (NUM_SPLITS * V)

    for s in range(NUM_SPLITS):
        off_lse = s * 2
        m_s = tl.load(partial_lse + base_lse + off_lse)
        l_s = tl.load(partial_lse + base_lse + off_lse + 1)

        off_out = s * V + v_offs
        acc_s = tl.load(partial_out + base_out + off_out)

        m_new = tl.maximum(m_global, m_s)

        # Safe reduction update
        mask = m_new > NEG_INF
        alpha_global = tl.where(mask, tl.exp2(m_global - m_new), 0.0)
        alpha_s = tl.where(mask, tl.exp2(m_s - m_new), 0.0)

        acc_global = tl.where(
            mask[:, None], acc_global * alpha_global + acc_s * alpha_s, acc_global
        )
        l_global = tl.where(mask, l_global * alpha_global + l_s * alpha_s, l_global)
        m_global = tl.where(mask, m_new, m_global)

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
        elif allow_perblock and NTB is not None and k_scale.shape[1] >= NTB:
            B, _, HKV, K = k_scale.shape
            expected_B, expected_HKV, expected_K = expect_shape
            if B != expected_B or HKV != expected_HKV or K != expected_K:
                raise ValueError(
                    f"Per-block k_scale shape mismatch: {k_scale.shape=}, expected (B={expected_B}, >=NTB={NTB}, HKV={expected_HKV}, K={expected_K})"
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
    k_new: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
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
    num_splits: int = 128,
    return_kernel_timings: bool = False,
    current_len: int | torch.Tensor = 0,
    force_compute_threshold: bool = False,
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
    Bk, T_buffer, HKV, K_packed = k_q.shape
    Bv, Tv, HKVv, V = v.shape

    # Extract valid length from kwargs
    T_valid = kwargs.get("seq_len", T_buffer)
    if T_valid is None:
        T_valid = T_buffer

    if k_new is not None:
        B_new, T_new, HKV_new, K_new = k_new.shape
        assert B == B_new
        assert HKV == HKV_new
        assert K == K_new
    else:
        T_new = 0

    vals_per_byte = 8 // k_bits
    expected_k_packed = (K + vals_per_byte - 1) // vals_per_byte
    assert K_packed == expected_k_packed
    assert B == Bk == Bv and Tq == 1 and HKVv == HKV
    G = HQ // HKV

    # Calculate NTB based on VALID length
    NTB = triton.cdiv(T_valid, BS)

    expect_shape = (B, HKV, K)
    # Allow k_scale to be larger (buffer)
    k_scale, use_perblock_scale = _normalize_scale(
        k_scale, expect_shape, allow_perblock=True, NTB=NTB
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    q = q.contiguous()
    k_q = k_q.contiguous()
    use_fp8_residual = use_fp8_residual and (k_residual is not None)
    k_res = k_residual.contiguous() if use_fp8_residual else k_q
    v = v.contiguous()
    if k_new is not None:
        k_new = k_new.contiguous()
    if v_new is not None:
        v_new = v_new.contiguous()

    kernel_times = {} if return_kernel_timings else None

    # Threshold
    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        use_ext_th = True  # Always use external threshold buffer logic

    th_kwargs = _kernel_kwargs(num_warps_th, num_stages_th)

    def _launch_threshold():
        attn_compute_threshold_qbits[(B, HKV)](
            q,
            k_q,
            k_scale,
            threshold_buf,
            scale,
            T_valid,
            T_buffer,
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

    # Launch threshold if not precomputed OR forced
    if precomputed_threshold is None or force_compute_threshold:
        _record_kernel_time(kernel_times, "threshold", _launch_threshold, q.device)

    # Counters for skip ratio
    if return_skip_ratio:
        counters = torch.zeros((2,), device=q.device, dtype=torch.int32)
    else:
        counters = None

    # Persistent Accumulation
    eff_splits = min(num_splits, NTB)
    if eff_splits < 1:
        eff_splits = 1

    # If we have a tail, we need an extra split for it
    # We use NTB > 0 check? No, T_new is better.
    # But wait, T_new is calculated later. Let's calculate it earlier.
    if k_new is not None:
        T_new = k_new.shape[1]
    else:
        T_new = 0

    max_splits = eff_splits + (1 if T_new > 0 else 0)

    partial_out = torch.empty(
        (B, HQ, max_splits, V), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (B, HQ, max_splits, 2), device=q.device, dtype=torch.float32
    )

    s1_kwargs = _kernel_kwargs(num_warps_s1, num_stages_s1)

    def _launch_persistent():
        attn_forward_stage1_persistent_unified[(eff_splits, B, HKV)](
            q,
            k_q,
            k_scale,
            k_res,
            v,
            partial_out,
            partial_lse,
            scale,
            T_valid,
            T_buffer,
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
            NUM_SPLITS=max_splits,
            counters=counters,
            ENABLE_COUNT=return_skip_ratio,
            **s1_kwargs,
        )

    _record_kernel_time(kernel_times, "stage1_persistent", _launch_persistent, q.device)

    # Tail (Unquantized)
    num_total_splits = eff_splits
    if T_new > 0:
        is_current_len_ptr = isinstance(current_len, torch.Tensor)
        # Tail uses an additional split index
        tail_split_idx = eff_splits
        num_total_splits += 1

        def _launch_tail():
            attn_forward_stage1_tail[(B, HKV)](
                q,
                k_new,
                v_new,
                partial_out,
                partial_lse,
                scale,
                T_new,
                current_len,
                B=B,
                HKV=HKV,
                HQ=HQ,
                K=K,
                V=V,
                G=G,
                SPLIT_IDX=tail_split_idx,
                NUM_SPLITS=num_total_splits,
                IS_CURRENT_LEN_PTR=is_current_len_ptr,
                **s1_kwargs,
            )

        _record_kernel_time(kernel_times, "stage1_tail", _launch_tail, q.device)

    # Reduction
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)

    def _launch_reduce():
        attn_reduce_persistent[(B, HQ)](
            partial_out, partial_lse, o, B=B, HQ=HQ, V=V, NUM_SPLITS=num_total_splits
        )

    _record_kernel_time(kernel_times, "stage2_reduce", _launch_reduce, q.device)

    # Calculate skip ratio
    skip_ratio = 0.0
    if return_skip_ratio and counters is not None:
        c_cpu = counters.cpu()
        total = c_cpu[0].item()
        kept = c_cpu[1].item()
        if total > 0:
            skip_ratio = 1.0 - (kept / total)

    if return_skip_ratio:
        if return_kernel_timings:
            return o, skip_ratio, kernel_times
        return o, skip_ratio
    if return_kernel_timings:
        return o, kernel_times
    return o


class CUDAGraphDecodeRunnerQ2FP8Unified:
    def __init__(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_new: torch.Tensor | None = None,
        v_new: torch.Tensor | None = None,
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
        current_len: int | torch.Tensor = 0,
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

        # OPTIMIZATION: Use references for large buffers to avoid massive copies per step.
        # We assume these buffers (k_q, k_scale, v, etc.) are stable (pre-allocated).
        # We only allocate static buffers for inputs that change every step (q, current_len).

        self._static_q = torch.empty_like(q, device=self._device)
        self._static_q.copy_(q)

        self._static_k_q = k_q
        self._static_k_scale = k_scale
        self._static_v = v
        self._static_k_new = k_new
        self._static_v_new = v_new
        self._static_k_residual = k_residual
        self._static_threshold = precomputed_threshold
        if self._static_threshold is None:
            # Always allocate a static threshold buffer to avoid dynamic allocation during capture
            self._static_threshold = torch.empty(
                (q.shape[0], q.shape[2]), device=self._device, dtype=torch.float32
            )

        # Allocate static current_len buffer
        self._static_current_len = torch.zeros(
            (1,), dtype=torch.int32, device=self._device
        )
        if isinstance(current_len, int):
            self._static_current_len.fill_(current_len)
        elif isinstance(current_len, torch.Tensor):
            self._static_current_len.copy_(current_len)

        # Warmup
        for _ in range(max(1, warmup)):
            attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                v=self._static_v,
                k_new=self._static_k_new,
                v_new=self._static_v_new,
                k_residual=self._static_k_residual,
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
                current_len=self._static_current_len,
                force_compute_threshold=not self._use_ext_th,
            )
        torch.cuda.synchronize(self._device)

        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            self._static_out = attn_forward_decode_quantized(
                q=self._static_q,
                k_q=self._static_k_q,
                k_scale=self._static_k_scale,
                v=self._static_v,
                k_new=self._static_k_new,
                v_new=self._static_v_new,
                k_residual=self._static_k_residual,
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
                current_len=self._static_current_len,
                force_compute_threshold=not self._use_ext_th,
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
        k_new: torch.Tensor | None,
        v_new: torch.Tensor | None,
        *,
        k_residual: Optional[torch.Tensor] = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
        return_lse: bool = False,
        current_len: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        # OPTIMIZATION: Only copy dynamic inputs.
        # Assume k_q, v, etc. are the SAME tensors (same address) as passed in __init__.
        # If they changed address, this runner is invalid!

        self._static_q.copy_(q)

        if current_len is not None:
            if isinstance(current_len, int):
                self._static_current_len.fill_(current_len)
            elif isinstance(current_len, torch.Tensor):
                self._static_current_len.copy_(current_len)

        if self._use_ext_th and precomputed_threshold is not None:
            self._static_threshold.copy_(precomputed_threshold)

        # We do NOT copy k_q, v, k_scale, k_residual.
        # We assume the user is using pre-allocated buffers that don't move.

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        return self._static_out, 0.0

    def replay_only(self) -> torch.Tensor:
        self._graph.replay()
        return self._static_out

    __call__ = replay
