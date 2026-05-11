"""
FFD paged decode interface.

This module resolves local decode kernels inside ffd-core.
"""

from ..kernels.paged_decode_kernel import (
    CUDAGraphDecodeRunnerQ2FP8Unified as CUDAGraphDecodeRunnerQ2FP8,
    attn_forward_decode_quantized,
)


def attn_forward_decode(
    *,
    q,
    k_q,
    k_scale,
    v,
    k_residual=None,
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    return_lse: bool = False,  # NEW: return (m, l) for accurate merging with k_current
    use_fp8_residual: bool = True,
    cudagraph_runner=None,  # NEW: CUDA Graph runner for acceleration
    # New args for fused current block
    k_current=None,
    v_current=None,
    current_len=0,
    **kwargs,
):
    """
    Q2FP8 Symmetric decode attention interface.

    Args:
        q: [B, 1, HQ, K] Query tensor
        k_q: [B, T, HKV, K_packed] 2-bit quantized K (packed uint8)
        k_scale: [B, HKV, K] (global) or [B, NTB, HKV, K] (per-block) quant scale
        v: [B, T, HKV, V] Full V cache
        k_residual: [B, T, HKV, K] FP8 residual (optional)
        k_bits: Quantization bits (default 2)
        scale: attention scale (default 1/sqrt(K))
        BS: Block size (default 128)
        SBS: Sub-block size (default equal to BS)
        delta: Threshold offset (default 5.0)
        return_skip_ratio: Whether to return skip ratio
        return_lse: Whether to return (m, l) for accurate merging with k_current
        use_fp8_residual: Whether to use FP8 residual
        cudagraph_runner: CUDA Graph runner for kernel acceleration
        k_current: [B, BS, HKV, K] Current FP16 block
        v_current: [B, BS, HKV, V] Current FP16 block
        current_len: int Current block valid length
        **kwargs: Additional arguments

    Returns:
        if return_lse:
            attn_output: [B, HQ, V], final_m: [B, HQ], final_l: [B, HQ]
            (if return_skip_ratio: adds skip_ratio)
        else:
            attn_output: [B, HQ, V] Attention output
            skip_ratio (optional): Fraction of blocks skipped
    """
    # Drop obsolete kwargs from older integrations.
    kwargs.pop("k_sample", None)
    kwargs.pop("k_full", None)
    kwargs.pop("return_lse", None)

    # Use CUDA Graph if available and conditions are met
    # Note: CUDA Graph now handles merge internally if initialized with k_current
    # But checking return_lse is tricky. If fused, we don't strictly need lse returned unless requested.
    # The runner supports returning lse if captured with it.
    if cudagraph_runner is not None and not return_skip_ratio and not return_lse:
        # CUDA Graph path - fastest, but doesn't support skip_ratio or lse
        # Map k_current/v_current to k_new/v_new
        return cudagraph_runner.replay(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_new=k_current,
            v_new=v_current,
            k_residual=k_residual,
            current_len=current_len,
        )
    else:
        # Standard path - supports all features
        # Map k_current/v_current to k_new/v_new
        return attn_forward_decode_quantized(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            v=v,
            k_new=k_current,
            v_new=v_current,
            k_residual=k_residual,
            k_bits=k_bits,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=return_skip_ratio,
            return_lse=return_lse,
            use_fp8_residual=use_fp8_residual,
            current_len=current_len,
            **kwargs,
        )
