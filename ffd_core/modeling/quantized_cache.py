"""
Quantized cache: symmetric quantized K cache + FP8 residual.

Page-wise quantization strategy:
- Each block/page is quantized independently with its own scale
- New tokens accumulate in the current block; quantization happens when full
- Incomplete blocks stay in FP16, unquantized

Supported quantization bit-widths:
- 2-bit: 4 values packed into 1 uint8, QMAX=3, QZERO=1.5
- 4-bit: 2 values packed into 1 uint8, QMAX=15, QZERO=7.5

Symmetric quantization formula:
- scale = abs_max / QZERO
- q = round(k / scale + QZERO)
- dequant = (q - QZERO) * scale

Data layout:
- k_q: [B, num_full_blocks * BS, HKV, K_packed] Quantized full blocks
- k_scale: [B, num_full_blocks, HKV, K] Per-block scale
- k_residual: [B, num_full_blocks * BS, HKV, K] FP8 residual
- k_current: [B, current_len, HKV, K] Current incomplete block (FP16)
- v: [B, T, HKV, V] Full V cache

This is a memory-optimized version:
1. All dequantization logic is removed.
2. update() returns empty tensors (shape=[B, 0, ...]).
   - This allows .transpose() in llama.py to succeed without error.
   - If the Flash Attention path is taken, seqlen=0 triggers a RuntimeError,
     ensuring only the sparse decode path is used.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

SUPPORTED_K_BITS = (2, 4)


def quantize_symmetric(k: torch.Tensor, k_bits: int = 2, eps: float = 1e-8):
    """
    Symmetric quantization of K.

    Args:
        k: [B, T, HKV, K] FP16/BF16 K tensor
        k_bits: Quantization bits (2 or 4)
        eps: Small epsilon to avoid division by zero

    Returns:
        k_q: [B, T, HKV, K_packed] Quantized K (packed into uint8)
        k_scale: [B, HKV, K] Quantization scale
        k_residual: [B, T, HKV, K] FP8 residual
    """
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")

    B, T, HKV, K = k.shape
    dtype = k.dtype

    # Quantization parameters
    QMAX = (1 << k_bits) - 1  # 2-bit: 3, 4-bit: 15
    QZERO = QMAX / 2  # 2-bit: 1.5, 4-bit: 7.5
    VALS_PER_BYTE = 8 // k_bits  # 2-bit: 4, 4-bit: 2
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    # Compute per-head per-dim scale (max absolute value across all tokens)
    k_abs_max = k.abs().amax(dim=1)  # [B, HKV, K]
    k_scale = k_abs_max / QZERO  # [B, HKV, K]
    k_scale = k_scale.clamp(min=eps)

    # Quantize: q = round(k / scale + QZERO), clamp to [0, QMAX]
    k_norm = k / k_scale.unsqueeze(1)  # [B, T, HKV, K]
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack values into uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, T, HKV, K_packed, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 4
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    # Compute dequantized values for residual
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(1)

    # Compute residual and convert to FP8
    k_residual = k - k_dequant
    try:
        k_residual = k_residual.to(torch.float8_e4m3fn)
    except (RuntimeError, TypeError):
        k_residual = k_residual.to(dtype)

    return k_q_packed, k_scale, k_residual


def quantize_symmetric_blocks(
    k_blocks: torch.Tensor,
    block_size: int,
    k_bits: int = 2,
    eps: float = 1e-8,
):
    """
    Quantize K across multiple full blocks.

    Args:
        k_blocks: [B, T, HKV, K], where T must be a multiple of block_size
        block_size: Length of each block
        k_bits: Quantization bits (2 or 4)
        eps: Small epsilon to avoid division by zero

    Returns:
        k_q: [B, T, HKV, K_packed]
        k_scale: [B, num_blocks, HKV, K] Per-block scale
        k_residual: [B, T, HKV, K]
    """
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, T, HKV, K = k_blocks.shape
    if T % block_size != 0:
        raise ValueError(f"T={T} is not divisible by block_size={block_size}")

    dtype = k_blocks.dtype
    num_blocks = T // block_size

    # Quantization parameters
    QMAX = (1 << k_bits) - 1  # 2-bit: 3, 4-bit: 15
    QZERO = QMAX / 2  # 2-bit: 1.5, 4-bit: 7.5
    VALS_PER_BYTE = 8 // k_bits  # 2-bit: 4, 4-bit: 2
    K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

    k_blocks = k_blocks.reshape(B, num_blocks, block_size, HKV, K)

    # Per-block scale (max absolute value within each block)
    k_abs_max = k_blocks.abs().amax(dim=2)  # [B, num_blocks, HKV, K]
    k_scale = (k_abs_max / QZERO).clamp(min=eps)

    # Quantize: q = round(k / scale + QZERO)
    k_norm = k_blocks / k_scale.unsqueeze(2)
    k_q_float = (k_norm + QZERO).round().clamp(0, QMAX)

    # Pack into uint8
    if K % VALS_PER_BYTE != 0:
        pad_size = VALS_PER_BYTE - (K % VALS_PER_BYTE)
        k_q_float = torch.nn.functional.pad(k_q_float, (0, pad_size), value=QZERO)

    k_q_int = k_q_float.to(torch.int32)
    k_q_int = k_q_int.view(B, num_blocks, block_size, HKV, K_packed, VALS_PER_BYTE)

    if k_bits == 2:
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 2) |
            (k_q_int[..., 2] << 4) |
            (k_q_int[..., 3] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 4
        k_q_packed = (
            k_q_int[..., 0] |
            (k_q_int[..., 1] << 4)
        ).to(torch.uint8)

    k_q_packed = k_q_packed.reshape(B, T, HKV, K_packed)

    # Dequantize for residual
    k_dequant = (k_q_float[..., :K] - QZERO) * k_scale.unsqueeze(2)
    k_residual = k_blocks - k_dequant
    try:
        k_residual = k_residual.to(torch.float8_e4m3fn)
    except (RuntimeError, TypeError):
        k_residual = k_residual.to(dtype)

    k_residual = k_residual.reshape(B, T, HKV, K)

    return k_q_packed, k_scale, k_residual


class QuantizedKVLayer(CacheLayerMixin):
    """
    Page-wise symmetric quantized Cache Layer.

    Features:
    - Each block/page is quantized independently with its own scale
    - New tokens accumulate in the current block; quantization happens when full
    - Incomplete blocks stay in FP16, unquantized
    - Pre-allocated fixed-size buffers to avoid address changes from torch.cat()
    """

    is_sliding = False

    def __init__(self, BS: int = 128, use_fp8_residual: bool = True, k_bits: int = 2, max_seq_len: int = 32768):
        super().__init__()
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.seq_dim = 1
        self.max_seq_len = max_seq_len
        self.max_blocks = (max_seq_len + BS - 1) // BS

        # Quantized full blocks — pre-allocated fixed size
        self.k_q: Optional[torch.Tensor] = None           # [B, max_blocks * BS, HKV, K_packed]
        self.k_scale: Optional[torch.Tensor] = None       # [B, max_blocks, HKV, K]
        self.k_residual: Optional[torch.Tensor] = None    # [B, max_blocks * BS, HKV, K]
        self.num_full_blocks: int = 0                     # Actual number of quantized blocks

        # Current incomplete block, kept in FP16 — fixed-size buffer + mask
        self.k_current: Optional[torch.Tensor] = None     # [B, BS, HKV, K] — fixed size
        self.v_current: Optional[torch.Tensor] = None     # [B, BS, HKV, V] — fixed size
        self.current_len: int = 0                         # Current valid length (0 <= current_len <= BS)

        # Full V cache — pre-allocated fixed size
        self.value: Optional[torch.Tensor] = None         # [B, max_seq_len, HKV, V]
        self.value_len: int = 0                           # Actual valid length of V cache

        # For compatibility with the transformers interface
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.key_full: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Pre-allocate all fixed-size buffers."""
        self.dtype, self.device = key_states.dtype, key_states.device
        B, _, HKV, K = key_states.shape
        V = value_states.shape[-1]

        # Compute packed K size
        VALS_PER_BYTE = 8 // self.k_bits
        K_packed = (K + VALS_PER_BYTE - 1) // VALS_PER_BYTE

        # Pre-allocate V cache — fixed size
        self.value = torch.zeros(
            (B, self.max_seq_len, HKV, V),
            dtype=self.dtype,
            device=self.device,
        )
        self.value_len = 0

        # Pre-allocate quantized K cache — fixed size
        max_quantized_len = self.max_blocks * self.BS
        self.k_q = torch.zeros(
            (B, max_quantized_len, HKV, K_packed),
            dtype=torch.uint8,
            device=self.device,
        )
        self.k_scale = torch.zeros(
            (B, self.max_blocks, HKV, K),
            dtype=torch.float32,
            device=self.device,
        )
        # FP8 residual
        try:
            self.k_residual = torch.zeros(
                (B, max_quantized_len, HKV, K),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
        except (RuntimeError, TypeError):
            self.k_residual = torch.zeros(
                (B, max_quantized_len, HKV, K),
                dtype=self.dtype,
                device=self.device,
            )
        self.num_full_blocks = 0

        # Pre-allocate k_current and v_current — fixed size BS
        self.k_current = torch.zeros(
            (B, self.BS, HKV, K),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_current = torch.zeros(
            (B, self.BS, HKV, V),
            dtype=self.dtype,
            device=self.device,
        )
        self.current_len = 0

        self.is_initialized = True

    def _quantize_and_store_block(self, k_block: torch.Tensor) -> None:
        """Quantize one full block and store it."""
        self._quantize_and_store_blocks(k_block)

    def _quantize_and_store_blocks(self, k_blocks: torch.Tensor) -> None:
        """Quantize multiple full blocks and store (in-place write to pre-allocated buffer)."""
        k_q_new, k_scale_new, k_residual_new = quantize_symmetric_blocks(
            k_blocks,
            block_size=self.BS,
            k_bits=self.k_bits,
        )

        num_new_blocks = k_scale_new.shape[1]
        new_tokens = num_new_blocks * self.BS

        # Check if pre-allocated space is exceeded
        if self.num_full_blocks + num_new_blocks > self.max_blocks:
            raise RuntimeError(
                f"Cache overflow: trying to store {self.num_full_blocks + num_new_blocks} blocks, "
                f"but max_blocks={self.max_blocks}. Increase max_seq_len (current: {self.max_seq_len})"
            )

        # In-place write to pre-allocated buffer (address unchanged)
        start_token = self.num_full_blocks * self.BS
        end_token = start_token + new_tokens
        start_block = self.num_full_blocks
        end_block = start_block + num_new_blocks

        self.k_q[:, start_token:end_token, :, :] = k_q_new
        self.k_scale[:, start_block:end_block, :, :] = k_scale_new

        # Handle FP8 residual
        if self.k_residual.dtype == torch.float8_e4m3fn and k_residual_new.dtype != torch.float8_e4m3fn:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new.to(torch.float8_e4m3fn)
        elif self.k_residual.dtype != torch.float8_e4m3fn and k_residual_new.dtype == torch.float8_e4m3fn:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new.to(self.k_residual.dtype)
        else:
            self.k_residual[:, start_token:end_token, :, :] = k_residual_new

        self.num_full_blocks += num_new_blocks

    def _refresh_fp_cache(self) -> None:
        """
        Refresh cache views.
        MODIFIED: No longer dequantize Key Cache; self.keys will be None.
        """
        self.key_full = None
        self.keys = None
        # V Cache retained for potential fallback
        self.values = self.value[:, :self.value_len, :, :] if self.value_len > 0 else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_len = key_states.shape[self.seq_dim]

        # Check if pre-allocated space is exceeded
        if self.value_len + new_len > self.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: trying to store {self.value_len + new_len} tokens, "
                f"but max_seq_len={self.max_seq_len}"
            )

        # Update V cache — in-place write
        self.value[:, self.value_len:self.value_len + new_len, :, :] = value_states
        self.value_len += new_len

        # Update K cache — in-place write to k_current
        # Write new keys into k_current
        
        # OPTIMIZATION: Use index_copy_ for single-token update if cache_position is provided
        # This enables CUDA Graph capture (address depends on tensor content, not fixed pointer)
        use_tensor_update = (
            new_len == 1 
            and self.current_len + new_len <= self.BS 
            and cache_kwargs is not None 
            and "cache_position" in cache_kwargs
        )

        if use_tensor_update:
             # Use tensor-based update
             pos = cache_kwargs["cache_position"][-1:] # [1]
             local_idx = pos % self.BS
             self.k_current.index_copy_(1, local_idx, key_states)
             self.v_current.index_copy_(1, local_idx, value_states)
             self.current_len += new_len
        elif self.current_len + new_len <= self.BS:
            # New tokens fit entirely in the current block
            self.k_current[:, self.current_len:self.current_len + new_len, :, :] = key_states
            self.v_current[:, self.current_len:self.current_len + new_len, :, :] = value_states
            self.current_len += new_len
        else:
            # Need to handle cross-block case
            # Fill the current block first
            remaining_in_block = self.BS - self.current_len
            if remaining_in_block > 0:
                self.k_current[:, self.current_len:self.BS, :, :] = key_states[:, :remaining_in_block, :, :]
                self.v_current[:, self.current_len:self.BS, :, :] = value_states[:, :remaining_in_block, :, :]

            # Quantize the now-full block
            self._quantize_and_store_blocks(self.k_current.contiguous())

            # Handle remaining tokens
            remaining_keys = key_states[:, remaining_in_block:, :, :]
            remaining_values = value_states[:, remaining_in_block:, :, :]
            remaining_len = remaining_keys.shape[self.seq_dim]

            # If remaining tokens exceed one block, continue quantizing
            while remaining_len >= self.BS:
                block_keys = remaining_keys[:, :self.BS, :, :]
                self._quantize_and_store_blocks(block_keys.contiguous())
                remaining_keys = remaining_keys[:, self.BS:, :, :]
                remaining_values = remaining_values[:, self.BS:, :, :]
                remaining_len = remaining_keys.shape[self.seq_dim]

            # Put remaining tokens into a new k_current
            self.k_current.zero_()
            self.v_current.zero_()
            if remaining_len > 0:
                self.k_current[:, :remaining_len, :, :] = remaining_keys
                self.v_current[:, :remaining_len, :, :] = remaining_values
            self.current_len = remaining_len

        # Check if current block is full
        if self.current_len == self.BS:
            self._quantize_and_store_blocks(self.k_current.contiguous())
            self.k_current.zero_()
            self.v_current.zero_()
            self.current_len = 0

        self._refresh_fp_cache()
        
        # MODIFIED RETURN:
        # To avoid OOM, we do not return full Keys/Values.
        # To allow .transpose() in llama.py to succeed (avoid AttributeError from None),
        # we return an empty tensor with shape=[B, 0, HKV, K].
        # 
        # Effect:
        # 1. 1. Sparse decode path: works correctly (reads internal k_q, not these key_states).
        # 2. 2. Flash Attention path: since Key/Value length is 0, FlashAttention
        #       will raise a RuntimeError (or CUDA error), providing an explicit failure.
        B, _, HKV, K = key_states.shape
        V = value_states.shape[-1]
        
        empty_k = torch.empty((B, 0, HKV, K), dtype=self.dtype, device=self.device)
        empty_v = torch.empty((B, 0, HKV, V), dtype=self.dtype, device=self.device)
        
        return empty_k, empty_v

    def get_seq_length(self) -> int:
        quantized_len = self.num_full_blocks * self.BS
        return quantized_len + self.current_len

    def get_quantized_len(self) -> int:
        return self.num_full_blocks * self.BS

    def get_current_len(self) -> int:
        return self.current_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def reset(self) -> None:
        if not self.is_initialized:
            return
        # Reset lengths but keep pre-allocated buffers (address unchanged)
        self.num_full_blocks = 0
        self.current_len = 0
        self.value_len = 0
        if self.k_q is not None:
            self.k_q.zero_()
        if self.k_scale is not None:
            self.k_scale.zero_()
        if self.k_residual is not None:
            self.k_residual.zero_()
        if self.k_current is not None:
            self.k_current.zero_()
        if self.v_current is not None:
            self.v_current.zero_()
        if self.value is not None:
            self.value.zero_()
        self.key_full = None
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        # Note: this changes batch size and requires re-allocating buffers
        if self.k_q is not None:
            self.k_q = self.k_q.repeat_interleave(repeats, dim=0)
            self.k_scale = self.k_scale.repeat_interleave(repeats, dim=0)
            self.k_residual = self.k_residual.repeat_interleave(repeats, dim=0)
        if self.k_current is not None:
            self.k_current = self.k_current.repeat_interleave(repeats, dim=0)
        if self.v_current is not None:
            self.v_current = self.v_current.repeat_interleave(repeats, dim=0)
        if self.value is not None:
            self.value = self.value.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.device)
        # Note: this changes batch size and requires re-allocating buffers
        if self.k_q is not None:
            self.k_q = self.k_q.index_select(0, indices)
            self.k_scale = self.k_scale.index_select(0, indices)
            self.k_residual = self.k_residual.index_select(0, indices)
        if self.k_current is not None:
            self.k_current = self.k_current.index_select(0, indices)
        if self.v_current is not None:
            self.v_current = self.v_current.index_select(0, indices)
        if self.value is not None:
            self.value = self.value.index_select(0, indices)
        self._refresh_fp_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class QuantizedKVCache(Cache):
    """
    Page-wise symmetric quantized Cache.

    Features:
    - Each block/page is quantized independently with its own scale
    - New tokens accumulate in the current block; quantization happens when full
    - Incomplete blocks stay in FP16, unquantized
    - Pre-allocated fixed-size buffers to avoid address changes (CUDA Graph compatible)
    """

    def __init__(
        self,
        BS: int = 128,
        use_fp8_residual: bool = True,
        k_bits: int = 2,
        max_seq_len: int = 32768,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"k_bits must be one of {SUPPORTED_K_BITS}, got {k_bits}")
        self.BS = BS
        self.use_fp8_residual = use_fp8_residual
        self.k_bits = k_bits
        self.max_seq_len = max_seq_len

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                QuantizedKVLayer(
                    BS=self.BS,
                    use_fp8_residual=self.use_fp8_residual,
                    k_bits=self.k_bits,
                    max_seq_len=self.max_seq_len,
                )
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)

        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def get_seq_length(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_seq_length()

    def get_quantized_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_quantized_len()

    def get_current_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].get_current_len()
