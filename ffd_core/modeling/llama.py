# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Name: transformers
# Version: 4.57.3
# Modified for FFD sparse decode attention

from typing import Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig

try:
    from .quantized_cache import QuantizedKVCache
    from .selection_thresholds import (
        build_q2_selector_threshold,
        normalize_selector_type,
        resolve_selector_value,
    )
except ImportError:
    from quantized_cache import QuantizedKVCache
    from selection_thresholds import (
        build_q2_selector_threshold,
        normalize_selector_type,
        resolve_selector_value,
    )

try:
    from flash_attn import flash_attn_func
except ImportError:
    def flash_attn_func(query, key, value, causal=True, **kwargs):
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        return out.transpose(1, 2)

# Alias for compatibility
Q2FP8SymCache = QuantizedKVCache
Q2FP8SymStaticCache = QuantizedKVCache

logger = logging.get_logger(__name__)


def _bump_stat(stats: dict, key: str, inc: int = 1) -> None:
    stats[key] = stats.get(key, 0) + inc


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # CUDA Graph state for Block-wise JIT capture
        self.current_graph_runner = None
        self.cached_num_blocks = -1
        self.cached_k_q_ptr = -1

        # Pre-allocated merge buffer for CUDA Graph (fix for output aliasing)
        # Will be lazily initialized on first use
        self._merge_output_buffer = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # Make a shallow copy so we can tweak defaults without mutating the config.
        attn_settings: dict = dict(getattr(self.config, "attn_settings", {}))
        use_sparse_decode = attn_settings.get(
            "use_ffd_decode",
            attn_settings.get("use_sparse_decode", False),
        )
        # Check if full graph mode is enabled (global capture)
        use_full_graph = attn_settings.get("use_full_graph", False)
        debug_stats = attn_settings.get("debug_stats")

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        q_len = query_states.shape[-2]

        # Check if we're using the quantized cache implementation
        is_quantized_cache = isinstance(
            past_key_values, (Q2FP8SymCache, Q2FP8SymStaticCache)
        )

        cache_layer = None
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # === FFD modification ===
            if is_quantized_cache:
                # Backup the current inputs (Rotated Keys/Values)
                # Because update() now returns empty tensors, we need this backup for the Prefill phase
                key_states_input = key_states
                value_states_input = value_states

                # Q2FP8SymCache update expects [B, T, HKV, K] layout
                key_states_cache = key_states.transpose(1, 2)
                value_states_cache = value_states.transpose(1, 2)

                # Call Cache Update (returns empty tensors!)
                key_states_ret, value_states_ret = past_key_values.update(
                    key_states_cache, value_states_cache, self.layer_idx, cache_kwargs
                )

                # Check if update returned empty tensors (length 0)
                if key_states_ret.shape[1] == 0:
                    if q_len > 1:
                        # Case 1: Prefill phase
                        # Cache stored the data but returned empty. Use current inputs for self-attention.
                        # Note: assumes first-turn prefill. For prefill with history, since we don't
                        # dequantize history, FlashAttn only attends to current input — consistent
                        # with the no-dequantization constraint.
                        key_states = key_states_input
                        value_states = value_states_input
                    else:
                        # Case 2: Decode phase
                        # Keep empty. Sparse decode path reads cache internals directly.
                        # Flash path with empty K/V raises RuntimeError (by design).
                        key_states = key_states_ret.transpose(1, 2)
                        value_states = value_states_ret.transpose(1, 2)
                else:
                    # Compatibility: update returned full data (legacy behavior)
                    key_states = key_states_ret.transpose(1, 2)
                    value_states = value_states_ret.transpose(1, 2)

                cache_layer = past_key_values.layers[self.layer_idx]
            # === End FFD modification ===
            else:
                # Use standard cache interface (DynamicCache, etc.)
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        attn_weights = None

        from .decode import attn_forward_decode, CUDAGraphDecodeRunnerQ2FP8

        pattern_layers = attn_settings.get("pattern_layers", None)
        if pattern_layers is None:
            pattern_layers = list(range(1000))
        assert type(pattern_layers) is list

        if debug_stats is not None:
            _bump_stat(debug_stats, "total_calls")
            if q_len == 1:
                _bump_stat(debug_stats, "decode_calls")
            else:
                _bump_stat(debug_stats, "prefill_calls")

        # Check if we have quantized blocks available for FFD decode
        has_quantized_blocks = False
        if is_quantized_cache:
            cache_layer = cache_layer or past_key_values.layers[self.layer_idx]
            has_quantized_blocks = (
                cache_layer.k_q is not None and cache_layer.k_scale is not None
            )

        # Use sparse decode when: decode mode, sparse decode enabled, has quantized blocks, in pattern layers

        use_sparse_path = (
            q_len == 1
            and use_sparse_decode
            and is_quantized_cache
            and has_quantized_blocks
            and self.layer_idx in pattern_layers
        )

        # Check if we have unquantized tokens in k_current
        current_len = 0
        k_current = None
        v_current = None

        if is_quantized_cache and cache_layer is not None:
            if use_full_graph:
                # [Fix for Full-Chain CUDA Graph]
                # 1. Use Tensor for current_len to support dynamic update inside graph
                bs = getattr(cache_layer, "BS", 128)
                if cache_position is not None:
                    # cache_position is [1] during decode usually.
                    # We need +1 because current_len represents the COUNT of valid tokens (1-based),
                    # whereas cache_position is 0-based index.
                    # After cache.update(), the current token is at index (cache_position % BS),
                    # so valid count is index + 1.
                    # We also need to modulo BS again, because if it reaches BS (e.g. 128),
                    # the cache resets current_len to 0 and moves data to quantized buffer.
                    current_len = (((cache_position[-1] % bs) + 1) % bs).to(torch.int32)
                else:
                    # Fallback (should not happen in correct usage)
                    current_len = torch.tensor(
                        cache_layer.get_current_len(),
                        dtype=torch.int32,
                        device=query_states.device,
                    )

                # 2. Always bind k_current/v_current buffers.
                # The kernel handles current_len=0 correctly (masking), so it's safe to pass them.
                # This ensures the graph captures the pointers.
                k_current = cache_layer.k_current
                if (
                    hasattr(cache_layer, "v_current")
                    and cache_layer.v_current is not None
                ):
                    v_current = cache_layer.v_current
                else:
                    # Fallback if v_current not explicit
                    quantized_len = cache_layer.get_quantized_len()
                    v_current = cache_layer.value[:, quantized_len:, :, :]
            else:
                # [Original Eager / Block-wise Graph Logic]
                current_len = cache_layer.get_current_len()
                # Prepare pointers for fused kernel
                if current_len > 0:
                    k_current = cache_layer.k_current
                    if (
                        hasattr(cache_layer, "v_current")
                        and cache_layer.v_current is not None
                    ):
                        v_current = cache_layer.v_current
                    else:
                        quantized_len = cache_layer.get_quantized_len()
                        v_current = cache_layer.value[:, quantized_len:, :, :]

        # Use sparse decode when: decode mode, sparse decode enabled, has quantized blocks, in pattern layers

        use_sparse_path = (
            q_len == 1
            and use_sparse_decode
            and is_quantized_cache
            and has_quantized_blocks
            and self.layer_idx in pattern_layers
        )

        if debug_stats is not None and q_len == 1:
            if not use_sparse_decode:
                _bump_stat(debug_stats, "sparse_decode_disabled")
            if not is_quantized_cache:
                _bump_stat(debug_stats, "not_quantized_cache")
            if not has_quantized_blocks:
                _bump_stat(debug_stats, "no_quantized_blocks")
            if self.layer_idx not in pattern_layers:
                _bump_stat(debug_stats, "not_in_pattern_layers")
            if use_sparse_path:
                _bump_stat(debug_stats, "sparse_path_calls")
            else:
                _bump_stat(debug_stats, "flash_path_decode_calls")

        if use_sparse_path:
            decode_kwargs = {
                k: v
                for k, v in attn_settings.items()
                if k
                not in (
                    "use_sparse_prefill",
                    "use_sparse_decode",
                    "use_ffd_prefill",
                    "use_ffd_decode",
                    "pattern_layers",
                    "debug_stats",
                )
            }

            # OPTIMIZATION: Use full buffers and pass seq_len to avoid slicing overhead
            quantized_len = cache_layer.get_quantized_len()
            k_q = cache_layer.k_q
            k_residual = cache_layer.k_residual
            num_full_blocks = cache_layer.num_full_blocks
            k_scale = cache_layer.k_scale
            v_quantized = cache_layer.value

            skip_ratio_store = attn_settings.get("skip_ratio_store")

            # FFD decode expects q: [B, 1, HQ, K]
            # Ensure q is contiguous for the kernel (low cost for 1 token)
            q_for_sparse = query_states.transpose(
                1, 2
            ).contiguous()  # [B, HQ, 1, K] -> [B, 1, HQ, K]

            # Add seq_len to decode_kwargs
            decode_kwargs["seq_len"] = quantized_len
            selector_type = normalize_selector_type(
                decode_kwargs.get("selector_type")
            )
            selector_value = resolve_selector_value(
                selector_type,
                decode_kwargs.get("selector_value"),
                delta=float(decode_kwargs.get("delta", 5.0)),
            )
            decode_kwargs["selector_type"] = selector_type
            decode_kwargs["selector_value"] = selector_value
            if selector_type == "top_k":
                decode_kwargs["precomputed_threshold"] = build_q2_selector_threshold(
                    q=q_for_sparse,
                    k_q=k_q,
                    k_scale=k_scale,
                    block_size=int(decode_kwargs.get("BS", 128)),
                    selector_type=selector_type,
                    selector_value=selector_value,
                    k_bits=int(decode_kwargs.get("k_bits", 2)),
                )

            # When we have k_current, fused kernel handles merging internally.
            # We don't need external LSE.
            need_lse = False
            return_skip = decode_kwargs.get("return_skip_ratio", False)

            # Block-wise JIT CUDA Graph Logic

            num_full_blocks = (
                cache_layer.num_full_blocks
                if hasattr(cache_layer, "num_full_blocks")
                else 0
            )

            if debug_stats is not None:
                if current_len > 0:
                    _bump_stat(debug_stats, "sparse_current_len_nonzero")
                else:
                    _bump_stat(debug_stats, "sparse_current_len_zero")
                if need_lse:
                    _bump_stat(debug_stats, "sparse_need_lse")
                if num_full_blocks > 0:
                    _bump_stat(debug_stats, "sparse_has_full_blocks")
                else:
                    _bump_stat(debug_stats, "sparse_no_full_blocks")

            # Only use CUDA Graph if we have full blocks
            if num_full_blocks > 0 and not use_full_graph:
                # Check if we need to recapture the graph
                k_q_ptr = k_q.data_ptr()
                need_recapture = (
                    self.current_graph_runner is None
                    or num_full_blocks != self.cached_num_blocks
                    or k_q_ptr != self.cached_k_q_ptr
                )

                if debug_stats is not None and need_recapture:
                    _bump_stat(debug_stats, "sparse_graph_recapture")

                if need_recapture:
                    # Free old graph runner to release memory
                    self.current_graph_runner = None

                    # Prepare kwargs for CUDA Graph runner (exclude return_skip_ratio, return_lse, max_decode_tokens)
                    graph_kwargs = {
                        k: v
                        for k, v in decode_kwargs.items()
                        if k
                        not in (
                            "return_skip_ratio",
                            "return_lse",
                            "max_decode_tokens",
                            "skip_ratio_store",
                        )
                    }

                    # Force use_fp8_residual=True if config enables it
                    use_fp8_residual = graph_kwargs.get("use_fp8_residual", True)
                    graph_kwargs["use_fp8_residual"] = use_fp8_residual

                    # Instantiate new CUDA Graph runner with warmup=2
                    self.current_graph_runner = CUDAGraphDecodeRunnerQ2FP8(
                        q=q_for_sparse,
                        k_q=k_q,
                        k_scale=k_scale,
                        v=v_quantized,
                        k_new=cache_layer.k_current,
                        v_new=cache_layer.v_current
                        if hasattr(cache_layer, "v_current")
                        else None,
                        k_residual=k_residual,
                        warmup=2,
                        **graph_kwargs,
                    )

                    # Update cached state
                    self.cached_num_blocks = num_full_blocks
                    self.cached_k_q_ptr = k_q_ptr

                # Replay the graph with LSE support
                # If we need_lse, it usually implies we also want to run the merge kernel inside the graph
                # The Runner now handles merge internally if k_current/v_current were provided
                attn_output_sparse = self.current_graph_runner.replay(
                    q=q_for_sparse,
                    k_q=k_q,
                    k_scale=k_scale,
                    v=v_quantized,
                    k_new=cache_layer.k_current,
                    v_new=cache_layer.v_current,
                    k_residual=k_residual,
                    return_lse=False,  # We don't need m/l back if merge is done internally
                    current_len=current_len,
                )

                if debug_stats is not None:
                    _bump_stat(debug_stats, "sparse_graph_replay")

                # Handle skip_ratio if needed (outside graph)
                if return_skip:
                    if debug_stats is not None:
                        _bump_stat(debug_stats, "sparse_skip_ratio_requested")
                    # Compute skip_ratio separately (not in graph)
                    # decode_kwargs might already contain return_skip_ratio, so we update it
                    ratio_kwargs = decode_kwargs.copy()
                    ratio_kwargs["return_skip_ratio"] = True

                    _, skip_ratio = attn_forward_decode(
                        q=q_for_sparse,
                        k_q=k_q,
                        k_scale=k_scale,
                        v=v_quantized,
                        k_residual=k_residual,
                        return_lse=False,
                        **ratio_kwargs,
                    )
                    if isinstance(skip_ratio_store, list):
                        skip_ratio_store.append(float(skip_ratio))
            else:
                # No full blocks yet, use regular path
                if debug_stats is not None:
                    _bump_stat(debug_stats, "sparse_no_graph")
                decode_result = attn_forward_decode(
                    q=q_for_sparse,
                    k_q=k_q,
                    k_scale=k_scale,
                    v=v_quantized,
                    k_residual=k_residual,
                    return_lse=need_lse,
                    k_current=k_current,
                    v_current=v_current,
                    current_len=current_len,
                    **decode_kwargs,
                )
                # Parse results based on what was requested

                if need_lse:
                    if return_skip:
                        attn_output_sparse, sparse_m, sparse_l, skip_ratio = decode_result
                        if isinstance(skip_ratio_store, list):
                            skip_ratio_store.append(float(skip_ratio))
                    else:
                        attn_output_sparse, sparse_m, sparse_l = decode_result
                else:
                    if return_skip:
                        attn_output_sparse, skip_ratio = decode_result
                        if isinstance(skip_ratio_store, list):
                            skip_ratio_store.append(float(skip_ratio))
                    else:
                        attn_output_sparse = decode_result

            # Sparse decode kernel returns [B, HQ, V], reshape to [B, 1, HQ, V]
            # attn_output_sparse = attn_output_sparse.unsqueeze(1) # [B, HQ, V] -> [B, 1, HQ, V]

            # Fused kernel returns already merged output (if k_current was present)
            # Just ensure shape is correct
            if attn_output_sparse.ndim == 3:
                attn_output = attn_output_sparse.unsqueeze(1)
            else:
                attn_output = attn_output_sparse

        else:
            # Use flash_attn for non-FFD path

            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=self.is_causal,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Skip mask creation for full graph decode
        if (
            inputs_embeds.shape[1] == 1
            and hasattr(self.config, "attn_settings")
            and self.config.attn_settings.get("use_full_graph")
        ):
            causal_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class CUDAGraphFullModelRunner:
    def __init__(self, model):
        self.model = model
        self.graph = None
        self.mempool = None
        self.static_input_ids = None
        self.static_position_ids = None
        self.static_cache_position = None
        self.static_output = None
        self.cached_num_blocks = -1

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position=None,
        **kwargs,
    ):
        # We only support input_ids based decoding for now
        assert input_ids is not None

        # Determine if we should run eager or graph
        # 1. Check for block boundary (Quantization step) -> Eager
        past_len = past_key_values.get_seq_length()
        bs = past_key_values.BS if hasattr(past_key_values, "BS") else 128

        # If this step will fill the block, it triggers quantization (complex python logic)
        # So we run eagerly.
        # current_len runs 0..BS-1.
        # Quantization happens when adding to a block that has BS-1 items.
        # past_len is total tokens. past_len % BS is current_len.
        if (past_len + 1) % bs == 0:
            return self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # 2. Check for graph validity (num_blocks changed?)
        current_num_blocks = 0
        if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
            current_num_blocks = past_key_values.layers[0].num_full_blocks

        if self.graph is None or current_num_blocks != self.cached_num_blocks:
            self.capture(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                **kwargs,
            )
            self.cached_num_blocks = current_num_blocks

        # Copy inputs to static buffers
        self.static_input_ids.copy_(input_ids)
        self.static_position_ids.copy_(position_ids)
        if cache_position is not None:
            self.static_cache_position.copy_(cache_position)

        # Replay graph
        self.graph.replay()

        return self.static_output

    def capture(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position=None,
        **kwargs,
    ):
        logger.info(
            f"Capturing CUDA Graph for num_blocks={past_key_values.layers[0].num_full_blocks if hasattr(past_key_values, 'layers') else 0}..."
        )

        # Create static tensors
        self.static_input_ids = input_ids.clone()
        self.static_position_ids = position_ids.clone()
        self.static_cache_position = (
            cache_position.clone() if cache_position is not None else None
        )

        # Prepare graph capture
        self.graph = torch.cuda.CUDAGraph()
        self.mempool = (
            torch.cuda.graph_pool_handle() if self.mempool is None else self.mempool
        )

        # Warmup — advance cache state, then restore current_len so the real
        # run overwrites the warmup data. Since we skip block-boundary steps and
        # re-capture only when num_blocks changes, the graph topology is stable.
        layer = past_key_values.layers[0]
        backup_current_len = layer.current_len

        # Run model (Warmup)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.model.forward(
                input_ids=self.static_input_ids,
                attention_mask=attention_mask,
                position_ids=self.static_position_ids,
                past_key_values=past_key_values,
                cache_position=self.static_cache_position,
                **kwargs,
            )
        torch.cuda.current_stream().wait_stream(stream)

        # Restore current_len across all layers so the real run overwrites warmup data
        for l in past_key_values.layers:
            l.current_len = backup_current_len

        # Capture
        with torch.cuda.graph(self.graph, pool=self.mempool):
            self.static_output = self.model.forward(
                input_ids=self.static_input_ids,
                attention_mask=attention_mask,
                position_ids=self.static_position_ids,
                past_key_values=past_key_values,  # passed by reference, writes to it are captured
                cache_position=self.static_cache_position,
                **kwargs,
            )

        # Restore cache state again (because capture run advanced it)
        for l in past_key_values.layers:
            l.current_len = backup_current_len

        logger.info("CUDA Graph capture complete.")


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.full_graph_runner = None

        # Initialize weights and apply final processing
        self.post_init()

    def enable_full_cudagraph(self):
        """Enables Full-Chain CUDA Graph capture for decoding."""
        logger.info("Enabling Full-Chain CUDA Graph optimization.")
        self.full_graph_runner = CUDAGraphFullModelRunner(self)
        # Enable flag in config so Attention layers know to skip internal graph
        if not hasattr(self.config, "attn_settings"):
            self.config.attn_settings = {}
        # Ensure it's a dict we can modify
        if isinstance(self.config.attn_settings, tuple):  # Handle tuple case if any
            self.config.attn_settings = {}
        self.config.attn_settings["use_full_graph"] = True

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        # Check for full graph execution
        if (
            getattr(self, "full_graph_runner", None) is not None
            and use_cache
            and past_key_values is not None
            and input_ids is not None
            and input_ids.shape[1] == 1
        ):
            # Ensure cache_position and position_ids are present
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + input_ids.shape[1],
                    device=input_ids.device,
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            return self.full_graph_runner.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(
    GenericForSequenceClassification, LlamaPreTrainedModel
): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


class LlamaForTokenClassification(
    GenericForTokenClassification, LlamaPreTrainedModel
): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
