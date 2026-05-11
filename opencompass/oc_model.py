"""
OpenCompass model wrapper for FFD.

Usage:
    Copy this file into your OpenCompass fork under
    ``opencompass/models/myModel/ffd/oc_model.py``, then reference it from
    eval configs as ``type=HF_ForCausalLM_FFD_OC``.
"""

from typing import List, Optional

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFaceBaseModel as HuggingFaceCausalLM,
)
from opencompass.registry import MODELS
from transformers import AutoConfig

from ffd_core import LlamaForCausalLM, QuantizedKVCache


@MODELS.register_module()
class HF_ForCausalLM_FFD_OC(HuggingFaceCausalLM):
    """FFD sparse-decode model for OpenCompass evaluation.

    Wires up ffd-core's LlamaForCausalLM + QuantizedKVCache so that
    RULER / LongBench evaluation runs with FFD sparse decode enabled.

    Supports Llama-architecture models. For Qwen2/3, extend this class
    with the corresponding ffd_core model class once available.
    """

    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        model_kwargs = kwargs

        if "max_seq_len" in model_kwargs:
            self.max_seq_len = model_kwargs.pop("max_seq_len")
        elif not hasattr(self, "max_seq_len") or self.max_seq_len is None:
            self.max_seq_len = 32768

        if "cache_max_seq_len" in model_kwargs:
            self.cache_max_seq_len = model_kwargs.pop("cache_max_seq_len")
        elif not hasattr(self, "cache_max_seq_len") or self.cache_max_seq_len is None:
            self.cache_max_seq_len = max(self.max_seq_len, 32768)

        attn_defaults = dict(
            use_sparse_decode=True,
            delta=5.0,
            pattern_layers=None,
            BS=128,
            SBS=None,
            return_skip_ratio=False,
            use_fp8_residual=True,
            use_full_graph=False,
            k_bits=2,
            skip_ratio_store=None,
        )

        config_attn_settings = {
            key: model_kwargs.pop(key, default)
            for key, default in attn_defaults.items()
        }
        config_attn_settings = {
            k: v for k, v in config_attn_settings.items() if v is not None
        }

        trust_remote_code = model_kwargs.get("trust_remote_code", False)
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", None)
        if model_type not in ("llama", "qwen2", "qwen3"):
            raise ValueError(
                f"FFD OC wrapper only supports llama/qwen models, "
                f"got model_type={model_type} for path={path}"
            )
        config.attn_settings = config_attn_settings
        self.config_attn_settings = config_attn_settings

        # ffd-core currently provides LlamaForCausalLM.
        # Qwen2/3 models can be added to ffd-core following the same pattern.
        if model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=path, config=config, **model_kwargs
            )
        elif model_type in ("qwen2", "qwen3"):
            raise NotImplementedError(
                "Qwen support in ffd-core is not yet available. "
                "Use the Llama model path or add a QwenForCausalLM to ffd-core."
            )

        try:
            from peft import PeftModel
            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)
        except ImportError:
            if peft_path is not None:
                raise ImportError("peft not installed but peft_path provided")

        if self.config_attn_settings.get("use_full_graph", False):
            if hasattr(self.model, "enable_full_cudagraph"):
                self.model.enable_full_cudagraph()
            else:
                print("[WARNING] use_full_graph=True but enable_full_cudagraph() not found.")

        self.model.eval()
        self.model.generation_config.do_sample = False

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        BS = self.config_attn_settings.get("BS", 128)
        use_fp8_residual = self.config_attn_settings.get("use_fp8_residual", True)
        k_bits = self.config_attn_settings.get("k_bits", 2)

        input_max_seq_len = getattr(self, "max_seq_len", 32768)
        cache_max_seq_len = getattr(self, "cache_max_seq_len", input_max_seq_len)
        max_out_len = kwargs.get("max_out_len")
        max_out_len = int(max_out_len) if max_out_len is not None else 0
        required_cache_len = input_max_seq_len + max_out_len
        if cache_max_seq_len < required_cache_len:
            cache_max_seq_len = required_cache_len

        fresh_cache = QuantizedKVCache(
            BS=BS,
            use_fp8_residual=use_fp8_residual,
            k_bits=k_bits,
            max_seq_len=cache_max_seq_len,
        )

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs["past_key_values"] = fresh_cache

        old_gen_kwargs = self.generation_kwargs
        self.generation_kwargs = gen_kwargs
        try:
            return super().generate(inputs, **kwargs)
        finally:
            self.generation_kwargs = old_gen_kwargs
            # Reset CUDA Graph runner between samples
            if hasattr(self.model, "full_graph_runner"):
                self.model.full_graph_runner = None
