# FFD: Faster Flash Decoding

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**Faster Flash Decoding (FFD)** is a training-free, plug-and-play sparse attention framework for efficient long-context LLM decoding. It combines 2-bit quantized key scanning with a top-δ selection strategy to break the memory wall in the autoregressive generation phase.

> Paper: *Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding*

## Key Features

- **2-bit Content-Aware Scanning**: Replace metadata-based indexing with compressed KV-cache scanning, eliminating auxiliary memory overhead.
- **Top-δ Selection**: Distribution-adaptive sparsity without the synchronization cost of top-p.
- **Fused Triton Kernel**: Selector-computer integration with CUDA Graph capture for minimal launch overhead.
- **Drop-in Integration**: Compatible with HuggingFace `transformers` Llama models — no retraining required.

## Hardware Requirements

- NVIDIA GPU with CUDA ≥ 12.8
- Triton ≥ 3.4
- Tested on: RTX 4090, H100

## Installation

```bash
conda create -n ffd python=3.10 -y
conda activate ffd
pip install -e .
```

For FlashAttention acceleration during prefill:

```bash
pip install flash-attn==2.8.3
```

## Quick Start

```python
import torch
from transformers import AutoConfig, AutoTokenizer
from ffd_core import LlamaForCausalLM, QuantizedKVCache

device = torch.device("cuda")
model_path = "meta-llama/Llama-3.1-8B-Instruct"

# Configure FFD
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.attn_settings = {
    "use_sparse_decode": True,   # Enable FFD sparse decode
    "delta": 5.0,                # Sparsity threshold (5.0 = aggressive, 7.0 = conservative)
    "k_bits": 2,                 # Quantization bits for key cache
    "use_fp8_residual": True,    # Use FP8 residual for refinement
    "BS": 128,                   # Block size for quantization
    "use_full_graph": False,     # Set True for full-chain CUDA Graph
}

model = LlamaForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer("Your long context prompt...", return_tensors="pt").to(device)

# Use QuantizedKVCache for sparse decode
cache = QuantizedKVCache(BS=128, use_fp8_residual=True, k_bits=2, max_seq_len=32768)

with torch.no_grad():
    output = model.generate(
        **inputs,
        past_key_values=cache,
        max_new_tokens=256,
        do_sample=False,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | float | 5.0 | Sparsity threshold. δ=5 prunes weights < 1/148 of max (aggressive). δ=7 prunes weights < 1/1096 (conservative). |
| `k_bits` | int | 2 | Key quantization bit-width. Currently 2-bit is recommended. |
| `use_fp8_residual` | bool | True | Use FP8 residual for high-precision refinement of kept blocks. |
| `BS` | int | 128 | Quantization block size. Use 128 for consumer GPUs, 256 for H100. |
| `use_sparse_decode` | bool | False | Enable FFD sparse decode path. |
| `use_full_graph` | bool | False | Enable full-chain CUDA Graph capture for maximum throughput. |

## Evaluation (RULER / LongBench)

FFD's downstream accuracy is evaluated with the [OpenCompass](https://github.com/open-compass/opencompass) framework. This repo includes ready-to-use eval configs and a model wrapper:

```
opencompass/
  oc_model.py              # OpenCompass model wrapper (imports ffd-core)
  eval_ruler_llama.py      # RULER (32k) eval config
  eval_longbench_llama.py  # LongBench eval config
```

**Setup:**

1. Install OpenCompass following its [official instructions](https://opencompass.readthedocs.io/).
2. Copy `opencompass/oc_model.py` into your OpenCompass fork at `opencompass/models/myModel/ffd/oc_model.py`.
3. Edit the eval config to set `MODEL_PATH` to your local checkpoint.
4. Run:
```bash
cd /path/to/opencompass
opencompass ffd-core/opencompass/eval_ruler_llama.py -w output/ruler
opencompass ffd-core/opencompass/eval_longbench_llama.py -w output/longbench
```

## Validate

```bash
# Kernel microbenchmark
python scripts/benchmark_kernel.py --seq-len 16384

# Kernel correctness test (compare against FlashAttention)
python scripts/test_kernel_correctness.py

# Generation smoke test (requires a local Llama checkpoint)
python scripts/verify_generate.py --model /path/to/Llama-3.1-8B-Instruct --prompt "Hello"
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
