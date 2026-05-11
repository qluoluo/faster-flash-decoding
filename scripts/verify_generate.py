from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffd_core import LlamaForCausalLM, QuantizedKVCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal generation smoke test for ffd-core.")
    p.add_argument("--model", required=True, help="Local HF model path")
    p.add_argument("--prompt", default="Hello, how are you today?")
    p.add_argument("--max-new-tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verify_generate.py")

    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.attn_settings = {
        "use_sparse_decode": True,
        "delta": 5.0,
        "k_bits": 2,
        "use_fp8_residual": True,
        "BS": 128,
        "use_full_graph": False,
    }
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    cache = QuantizedKVCache(BS=128, use_fp8_residual=True, k_bits=2)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
