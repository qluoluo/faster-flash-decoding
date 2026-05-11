"""End-to-end decode throughput benchmark for FFD.

Measures tokens/sec during the decode phase across context lengths.
Requires a local Llama checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffd_core import LlamaForCausalLM, QuantizedKVCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark end-to-end decode throughput.")
    p.add_argument("--model", required=True, help="Local HF model path")
    p.add_argument("--prefill-len", type=int, default=4096, help="Prefill context length")
    p.add_argument("--decode-steps", type=int, default=128, help="Number of decode steps")
    p.add_argument("--delta", type=float, default=5.0, help="Sparsity threshold")
    p.add_argument("--bs", type=int, default=128, help="Block size")
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    p.add_argument("--no-ffd", action="store_true", help="Run baseline FlashAttention decode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"Loading model from {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    if not args.no_ffd:
        config.attn_settings = {
            "use_sparse_decode": True,
            "delta": args.delta,
            "k_bits": 2,
            "use_fp8_residual": True,
            "BS": args.bs,
            "use_full_graph": False,
        }

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build a prefill prompt of the desired length
    prefill_text = "The quick brown fox jumps over the lazy dog. " * ((args.prefill_len // 8) + 1)
    inputs = tokenizer(prefill_text, return_tensors="pt", truncation=True, max_length=args.prefill_len)
    input_ids = inputs.input_ids.to(device)
    actual_len = input_ids.shape[1]
    print(f"Prefill length: {actual_len} tokens")

    cache = QuantizedKVCache(
        BS=args.bs,
        use_fp8_residual=not args.no_ffd,
        k_bits=2,
        max_seq_len=actual_len + args.decode_steps + 1024,
    )

    mode = "FFD sparse decode" if not args.no_ffd else "FlashAttention dense decode"
    print(f"Mode: {mode}")
    print(f"Warmup: 2 prefill + 16 decode steps...")

    # Warmup
    with torch.no_grad():
        # Prefill warmup
        _ = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
    cache.reset()
    torch.cuda.synchronize(device)

    with torch.no_grad():
        # Prefill (timed)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize(device)
        prefill_time = time.perf_counter() - t0

        # Decode (timed)
        decode_latencies = []
        for _ in range(args.decode_steps):
            dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model.generate(
                input_ids=dummy_input,
                past_key_values=cache,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize(device)
            decode_latencies.append(time.perf_counter() - t0)

    avg_decode_ms = sum(decode_latencies) / len(decode_latencies) * 1000
    throughput = 1.0 / (sum(decode_latencies) / len(decode_latencies))

    print(f"\nResults ({mode}):")
    print(f"  Prefill time: {prefill_time:.3f}s")
    print(f"  Decode steps: {args.decode_steps}")
    print(f"  Avg decode latency: {avg_decode_ms:.2f} ms/token")
    print(f"  Throughput: {throughput:.1f} tokens/s")


if __name__ == "__main__":
    main()
