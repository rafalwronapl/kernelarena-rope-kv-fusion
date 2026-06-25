from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmark_rope_real_vllm_contract import DTYPES, build_cases, build_contract, build_cos_sin, cuda_time_us, rope_ref


REPO_ROOT = Path(__file__).resolve().parent


def try_vllm_rotary(head_dim: int, max_position: int, dtype: torch.dtype):
    try:
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        emb = RotaryEmbedding(head_dim, head_dim, max_position, 10000, False, dtype).cuda()
        return emb, None
    except Exception as exc:
        return None, repr(exc)


def run_case(case, warmup: int, repeats: int) -> dict[str, object]:
    dtype = DTYPES[case.dtype_name]
    contract = build_contract(case)
    total_tokens = int(contract["total_tokens"])
    max_position = int(contract["max_position"])
    positions = torch.tensor(contract["positions"], device="cuda", dtype=torch.long)
    k = torch.randn(total_tokens, case.kv_heads, case.head_dim, device="cuda", dtype=dtype)
    cos, sin = build_cos_sin(positions, case.head_dim, dtype)

    def local_rope():
        return rope_ref(k, cos, sin)

    compiled_rope = torch.compile(rope_ref, mode="max-autotune-no-cudagraphs")
    _ = compiled_rope(k, cos, sin)
    torch.cuda.synchronize()

    def compiled():
        return compiled_rope(k, cos, sin)

    local_us = cuda_time_us(local_rope, warmup, repeats)
    compiled_us = cuda_time_us(compiled, warmup, repeats)

    vllm_us = None
    vllm_error = None
    emb, vllm_error = try_vllm_rotary(case.head_dim, max_position, dtype)
    if emb is not None:
        def vllm_forward_cuda():
            tmp = k.clone()
            emb.forward_cuda(positions, tmp, None)
            return tmp

        try:
            vllm_us = cuda_time_us(vllm_forward_cuda, warmup, repeats)
        except Exception as exc:
            vllm_error = repr(exc)
            torch.cuda.synchronize()

    return {
        "case": case.name,
        "mode": case.mode,
        "dtype": case.dtype_name,
        "total_tokens": total_tokens,
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "local_rope_ref_us": round(local_us, 3),
        "compiled_rope_ref_us": round(compiled_us, 3),
        "compiled_vs_local": round(local_us / compiled_us, 4),
        "vllm_forward_cuda_us": None if vllm_us is None else round(vllm_us, 3),
        "vllm_vs_local": None if vllm_us is None else round(local_us / vllm_us, 4),
        "vllm_error": vllm_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RoPE provider cost split.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "rope_provider_split.json"))
    args = parser.parse_args()

    cases = build_cases()
    rows = []
    started = time.time()
    for idx, case in enumerate(cases, 1):
        row = run_case(case, args.warmup, args.repeats)
        rows.append(row)
        print(
            f"PROGRESS {idx}/{len(cases)} {idx / len(cases):.0%} {case.name} "
            f"compiled_vs_local={row['compiled_vs_local']:.3f} "
            f"vllm={'n/a' if row['vllm_forward_cuda_us'] is None else row['vllm_forward_cuda_us']}",
            flush=True,
        )

    payload = {
        "benchmark": "rope_provider_split",
        "rows": rows,
        "elapsed_s": round(time.time() - started, 3),
        "purpose": "Estimate how much baseline time is RoPE-provider choice rather than fusion.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
