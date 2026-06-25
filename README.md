# KernelArena Finding: RoPE + KV-Cache Write Fusion v1

```text
Status: narrow microbenchmark finding
Scope:  NVIDIA RTX 4090, selected vLLM paged-cache layouts
Now:    real vLLM `vllm._C` cache writer + CUDA Graph decode evidence
Not:    full serving benchmark | end-to-end vLLM win | FlashInfer timing comparison
```

Date: 2026-06-25

## Summary

This repository reports a narrow NVIDIA/Triton microbenchmark result for:

```text
K RoPE + KV-cache write
```

On an RTX 4090, a fused Triton kernel beat a decomposed path that writes through
vLLM's real CUDA cache writer:

```text
torch.ops._C_cache_ops.reshape_and_cache
```

The strongest new evidence is that the decode advantage also survives CUDA Graph
replay. That reduces the main prior concern that the decode result was only a
kernel-launch-overhead artifact.

This is still a microbenchmark finding. It is not an end-to-end vLLM serving
benchmark, not an official TritonBench result, and not a FlashInfer comparison.

## Main Result: Real vLLM Cache Writer

Environment:

```text
GPU: NVIDIA GeForce RTX 4090
vLLM: 0.9.2
torch: 2.7.0+cu126
triton: 3.3.0
oracle: real_vllm_oracle
cache writer: vLLM `vllm._C` / `torch.ops._C_cache_ops.reshape_and_cache`
RoPE provider in this run: local reference RoPE fallback
```

Full run:

```text
rows=16
correct=16/16
fused vs real vLLM reshape_and_cache: min 1.6572x, median 3.0952x, max 4.6780x
prefill rows: min 1.6572x, median 1.9319x, max 2.2122x
decode rows:  min 3.9782x, median 4.3409x, max 4.6780x
```

Same-pod repeats:

```text
repeat_runs=3
rows=48
correct=48/48
prefill median vs real vLLM reshape_and_cache: about 1.93x-1.97x
decode median vs real vLLM reshape_and_cache: about 4.30x-4.39x
```

Important caveat: the cache writer is vLLM's real CUDA custom op, but the RoPE
calculation in this run used a local reference fallback because the RunPod image
had only the narrow vLLM extension path installed, not the full vLLM Python
runtime dependency stack. Therefore the fair claim is about a fused RoPE + real
vLLM cache-write microbenchmark, not a full vLLM model path.

## CUDA Graph Decode Result

CUDA Graph decode benchmark:

```text
rows=8
correct=8/8
eager speedup: min 5.1667x, median 5.7302x, max 6.1404x
CUDA Graph replay speedup: min 2.6659x, median 2.7713x, max 3.1179x
```

Interpretation: decode speedup is not only launch overhead. CUDA Graph replay
removes most per-launch overhead, and the fused kernel still keeps about
`2.7x-3.1x` on these selected decode layouts.

## Why This Is Interesting

Standalone RoPE is not the strongest story. The more useful boundary is:

```text
decomposed: RoPE(K) -> write K/V to paged cache
fused:      RoPE(K) + write K/V to paged cache
```

This matches a real performance area: vLLM issue `#24678` discusses
`ROPE + KV-Cache-Write + pre-attn prepare-ops fusion`.

## Competition Context

This is not an unknown problem category.

- vLLM has public work around RoPE + KV-cache fusion. Current docs include a
  `rope_kvcache_fusion` pass, with stable docs scoping that support to
  ROCm/AITER rather than NVIDIA CUDA/CPU.
- FlashInfer is a real adjacent competitor. Its API probe found separate RoPE
  and paged KV-cache append APIs plus an FP8/quantized fused path,
  `rope_quantize_fp8_append_paged_kv_cache`.

KernelArena's contribution here is narrower: a small NVIDIA/Triton repro with
JSON artifacts showing selected-layout speedups against vLLM's real cache writer
and under CUDA Graph replay.

## Files

Scripts:

```text
benchmark_rope_real_vllm_contract.py
benchmark_rope_cudagraph_decode.py
benchmark_rope_vllm_cache_contract.py  # older contract-oracle baseline
```

New artifacts:

```text
artifacts/real_vllm_4090_rope_real_vllm_contract_summary.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat1.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat2.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat3.json
artifacts/cudagraph_4090_rope_cudagraph_decode_summary.json
artifacts/results_real_vllm_contract_4090_2026-06-25.zip
artifacts/results_cudagraph_decode_4090_2026-06-25.zip
```

Read:

```text
REPRO.md
CLAIMS.md
ARTIFACT_MANIFEST.md
VLLM_ISSUE_COMMENT.md
```

## Recommended Public Wording

```text
KernelArena has RTX 4090 microbenchmark evidence that a fused Triton
K-RoPE + KV-cache-write kernel beats vLLM's real CUDA `reshape_and_cache` cache
writer on selected paged-cache layouts, and that selected decode speedups remain
about 2.7x-3.1x under CUDA Graph replay. This is a kernel microbenchmark, not a
full vLLM serving-speedup claim.
```

## How to Cite

```text
rafalwronapl. KernelArena Finding: RoPE + KV-Cache Write Fusion v1. 2026.
https://github.com/rafalwronapl/kernelarena-rope-kv-fusion
Artifact manifest: ARTIFACT_MANIFEST.md
```
