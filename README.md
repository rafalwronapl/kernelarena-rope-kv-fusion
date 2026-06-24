# KernelArena Finding: RoPE + KV-Cache Write Fusion v0

Date: 2026-06-24

## Summary

This finding reports a narrow NVIDIA/Triton microbenchmark result for:

```text
K RoPE + KV-cache write
```

On an RTX 4090, a fused Triton kernel beat decomposed `vLLM RoPE + contract
write` and `Inductor RoPE + contract write` on selected vLLM-cache-contract
rows.

This is not an end-to-end serving benchmark and not a production vLLM cache-path
claim.

## Main Result

Environment:

```text
GPU: NVIDIA GeForce RTX 4090
driver: 550.127.05
vLLM: 0.9.2
torch: 2.7.0+cu126
triton: 3.3.0
oracle: contract_oracle
```

Single full run:

```text
rows=16
correct=16/16
vLLM baseline blocked rows=0
fused vs vLLM RoPE + contract write: min 1.5198x, median 2.08085x, max 2.7586x
fused vs Inductor RoPE + contract write: min 1.6120x, median 2.8805x, max 5.0255x
```

Same-pod repeat stability:

```text
repeat_runs=3
rows=48
correct=48/48
vLLM baseline blocked rows=0
fused vs vLLM RoPE + contract write: min 1.5026x, median 2.05375x
robust median excluding obvious >5x timing outliers: 1.8125x
```

Outliers above `5x` are timing noise and are not headline claims.

## Why This Is Interesting

Standalone RoPE is not the strongest story: optimized vLLM standalone RoPE is
already good. The interesting lane is the boundary between RoPE and writing K/V
into cache.

The fused Triton path removes a separate memory pass:

```text
decomposed: RoPE(K) -> write K/V to cache
fused:      RoPE(K) + write K/V to cache
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

KernelArena's contribution here is narrower:

```text
a small NVIDIA/Triton contract-oracle repro with JSON artifacts and repeat
evidence for fp16/bf16 selected layouts
```

## FlashInfer Probe

FlashInfer API probe on RTX 4090:

```text
flashinfer=0.6.12
classification=partial_comparable
```

No timing comparison against FlashInfer is claimed. The discovered fused
FlashInfer path is FP8/quantized and includes Q/K/nope inputs, so comparing it
directly against this fp16/bf16 contract benchmark would be misleading.

## Files

Script:

```text
benchmark_rope_vllm_cache_contract.py
```

Artifacts:

```text
artifacts/rope_vllm_cache_contract_4090_summary.json
artifacts/rope_vllm_cache_contract_4090.json
artifacts/rope_vllm_cache_contract_4090_results.tar.gz
artifacts/rope_vllm_cache_contract_4090_repeat3_summary.json
artifacts/rope_vllm_cache_contract_4090_repeat3_results.tar.gz
artifacts/flashinfer_api_probe_4090.json
artifacts/rope_flashinfer_compare_4090_blocked.json
artifacts/rope_flashinfer_compare_4090_results.tar.gz
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
KernelArena has RTX 4090 contract-oracle evidence that a fused Triton
K-RoPE + KV-cache-write kernel beats decomposed vLLM RoPE + contract write and
Inductor RoPE + contract write on selected vLLM-cache-contract rows. This is a
microbenchmark finding, not a full serving or production vLLM cache-path claim.
```
