# KernelArena Finding: RoPE + KV-Cache Write Fusion v1

```text
Status: narrow microbenchmark finding
Scope:  NVIDIA RTX 4090/3090, selected vLLM paged-cache layouts
Now:    real vLLM `vllm._C` cache writer evidence + real-cache-writer CUDA Graph decode evidence
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

The strongest new cache-writer evidence uses vLLM's real CUDA
`reshape_and_cache` op. After an external review pointed out a baseline mismatch,
we added a second RTX 3090 CUDA Graph decode benchmark that also uses real
`reshape_and_cache` as the baseline.

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
runtime dependency stack. A provider-split check on RTX 3090 showed compiled
RoPE is about `4.06x` faster than the local eager RoPE reference on prefill rows.
Therefore the prefill numbers should not be presented as pure fusion benefit.

## CUDA Graph Decode Result

Original RTX 4090 contract-layout CUDA Graph decode benchmark:

```text
rows=8
correct=8/8
oracle: contract_oracle
baseline: local RoPE reference + flat-layout contract write
eager speedup: min 5.1667x, median 5.7302x, max 6.1404x
CUDA Graph replay speedup: min 2.6659x, median 2.7713x, max 3.1179x
```

Interpretation: for the flat contract-layout decode benchmark, speedup is not
only launch overhead. CUDA Graph replay removes most per-launch overhead, and
the fused kernel still keeps about `2.7x-3.1x` on these selected decode layouts.

This is **not yet** a CUDA Graph claim against vLLM's real
`reshape_and_cache` cache writer. That is the next validation target.

Follow-up RTX 3090 real-cache-writer CUDA Graph decode benchmark:

```text
rows=8
oracle: real_vllm_oracle
baseline: local RoPE reference + real torch.ops._C_cache_ops.reshape_and_cache
eager speedup: min 4.6724x, median 4.7766x, max 5.1641x
CUDA Graph replay speedup: min 3.0316x, median 4.3983x, max 5.0125x
```

Interpretation: the CUDA Graph baseline mismatch is fixed for decode on RTX
3090. The remaining caveat is RoPE-provider contamination: baseline RoPE is a
local tensor reference, not vLLM `RotaryEmbedding.forward_cuda`.

## RoPE Provider Split

RTX 3090 provider split:

```text
rows=16
compiled_rope_ref vs local_rope_ref: min 1.4786x, median 2.8656x, max 4.1636x
prefill compiled_rope_ref vs local_rope_ref median: 4.0641x
decode compiled_rope_ref vs local_rope_ref median: 1.4994x
vLLM RotaryEmbedding.forward_cuda: unavailable on this minimal vLLM install
```

This confirms the reviewer concern: part of the prefill gap is RoPE-provider
choice, not only fusion. Decode remains more robust, especially after the real
`reshape_and_cache` CUDA Graph test.

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
JSON artifacts showing selected-layout speedups against vLLM's real cache writer,
a real-cache-writer CUDA Graph decode check on RTX 3090, and an explicit
RoPE-provider split caveat.

## Files

Scripts:

```text
benchmark_rope_real_vllm_contract.py
benchmark_rope_real_vllm_cudagraph_decode.py  # follow-up for real reshape_and_cache under CUDA Graph
benchmark_rope_cudagraph_decode.py
benchmark_rope_provider_split.py
benchmark_rope_vllm_cache_contract.py  # older contract-oracle baseline
```

New artifacts:

```text
artifacts/real_vllm_4090_rope_real_vllm_contract_summary.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat1.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat2.json
artifacts/real_vllm_4090_rope_real_vllm_contract_repeat3.json
artifacts/cudagraph_4090_rope_cudagraph_decode_summary.json
artifacts/real_vllm_cudagraph_decode_summary_3090.json
artifacts/rope_provider_split_3090.json
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
writer on selected paged-cache layouts. A follow-up RTX 3090 decode benchmark
using real `reshape_and_cache` under CUDA Graph replay retains about
3.0x-5.0x speedup on selected rows. Prefill numbers are RoPE-provider
contaminated and should not be framed as pure fusion benefit. This is a kernel
microbenchmark, not a full vLLM serving-speedup claim.
```

## How to Cite

```text
rafalwronapl. KernelArena Finding: RoPE + KV-Cache Write Fusion v1. 2026.
https://github.com/rafalwronapl/kernelarena-rope-kv-fusion
Artifact manifest: ARTIFACT_MANIFEST.md
```
