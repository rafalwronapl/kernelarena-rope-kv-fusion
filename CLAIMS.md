# Claim Policy

## Allowed Claims

Allowed:

```text
RTX 4090 microbenchmark evidence for a fused Triton K-RoPE + KV-cache-write
kernel on selected vLLM paged-cache layouts.
```

Allowed:

```text
The fused Triton kernel beat vLLM's real CUDA cache writer
`torch.ops._C_cache_ops.reshape_and_cache` on the recorded selected RTX 4090
layouts, with 16/16 correct rows in the summary run.
```

Allowed:

```text
In the 2026-06-25 RTX 4090 summary run, fused vs real vLLM cache writer was:
prefill median 1.9319x and decode median 4.3409x, with 16/16 correct rows.
```

Allowed:

```text
In the CUDA Graph decode benchmark, speedup remained min 2.6659x, median
2.7713x, max 3.1179x across 8/8 correct selected decode rows.
```

Allowed:

```text
FlashInfer API probe was partial-comparable; no timing claim against FlashInfer
is made.
```

## Required Caveats

Use these caveats near any public result:

```text
selected-layout microbenchmark
NVIDIA RTX 4090
vLLM 0.9.2 real cache writer only
RoPE provider was local reference fallback in the real-cache-writer run
not full serving
not end-to-end vLLM
not FlashInfer timing comparison
not official TritonBench
```

## Blocked Claims

Do not claim:

- full LLM serving speedup
- end-to-end vLLM win
- production vLLM path win
- official TritonBench result
- upstream result
- SOTA RoPE
- FlashInfer win/loss
- broad inference acceleration
- that vLLM has no implementation at all
- that this replaces vLLM, FlashInfer, FlashAttention, or any production stack

## Decode Results

The old contract-oracle decode results were launch-overhead suspect. The new
CUDA Graph replay benchmark reduces that concern: selected decode rows still
showed 2.6659x-3.1179x speedup under graph replay.

Allowed wording:

```text
The selected decode microbenchmark retained about 2.7x-3.1x speedup under CUDA
Graph replay.
```

Do not phrase this as an end-to-end serving speedup.

## Outliers

Avoid headline claims from single outlier rows. Use min/median/max and separate
prefill from decode. The repeat3 decode max 5.2003x should not be used as the
headline.

## Method Caveat

The real-cache-writer run loaded vLLM's CUDA extension and used
`torch.ops._C_cache_ops.reshape_and_cache`. It did not use the full vLLM Python
runtime RoPE path because that pod had only the narrow extension path installed.
RoPE was computed by the local reference fallback. This is acceptable for a
cache-writer microbenchmark, but it is not a full vLLM runtime benchmark.
