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

Allowed with caveat:

```text
In the 2026-06-25 RTX 4090 summary run, fused vs real vLLM cache writer was:
prefill median 1.9319x and decode median 4.3409x, with 16/16 correct rows.
```

Required caveat: the prefill number is RoPE-provider contaminated because the
baseline used local eager RoPE reference plus real `reshape_and_cache`.

Allowed:

```text
In the separate contract-layout CUDA Graph decode benchmark, speedup remained
min 2.6659x, median 2.7713x, max 3.1179x across 8/8 correct selected decode
rows.
```

Allowed:

```text
In the RTX 3090 real-cache-writer CUDA Graph decode benchmark, baseline
was local RoPE reference plus real `torch.ops._C_cache_ops.reshape_and_cache`;
graph replay speedup was min 2.8578x, median 3.1735x, max 3.6700x. The same
captured fused and baseline paths have 8/8 path correctness checks.
```

Required caveat: this is still a microbenchmark. The baseline cache writer is
real vLLM `reshape_and_cache`, but RoPE is a local tensor reference rather than
vLLM `RotaryEmbedding.forward_cuda`. The current artifact validates path outputs
before graph timing, not cache contents after graph replay.

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
CUDA Graph result uses contract_oracle, not real reshape_and_cache baseline
RTX 3090 follow-up includes real reshape_and_cache under CUDA Graph for decode
prefill numbers are contaminated by local RoPE reference cost
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
showed 2.6659x-3.1179x speedup under graph replay. This CUDA Graph benchmark
still uses `contract_oracle` / flat-layout contract write, not vLLM's real
`reshape_and_cache` cache writer.

The RTX 3090 follow-up fixes that baseline mismatch for decode: it uses real
`torch.ops._C_cache_ops.reshape_and_cache` and shows graph replay speedup from
2.8578x to 3.6700x. It also validates the fused and baseline path cache outputs,
with `correct=8/8`, `baseline_k_correct=8/8`, and `baseline_v_correct=8/8`.
The current artifact does not separately validate the cache after replaying the
captured CUDA Graphs.

Allowed wording:

```text
The selected real-cache-writer decode microbenchmark retained about 2.9x-3.7x
speedup under CUDA Graph replay on RTX 3090, with 8/8 path correctness for the
captured fused and baseline paths.
```

Do not phrase this as an end-to-end serving speedup or as a full vLLM runtime
win.

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

## Open Validation Gaps

Before strengthening the claim, run:

- Full vLLM `RotaryEmbedding.forward_cuda` provider split. Current 3090 split
  measured local eager RoPE versus compiled RoPE; full vLLM Rotary import was
  unavailable on the minimal vLLM install.
- Rerun RTX 3090 real-cache-writer CUDA Graph decode with the updated script to
  add explicit post-replay cache correctness fields.
- End-to-end vLLM integration before any production-path claim.
