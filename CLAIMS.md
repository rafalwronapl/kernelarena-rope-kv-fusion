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
In the RTX 3090 real-cache-writer CUDA Graph decode timing benchmark, baseline
was local RoPE reference plus real `torch.ops._C_cache_ops.reshape_and_cache`;
graph replay speedup was min 3.0316x, median 4.3983x, max 5.0125x across 8
selected decode rows.
```

Required caveat: the first 3090 timing artifact does not contain cache
correctness fields. Do not claim `correct=8/8` for that follow-up until rerun
with the updated correctness-checking script.

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

The RTX 3090 follow-up fixes that baseline mismatch for decode timing: it uses
real `torch.ops._C_cache_ops.reshape_and_cache` and still shows graph replay
speedup from 3.0316x to 5.0125x. Its first artifact lacks correctness fields, so
it is timing evidence pending rerun.

Allowed wording:

```text
The selected contract-layout decode microbenchmark retained about 2.7x-3.1x
speedup under CUDA Graph replay.
```

Do not phrase this as an end-to-end serving speedup or as a CUDA Graph win
against real vLLM `reshape_and_cache`.

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
- Rerun RTX 3090 real-cache-writer CUDA Graph decode with the updated script so
  the JSON includes `correct`, `k_correct`, `v_correct`, baseline correctness,
  and max-diff fields.
- End-to-end vLLM integration before any production-path claim.
