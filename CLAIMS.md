# Claim Policy

## Allowed Claims

Allowed:

```text
RTX 4090 contract-oracle evidence for a fused Triton K-RoPE + KV-cache-write
kernel on selected vLLM-cache-contract rows.
```

Allowed:

```text
On the recorded RTX 4090 contract benchmark, fused RoPE+KV-write beat decomposed
vLLM RoPE + contract write for prefill cases with median 1.68x on 8/16 correct
prefill rows. Robust median across all rows (excluding >5x outliers): 1.81x.
```

Allowed:

```text
Same-pod repeat stability showed 48/48 correct rows and prefill-robust median
~1.68x vs vLLM RoPE + contract write.
```

Allowed:

```text
FlashInfer API probe was partial-comparable; no timing claim against FlashInfer
is made.
```

## Blocked Claims

Do not claim:

- full LLM serving speedup
- production vLLM cache-path win
- end-to-end vLLM win
- official TritonBench result
- upstream result
- SOTA RoPE
- FlashInfer win/loss
- broad inference acceleration
- that vLLM has no implementation at all
- that this replaces vLLM, FlashInfer, FlashAttention, or any production stack

## Required Caveats

Use these caveats near any public result:

```text
contract_oracle microbenchmark
selected layouts
NVIDIA RTX 4090
not full serving
not production cache path
not FlashInfer comparison
```

## Decode Results

Decode cases (1 token per sequence, small batch) show higher raw speedups
but are kernel-launch-overhead dominated in eager mode. Under CUDA Graphs
(production vLLM decode path) this overhead is amortized and these margins
likely do not survive. Do not use decode numbers as headline claims.

## Outliers

Outliers above `5x` in repeat rows are timing noise and must not be used as
headline claims.

## Clone Asymmetry

The baseline calls k.clone() inside the timed loop because vLLM's forward_cuda
modifies k in-place. The fused kernel reads k without cloning. This adds ~1-2%
overhead to baseline for prefill. Results are labeled contract_oracle to reflect
this methodological caveat.
