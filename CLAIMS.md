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
vLLM RoPE + contract write with min 1.5198x and median 2.08085x on 16/16
correct rows.
```

Allowed:

```text
Same-pod repeat stability showed 48/48 correct rows and min 1.5026x vs vLLM
RoPE + contract write.
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

## Outliers

Outliers above `5x` in repeat rows are timing noise and must not be used as
headline claims.
