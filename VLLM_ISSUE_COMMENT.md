# Draft Comment For vLLM Issue #24678

Hi, this issue matches a narrow fusion lane we have been testing in
KernelArena.

I saw that vLLM now documents a `rope_kvcache_fusion` pass, with the stable
fusion docs describing RoPE + KV-cache update fusion as ROCm/AITER-only rather
than NVIDIA CUDA. This note is therefore specifically about a small
NVIDIA/Triton microbenchmark prototype, not a claim that vLLM has no work in
this area.

The prototype covers the narrower:

```text
K RoPE + KV-cache write
```

part of the broader `ROPE + KV-Cache-Write + pre-attn prepare-ops fusion`
problem.

Latest contract-oracle result:

```text
RTX 4090
vLLM 0.9.2 / torch 2.7.0+cu126 / triton 3.3.0
16/16 rows correct
vLLM baseline not blocked
fused vs vLLM RoPE + contract write: min 1.5198x, median 2.08085x, max 2.7586x
fused vs Inductor RoPE + contract write: min 1.6120x, median 2.8805x, max 5.0255x
oracle=contract_oracle
```

Same-pod repeat stability:

```text
RTX 4090 repeat_runs=3
48/48 rows correct
vLLM baseline not blocked
fused vs vLLM RoPE + contract write: min 1.5026x, median 2.05375x
robust median excluding obvious >5x timing outliers: 1.8125x
```

This is not an end-to-end serving benchmark and not a production vLLM cache-path
claim. It is a selected-layout NVIDIA/Triton contract microbenchmark for the
RoPE + cache-write subproblem.

We also have earlier synthetic contiguous and paged results on RTX 3090 and A40
showing similar margins (RTX 3090 contiguous: min 1.9714x, median 2.0815x vs
vLLM+write; A40 contiguous: min 2.0015x, median 2.0295x).

If useful, I can share the benchmark script, JSON summaries, environment notes,
and the full 3-GPU synthetic artifact pack.
