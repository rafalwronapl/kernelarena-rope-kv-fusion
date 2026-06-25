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

Latest contract-oracle result (prefill only — see note below):

```text
RTX 4090
vLLM 0.9.2 / torch 2.7.0+cu126 / triton 3.3.0
prefill cases only (8/16 rows)
fused vs vLLM RoPE + contract write: prefill median 1.6810x
robust median all rows excluding >5x outliers: 1.8125x
oracle=contract_oracle
```

**Note on decode results:** decode cases (small batches, 1 token/sequence) show
higher raw speedups but these are dominated by kernel launch overhead, not compute.
Under CUDA Graphs (which production vLLM uses for decode), launch overhead is
amortized and these margins likely do not survive. We are not reporting decode
as a headline number for this reason.

Prefill at large context lengths is compute-bounded and is the honest headline:
~1.5-1.7x for sequences of 1k-8k tokens.

This is not an end-to-end serving benchmark and not a production vLLM cache-path
claim. It is a selected-layout NVIDIA/Triton contract microbenchmark for the
prefill RoPE + cache-write subproblem.

Additional context — synthetic contiguous results on RTX 3090 and A40 showed
similar prefill margins (RTX 3090: median ~2.0x, A40: median ~2.0x vs
decomposed vLLM+write, also non-CUDA-Graph eager measurement).

If useful, I can share the benchmark script, JSON summaries, environment notes,
and the full 3-GPU synthetic artifact pack.
