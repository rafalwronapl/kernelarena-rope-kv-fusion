# Draft Comment For vLLM Issue #24678

I have a small NVIDIA/Triton microbenchmark artifact that may be relevant to
`ROPE + KV-Cache-Write + pre-attn prepare-ops fusion`.

Scope first: this is not an end-to-end vLLM serving benchmark and not a claim
that vLLM has no related work. It is a selected-layout kernel microbenchmark for
K RoPE + KV-cache write on RTX 4090.

Stack:

```text
RTX 4090
vLLM 0.9.2 / torch 2.7.0+cu126 / triton 3.3.0
cache writer baseline: torch.ops._C_cache_ops.reshape_and_cache from vllm._C
RoPE provider in this run: local reference fallback
```

Result summary:

```text
real vLLM cache writer run:
rows=16, correct=16/16
prefill: min 1.6572x, median 1.9319x, max 2.2122x vs real reshape_and_cache path
decode:  min 3.9782x, median 4.3409x, max 4.6780x vs real reshape_and_cache path

CUDA Graph decode run:
rows=8, correct=8/8
CUDA Graph replay speedup: min 2.6659x, median 2.7713x, max 3.1179x
```

The CUDA Graph result is the important part: the decode improvement does not
collapse to 1x when launch overhead is mostly removed.

Caveat: I only loaded the narrow vLLM CUDA extension path on the pod, not the
full vLLM Python runtime dependency stack, so RoPE itself used a local reference
fallback. The cache writer baseline is still the real vLLM CUDA op:
`torch.ops._C_cache_ops.reshape_and_cache`.

Artifacts and scripts are in:

```text
https://github.com/rafalwronapl/kernelarena-rope-kv-fusion
```

I would treat this as evidence that an NVIDIA/Triton fused RoPE + KV-cache-write
path is worth testing inside vLLM proper, not as production-ready replacement
code.
