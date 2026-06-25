# Reproduction Notes

## Known Working Stack

The clean RTX 4090 real-cache-writer result used:

```text
GPU: NVIDIA GeForce RTX 4090
vLLM: 0.9.2
torch: 2.7.0+cu126
triton: 3.3.0
python: 3.11
```

The key requirement is that `vllm._C` loads successfully and registers:

```text
torch.ops._C_cache_ops.reshape_and_cache
```

The recorded 2026-06-25 run used vLLM's real CUDA cache writer but a local RoPE
reference fallback. That was intentional after the pod image lacked the full
vLLM Python dependency stack. The benchmark scope is therefore the fused RoPE +
real vLLM cache-write microbenchmark, not full vLLM serving.

## Setup Sketch

```bash
python -m venv --system-site-packages /workspace/ka-vllm092-env
source /workspace/ka-vllm092-env/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.0 triton==3.3.0
pip install vllm==0.9.2 --no-deps
```

If PyTorch CUDA libraries are shadowed by system packages, set `LD_LIBRARY_PATH`
so the venv NVIDIA package libraries come first.

Smoke check:

```bash
python - <<'PY'
import torch
import vllm._C
print(torch.ops._C_cache_ops.reshape_and_cache)
print(torch.cuda.get_device_name(0))
PY
```

## Real vLLM Cache Writer Benchmark

```bash
python benchmark_rope_real_vllm_contract.py \
  --warmup 20 \
  --repeats 100 \
  --output artifacts/rope_real_vllm_contract_summary.json
```

Repeat stability:

```bash
for i in 1 2 3; do
  python benchmark_rope_real_vllm_contract.py \
    --warmup 20 \
    --repeats 100 \
    --output artifacts/rope_real_vllm_contract_repeat${i}.json
done
```

Expected result shape:

```text
rows=16
correct=16/16
oracle=real_vllm_oracle
reshape_and_cache_path=vllm._C + torch.ops._C_cache_ops.reshape_and_cache
```

## CUDA Graph Decode Benchmark

```bash
python benchmark_rope_cudagraph_decode.py \
  --warmup 20 \
  --repeats 200 \
  --output artifacts/rope_cudagraph_decode_summary.json
```

Expected result shape:

```text
rows=8
correct=8/8 for the flat-layout contract oracle
speedup_graph present for all rows
```

## Real vLLM CUDA Graph Decode Follow-Up

This follow-up uses real `torch.ops._C_cache_ops.reshape_and_cache` in the
baseline. Use the current script version so the JSON includes cache correctness
fields.

```bash
python benchmark_rope_real_vllm_cudagraph_decode.py \
  --warmup 20 \
  --repeats 200 \
  --output artifacts/rope_real_vllm_cudagraph_decode_summary_<gpu>.json
```

Expected result shape:

```text
rows=8
oracle=real_vllm_oracle
correct=8/8
baseline_k_correct=8/8
baseline_v_correct=8/8
k_max_abs_diff and v_max_abs_diff present for each row
speedup_graph present for all rows
```

## RoPE Provider Split

```bash
python benchmark_rope_provider_split.py \
  --warmup 10 \
  --repeats 50 \
  --output artifacts/rope_provider_split_<gpu>.json
```

This estimates how much of the baseline cost comes from the RoPE provider
choice. On the recorded RTX 3090 run, full vLLM `RotaryEmbedding.forward_cuda`
was unavailable in the minimal install, so the artifact compares local eager
RoPE reference versus compiled RoPE reference.

The exact latency ratios may vary by pod, driver, thermal state, and GPU clocks.
