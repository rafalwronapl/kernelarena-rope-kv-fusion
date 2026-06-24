# Reproduction Notes

## Known Working Stack

The clean RTX 4090 contract result used:

```text
GPU: NVIDIA GeForce RTX 4090
driver: 550.127.05
vLLM: 0.9.2
torch: 2.7.0+cu126
triton: 3.3.0
python: 3.11
```

`vLLM 0.9.2` was selected because its CUDA RoPE path supports
`key: Optional[torch.Tensor] = None`. Older `vLLM 0.8.5` required both query and
key tensors and was not a fair K-only baseline.

## Setup Sketch

```bash
python -m venv --system-site-packages /workspace/ka-vllm092-env
source /workspace/ka-vllm092-env/bin/activate
python -m pip install --upgrade pip
TMPDIR=/workspace PIP_CACHE_DIR=/workspace/pip-cache-vllm092 \
  pip install vllm==0.9.2 torch==2.7.0 triton torchaudio==2.7.0 torchvision==0.22.0
```

Depending on the base image, additional vLLM dependencies may be needed. The
recorded run installed the minimum dependencies required for:

```text
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm import _custom_ops
```

## Commands

Dry run:

```bash
python benchmark_rope_vllm_cache_contract.py \
  --dry-run \
  --output artifacts/rope_vllm_cache_contract_dry_run_4090.json
```

Full run:

```bash
python benchmark_rope_vllm_cache_contract.py \
  --warmup 20 \
  --repeats 100 \
  --output artifacts/rope_vllm_cache_contract_4090.json
```

Repeat stability:

```bash
for i in 1 2 3; do
  python benchmark_rope_vllm_cache_contract.py \
    --warmup 20 \
    --repeats 100 \
    --output artifacts/rope_vllm_cache_contract_4090_repeat${i}.json
done
```

## Expected Result Shape

The clean 4090 run should produce:

```text
rows=16
correct=16/16
vLLM baseline blocked rows=0
```

The exact latency ratios may vary by pod and driver.
