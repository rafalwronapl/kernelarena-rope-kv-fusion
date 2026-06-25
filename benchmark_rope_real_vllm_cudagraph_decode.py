from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from benchmark_rope_real_vllm_contract import (
    DTYPES,
    build_cases,
    build_contract,
    build_cos_sin,
    cuda_time_us,
    fused_vllm_layout_write,
    reshape_and_cache,
    rope_ref,
    vllm_layout_cache,
    vllm_layout_flat_views,
)


REPO_ROOT = Path(__file__).resolve().parent


def optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else None


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def run_case(case, warmup: int, repeats: int) -> dict[str, object]:
    if reshape_and_cache is None:
        raise RuntimeError("vLLM reshape_and_cache is unavailable")

    dtype = DTYPES[case.dtype_name]
    contract = build_contract(case)
    total_tokens = int(contract["total_tokens"])
    slots = torch.tensor(contract["slots"], device="cuda", dtype=torch.long)
    positions = torch.tensor(contract["positions"], device="cuda", dtype=torch.long)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    k = torch.randn(total_tokens, case.kv_heads, case.head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    k_cache, v_cache = vllm_layout_cache(
        int(contract["num_physical_blocks"]), case.kv_heads, case.head_dim, case.block_size, dtype
    )
    cos, sin = build_cos_sin(positions, case.head_dim, dtype)
    expected_k = rope_ref(k, cos, sin)

    fused_vllm_layout_write(k, v, cos, sin, slots, k_cache, v_cache, case.triton_block)
    torch.cuda.synchronize()
    flat_k_cache, flat_v_cache = vllm_layout_flat_views(k_cache, v_cache)
    k_correct = torch.allclose(flat_k_cache[slots].float(), expected_k.float(), atol=2e-2, rtol=2e-2)
    v_correct = torch.allclose(flat_v_cache[slots].float(), v.float(), atol=2e-2, rtol=2e-2)
    k_diff = (flat_k_cache[slots].float() - expected_k.float()).abs().max().item()
    v_diff = (flat_v_cache[slots].float() - v.float()).abs().max().item()

    baseline_check_k, baseline_check_v = vllm_layout_cache(
        int(contract["num_physical_blocks"]), case.kv_heads, case.head_dim, case.block_size, dtype
    )
    reshape_and_cache(expected_k, v, baseline_check_k, baseline_check_v, slots, "auto", scale, scale)
    torch.cuda.synchronize()
    base_flat_k, base_flat_v = vllm_layout_flat_views(baseline_check_k, baseline_check_v)
    baseline_k_correct = torch.allclose(base_flat_k[slots].float(), expected_k.float(), atol=2e-2, rtol=2e-2)
    baseline_v_correct = torch.allclose(base_flat_v[slots].float(), v.float(), atol=2e-2, rtol=2e-2)

    k_graph = k.clone()
    v_graph = v.clone()
    cos_graph = cos.clone()
    sin_graph = sin.clone()
    slots_graph = slots.clone()
    baseline_tmp = torch.empty_like(k_graph)
    k_cache_base, v_cache_base = vllm_layout_cache(
        int(contract["num_physical_blocks"]), case.kv_heads, case.head_dim, case.block_size, dtype
    )
    k_cache_fused, v_cache_fused = vllm_layout_cache(
        int(contract["num_physical_blocks"]), case.kv_heads, case.head_dim, case.block_size, dtype
    )

    def baseline_eager():
        tmp = rope_ref(k, cos, sin)
        reshape_and_cache(tmp, v, k_cache, v_cache, slots, "auto", scale, scale)

    def fused_eager():
        fused_vllm_layout_write(k, v, cos, sin, slots, k_cache, v_cache, case.triton_block)

    def baseline_static():
        baseline_tmp.copy_(rope_ref(k_graph, cos_graph, sin_graph))
        reshape_and_cache(baseline_tmp, v_graph, k_cache_base, v_cache_base, slots_graph, "auto", scale, scale)

    def fused_static():
        fused_vllm_layout_write(
            k_graph, v_graph, cos_graph, sin_graph, slots_graph, k_cache_fused, v_cache_fused, case.triton_block
        )

    eager_baseline_us = cuda_time_us(baseline_eager, warmup, repeats)
    eager_fused_us = cuda_time_us(fused_eager, warmup, repeats)

    graph_error = None
    graph_baseline_us = None
    graph_fused_us = None
    speedup_graph = None
    graph_fused_k_correct = None
    graph_fused_v_correct = None
    graph_baseline_k_correct = None
    graph_baseline_v_correct = None
    graph_fused_k_diff = None
    graph_fused_v_diff = None
    graph_baseline_k_diff = None
    graph_baseline_v_diff = None
    try:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                baseline_static()
                fused_static()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        g_baseline = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_baseline):
            baseline_static()
        g_fused = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fused):
            fused_static()

        graph_baseline_us = cuda_time_us(g_baseline.replay, 20, max(repeats, 200))
        graph_fused_us = cuda_time_us(g_fused.replay, 20, max(repeats, 200))
        speedup_graph = round(graph_baseline_us / graph_fused_us, 4)

        g_baseline.replay()
        g_fused.replay()
        torch.cuda.synchronize()
        graph_base_flat_k, graph_base_flat_v = vllm_layout_flat_views(k_cache_base, v_cache_base)
        graph_fused_flat_k, graph_fused_flat_v = vllm_layout_flat_views(k_cache_fused, v_cache_fused)
        graph_baseline_k_diff = (
            graph_base_flat_k[slots_graph].float() - expected_k.float()
        ).abs().max().item()
        graph_baseline_v_diff = (
            graph_base_flat_v[slots_graph].float() - v_graph.float()
        ).abs().max().item()
        graph_fused_k_diff = (
            graph_fused_flat_k[slots_graph].float() - expected_k.float()
        ).abs().max().item()
        graph_fused_v_diff = (
            graph_fused_flat_v[slots_graph].float() - v_graph.float()
        ).abs().max().item()
        graph_baseline_k_correct = torch.allclose(
            graph_base_flat_k[slots_graph].float(), expected_k.float(), atol=2e-2, rtol=2e-2
        )
        graph_baseline_v_correct = torch.allclose(
            graph_base_flat_v[slots_graph].float(), v_graph.float(), atol=2e-2, rtol=2e-2
        )
        graph_fused_k_correct = torch.allclose(
            graph_fused_flat_k[slots_graph].float(), expected_k.float(), atol=2e-2, rtol=2e-2
        )
        graph_fused_v_correct = torch.allclose(
            graph_fused_flat_v[slots_graph].float(), v_graph.float(), atol=2e-2, rtol=2e-2
        )
    except Exception as exc:
        graph_error = repr(exc)
        torch.cuda.synchronize()

    return {
        "case": case.name,
        "mode": case.mode,
        "dtype": case.dtype_name,
        "seq_lens": list(case.seq_lens),
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "block_size": case.block_size,
        "total_tokens": total_tokens,
        "oracle": "real_vllm_oracle",
        "baseline": "local_rope_ref + torch.ops._C_cache_ops.reshape_and_cache",
        "fused": "fused Triton RoPE + vLLM paged-layout KV write",
        "correct": bool(k_correct and v_correct and baseline_k_correct and baseline_v_correct),
        "k_correct": bool(k_correct),
        "v_correct": bool(v_correct),
        "baseline_k_correct": bool(baseline_k_correct),
        "baseline_v_correct": bool(baseline_v_correct),
        "graph_correct": None
        if graph_fused_k_correct is None
        else bool(graph_fused_k_correct and graph_fused_v_correct and graph_baseline_k_correct and graph_baseline_v_correct),
        "graph_fused_k_correct": None if graph_fused_k_correct is None else bool(graph_fused_k_correct),
        "graph_fused_v_correct": None if graph_fused_v_correct is None else bool(graph_fused_v_correct),
        "graph_baseline_k_correct": None if graph_baseline_k_correct is None else bool(graph_baseline_k_correct),
        "graph_baseline_v_correct": None if graph_baseline_v_correct is None else bool(graph_baseline_v_correct),
        "k_max_abs_diff": k_diff,
        "v_max_abs_diff": v_diff,
        "graph_fused_k_max_abs_diff": graph_fused_k_diff,
        "graph_fused_v_max_abs_diff": graph_fused_v_diff,
        "graph_baseline_k_max_abs_diff": graph_baseline_k_diff,
        "graph_baseline_v_max_abs_diff": graph_baseline_v_diff,
        "eager_baseline_us": round(eager_baseline_us, 3),
        "eager_fused_us": round(eager_fused_us, 3),
        "speedup_eager": round(eager_baseline_us / eager_fused_us, 4),
        "graph_baseline_us": None if graph_baseline_us is None else round(graph_baseline_us, 3),
        "graph_fused_us": None if graph_fused_us is None else round(graph_fused_us, 3),
        "speedup_graph": speedup_graph,
        "graph_error": graph_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA Graph decode benchmark against real vLLM reshape_and_cache.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "rope_real_vllm_cudagraph_decode.json"))
    args = parser.parse_args()

    cases = [case for case in build_cases() if case.mode == "decode"]
    rows = []
    started = time.time()
    for idx, case in enumerate(cases, 1):
        row = run_case(case, args.warmup, args.repeats)
        rows.append(row)
        print(
            f"PROGRESS {idx}/{len(cases)} {idx / len(cases):.0%} {case.name} "
            f"eager={format_ratio(row['speedup_eager'])} graph={format_ratio(row['speedup_graph'])} "
            f"oracle={row['oracle']}",
            flush=True,
        )

    payload = {
        "benchmark": "rope_real_vllm_cudagraph_decode",
        "oracle": "real_vllm_oracle",
        "stack": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "triton": optional_version("triton"),
            "vllm": optional_version("vllm"),
            "cuda": torch.version.cuda,
            "driver": nvidia_driver_version(),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "rows": rows,
        "elapsed_s": round(time.time() - started, 3),
        "non_claims": [
            "not an end-to-end vLLM serving benchmark",
            "RoPE provider is local tensor reference unless a separate provider split says otherwise",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
