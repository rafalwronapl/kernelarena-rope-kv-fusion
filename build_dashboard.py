from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from validate_artifacts import load_json, validate_benchmarks


ROOT = Path(__file__).resolve().parent


def payload(root: Path) -> dict[str, Any]:
    real = load_json(root / "artifacts" / "real_vllm_4090_rope_real_vllm_contract_summary.json")
    graph = load_json(root / "artifacts" / "real_vllm_cudagraph_decode_summary_3090_postreplay.json")
    provider = load_json(root / "artifacts" / "rope_provider_split_3090.json")
    blocked = load_json(root / "artifacts" / "rope_flashinfer_compare_4090_blocked.json")
    validation, errors = validate_benchmarks(root)
    rows = []
    for row in real["rows"]:
        rows.append({
            "dataset": "real-cache writer",
            "gpu": "RTX 4090",
            "case": row["case"],
            "mode": row["mode"],
            "dtype": row["dtype"],
            "tokens": row["total_tokens"],
            "kv_heads": row["kv_heads"],
            "head_dim": row["head_dim"],
            "fragmented": row["fragmented_blocks"],
            "speedup": row["fused_vs_vllm_reshape_and_cache"],
            "baseline_us": row["vllm_then_reshape_and_cache_us"],
            "fused_us": row["fused_contract_rope_kv_write_us"],
            "correct": row["correct"] and row["k_correct"] and row["v_correct"],
            "oracle": row["oracle"],
        })
    for row in graph["rows"]:
        rows.append({
            "dataset": "CUDA Graph replay",
            "gpu": "RTX 3090",
            "case": row["case"],
            "mode": row["mode"],
            "dtype": row["dtype"],
            "tokens": row["total_tokens"],
            "kv_heads": row["kv_heads"],
            "head_dim": row["head_dim"],
            "fragmented": "fragmented" in row["case"],
            "speedup": row["speedup_graph"],
            "baseline_us": row["graph_baseline_us"],
            "fused_us": row["graph_fused_us"],
            "correct": all(row[key] for key in (
                "correct", "baseline_k_correct", "baseline_v_correct", "graph_correct",
                "graph_fused_k_correct", "graph_fused_v_correct",
                "graph_baseline_k_correct", "graph_baseline_v_correct",
            )),
            "oracle": row["oracle"],
        })
    return {
        "schema_version": 1,
        "title": "RoPE + KV-Cache Fusion Evidence",
        "validation": validation,
        "validation_errors": errors,
        "rows": rows,
        "provider_rows": provider["rows"],
        "flashinfer": {
            "classification": blocked.get("classification"),
            "reason": blocked.get("reason"),
            "benchmark_ran": blocked.get("benchmark_ran"),
        },
        "claim_boundary": "Selected-layout NVIDIA microbenchmarks, not an end-to-end serving result or production vLLM speedup.",
        "required_caveats": [
            "RTX 4090 prefill results use a local eager RoPE reference plus vLLM's real reshape_and_cache writer, so they are RoPE-provider contaminated.",
            "RTX 3090 CUDA Graph decode uses the real vLLM cache writer, but still uses a local tensor RoPE reference.",
            "No FlashInfer timing comparison was run; the API probe was only partially comparable.",
        ],
        "blocked_claims": ["full serving speedup", "production vLLM path win", "FlashInfer win", "official TritonBench result", "broad inference acceleration"],
    }


def html_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c")


def write_dashboard(data: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = ROOT / "dashboard_assets"
    template = (assets / "index.html").read_text(encoding="utf-8")
    (out_dir / "index.html").write_text(template.replace("__KERNEL_DATA__", html_json(data)), encoding="utf-8")
    (out_dir / "evidence-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name in ("app.js", "styles.css"):
        shutil.copyfile(assets / name, out_dir / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a static explorer for recorded KernelArena artifacts.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_dashboard(payload(args.root.resolve()), args.out_dir.resolve())
    print(f"Dashboard written to {args.out_dir.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
