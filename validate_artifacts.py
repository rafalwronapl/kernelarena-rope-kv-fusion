from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


MANIFEST_ROW = re.compile(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$")


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".zip" or path.name.endswith(".tar.gz"):
        return data
    return data.replace(b"\r\n", b"\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def manifest_rows(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_ROW.match(line)
        if match:
            rows.append((match.group(1), int(match.group(2)), match.group(3)))
    if not rows:
        raise ValueError("ARTIFACT_MANIFEST.md contains no parseable rows")
    return rows


def check_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    for relative, expected_size, expected_hash in manifest_rows(root / "ARTIFACT_MANIFEST.md"):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"manifest path escapes repository: {relative}")
        elif not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        else:
            if len(canonical_bytes(path)) != expected_size:
                errors.append(f"size mismatch: {relative}")
            if sha256(path) != expected_hash:
                errors.append(f"sha256 mismatch: {relative}")
    return errors


def update_manifest(root: Path) -> None:
    path = root / "ARTIFACT_MANIFEST.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        match = MANIFEST_ROW.match(line)
        if not match:
            updated.append(line)
            continue
        relative = match.group(1)
        file_path = (root / relative).resolve()
        if not file_path.is_relative_to(root.resolve()) or not file_path.is_file():
            raise ValueError(f"Cannot update missing or unsafe manifest path: {relative}")
        updated.append(f"| `{relative}` | {len(canonical_bytes(file_path))} | `{sha256(file_path)}` |")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def validate_benchmarks(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    real = load_json(root / "artifacts" / "real_vllm_4090_rope_real_vllm_contract_summary.json")
    real_rows = real.get("rows", [])
    if len(real_rows) != 16:
        errors.append(f"RTX 4090 real-cache summary expected 16 rows, found {len(real_rows)}")
    if real.get("oracle") != "real_vllm_oracle":
        errors.append("RTX 4090 real-cache summary has the wrong oracle")
    for row in real_rows:
        if not all(row.get(key) is True for key in ("correct", "k_correct", "v_correct")):
            errors.append(f"correctness failure in {row.get('case', 'unknown 4090 row')}")
        if not isinstance(row.get("fused_vs_vllm_reshape_and_cache"), (int, float)) or row["fused_vs_vllm_reshape_and_cache"] <= 0:
            errors.append(f"missing speedup in {row.get('case', 'unknown 4090 row')}")

    graph = load_json(root / "artifacts" / "real_vllm_cudagraph_decode_summary_3090_postreplay.json")
    graph_rows = graph.get("rows", [])
    correctness_keys = (
        "correct", "k_correct", "v_correct", "baseline_k_correct", "baseline_v_correct",
        "graph_correct", "graph_fused_k_correct", "graph_fused_v_correct",
        "graph_baseline_k_correct", "graph_baseline_v_correct",
    )
    if len(graph_rows) != 8:
        errors.append(f"RTX 3090 graph summary expected 8 rows, found {len(graph_rows)}")
    for row in graph_rows:
        if not all(row.get(key) is True for key in correctness_keys):
            errors.append(f"graph correctness failure in {row.get('case', 'unknown 3090 row')}")
        if "reshape_and_cache" not in str(row.get("baseline", "")):
            errors.append(f"real cache-writer baseline missing in {row.get('case', 'unknown 3090 row')}")
        if not isinstance(row.get("speedup_graph"), (int, float)) or row["speedup_graph"] <= 0:
            errors.append(f"graph speedup missing in {row.get('case', 'unknown 3090 row')}")

    provider = load_json(root / "artifacts" / "rope_provider_split_3090.json")
    provider_rows = provider.get("rows", [])
    if len(provider_rows) != 16:
        errors.append(f"provider split expected 16 rows, found {len(provider_rows)}")
    if any(not isinstance(row.get("compiled_vs_local"), (int, float)) for row in provider_rows):
        errors.append("provider split contains a row without compiled_vs_local")

    real_speedups = [float(row["fused_vs_vllm_reshape_and_cache"]) for row in real_rows if isinstance(row.get("fused_vs_vllm_reshape_and_cache"), (int, float))]
    graph_speedups = [float(row["speedup_graph"]) for row in graph_rows if isinstance(row.get("speedup_graph"), (int, float))]
    manifest_path = root / "ARTIFACT_MANIFEST.md"
    summary = {
        "status": "pass" if not errors else "fail",
        "manifest_entries": len(manifest_rows(manifest_path)) if manifest_path.is_file() else 0,
        "rtx4090": {
            "rows": len(real_rows),
            "correct": sum(row.get("correct") is True for row in real_rows),
            "speedup": stats(real_speedups) if real_speedups else {},
        },
        "rtx3090_graph": {
            "rows": len(graph_rows),
            "correct": sum(row.get("graph_correct") is True for row in graph_rows),
            "speedup": stats(graph_speedups) if graph_speedups else {},
        },
        "provider_split_rows": len(provider_rows),
    }
    return summary, errors


def validate(root: Path) -> tuple[dict[str, Any], list[str]]:
    summary, errors = validate_benchmarks(root)
    errors = check_manifest(root) + errors
    summary["status"] = "pass" if not errors else "fail"
    summary["errors"] = errors
    return summary, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate recorded KernelArena artifacts without requiring a GPU.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.update_manifest:
        update_manifest(root)
    summary, errors = validate(root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
