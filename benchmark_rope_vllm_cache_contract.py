from __future__ import annotations
# Tested stack: vLLM 0.9.2, torch 2.7.0+cu126, triton 3.3.0, CUDA 12.6, Python 3.11.
# vLLM internal APIs (RotaryEmbedding, forward_cuda) are version-sensitive.
# Other vLLM versions may require adjustments to the import paths and call signatures.

import argparse
from contextlib import nullcontext
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
    try:
        import torch._inductor as _torch_inductor
        import torch._inductor.config as _torch_inductor_config

        if not hasattr(_torch_inductor, "config"):
            _torch_inductor.config = _torch_inductor_config
    except Exception:
        pass
    import triton
    import triton.language as tl

    HAS_TORCH_TRITON = True
except ModuleNotFoundError:
    torch = None
    triton = None
    tl = None
    HAS_TORCH_TRITON = False

try:
    try:
        from vllm.config.vllm import VllmConfig, set_current_vllm_config
    except ModuleNotFoundError:
        try:
            from vllm.config import VllmConfig, set_current_vllm_config
        except (ImportError, ModuleNotFoundError):
            VllmConfig = None
            set_current_vllm_config = None
    try:
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    except ModuleNotFoundError:
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    HAS_VLLM = True
except ModuleNotFoundError:
    VllmConfig = None
    set_current_vllm_config = None
    RotaryEmbedding = object
    HAS_VLLM = False


REPO_ROOT = Path(__file__).resolve().parent

DTYPES = {"fp16": None, "bf16": None}
if HAS_TORCH_TRITON:
    DTYPES = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }


@dataclass(frozen=True)
class ContractCase:
    name: str
    mode: str
    dtype_name: str
    seq_lens: tuple[int, ...]
    kv_heads: int
    head_dim: int
    block_size: int
    triton_block: int
    fragmented_blocks: bool
    position_offsets: tuple[int, ...]


def build_cases() -> list[ContractCase]:
    prefill_lens = (128, 1024, 4096, 8192)
    prefill_offsets = (0, 512, 4096, 12288)
    cases: list[ContractCase] = []
    for dtype in ("fp16", "bf16"):
        for kv_heads in (8, 16):
            cases.append(
                ContractCase(
                    name=f"{dtype}_prefill_contract_kv{kv_heads}_d128_b16",
                    mode="prefill",
                    dtype_name=dtype,
                    seq_lens=prefill_lens,
                    kv_heads=kv_heads,
                    head_dim=128,
                    block_size=16,
                    triton_block=256,
                    fragmented_blocks=False,
                    position_offsets=prefill_offsets,
                )
            )
            cases.append(
                ContractCase(
                    name=f"{dtype}_prefill_fragmented_kv{kv_heads}_d128_b16",
                    mode="prefill",
                    dtype_name=dtype,
                    seq_lens=prefill_lens,
                    kv_heads=kv_heads,
                    head_dim=128,
                    block_size=16,
                    triton_block=256,
                    fragmented_blocks=True,
                    position_offsets=prefill_offsets,
                )
            )

    for dtype in ("fp16", "bf16"):
        for batch in (1, 8, 32, 128):
            existing = tuple(127 + 17 * i for i in range(batch))
            offsets = tuple(3 * i for i in range(batch))
            cases.append(
                ContractCase(
                    name=f"{dtype}_decode_contract_batch{batch}_kv8_d128_b16",
                    mode="decode",
                    dtype_name=dtype,
                    seq_lens=existing,
                    kv_heads=8,
                    head_dim=128,
                    block_size=16,
                    triton_block=128,
                    fragmented_blocks=True,
                    position_offsets=offsets,
                )
            )
    return cases


if HAS_TORCH_TRITON:

    @triton.jit
    def _contract_write_kernel(
        k_ptr,
        v_ptr,
        slot_ptr,
        k_cache_ptr,
        v_cache_ptr,
        total_elems,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elems
        token_stride = heads * head_dim
        token = offsets // token_stride
        rem = offsets - token * token_stride
        slot = tl.load(slot_ptr + token, mask=mask, other=0)
        cache_offset = slot * token_stride + rem
        k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offset, k, mask=mask)
        tl.store(v_cache_ptr + cache_offset, v, mask=mask)

    @triton.jit
    def _contract_rope_kv_write_kernel(
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        slot_ptr,
        k_cache_ptr,
        v_cache_ptr,
        total_elems,
        heads: tl.constexpr,
        half_dim: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elems
        token_stride = heads * head_dim
        token = offsets // token_stride
        rem = offsets - token * token_stride
        dim = rem % head_dim
        dim_pair = dim // 2
        pair_base = offsets - (dim % 2)
        slot = tl.load(slot_ptr + token, mask=mask, other=0)
        cache_offset = slot * token_stride + rem

        k0 = tl.load(k_ptr + pair_base, mask=mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + pair_base + 1, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + token * half_dim + dim_pair, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + token * half_dim + dim_pair, mask=mask, other=0.0).to(tl.float32)
        even_out = k0 * c - k1 * s
        odd_out = k0 * s + k1 * c
        kout = tl.where((dim % 2) == 1, odd_out, even_out)
        vv = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offset, kout, mask=mask)
        tl.store(v_cache_ptr + cache_offset, vv, mask=mask)


def rope_ref(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = k[..., 0::2]
    odd = k[..., 1::2]
    out = torch.empty_like(k)
    out[..., 0::2] = even * cos[:, None, :] - odd * sin[:, None, :]
    out[..., 1::2] = even * sin[:, None, :] + odd * cos[:, None, :]
    return out


def cuda_time_us(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def get_cos_sin_cache(emb: RotaryEmbedding, query: torch.Tensor) -> torch.Tensor:
    if hasattr(emb, "_match_cos_sin_cache_dtype"):
        return emb._match_cos_sin_cache_dtype(query)
    return emb.cos_sin_cache.to(device=query.device, dtype=query.dtype)


def vllm_config_context():
    if VllmConfig is None or set_current_vllm_config is None:
        return nullcontext()
    return set_current_vllm_config(VllmConfig())


def format_ratio(value: float | None) -> str:
    if value is None:
        return "blocked"
    return f"{value:.3f}x"


def contract_write(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block: int,
) -> None:
    total_elems = k.numel()
    _, heads, head_dim = k.shape
    block_size = k_cache.shape[1]
    grid = (triton.cdiv(total_elems, block),)
    _contract_write_kernel[grid](k, v, slots, k_cache, v_cache, total_elems, heads, head_dim, block_size, BLOCK=block)


def fused_contract_write(
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block: int,
) -> None:
    total_elems = k.numel()
    _, heads, head_dim = k.shape
    half_dim = head_dim // 2
    block_size = k_cache.shape[1]
    grid = (triton.cdiv(total_elems, block),)
    _contract_rope_kv_write_kernel[grid](
        k, v, cos, sin, slots, k_cache, v_cache, total_elems, heads, half_dim, head_dim, block_size, BLOCK=block
    )


def _permute_blocks(num_blocks: int, fragmented: bool) -> list[int]:
    blocks = list(range(num_blocks))
    if fragmented and num_blocks > 1:
        # Deterministic permutation without requiring random state on host/GPU.
        blocks = sorted(blocks, key=lambda item: ((item * 1103515245 + 12345) % 2**31))
    return blocks


def build_contract(case: ContractCase) -> dict[str, object]:
    if len(case.seq_lens) != len(case.position_offsets):
        raise ValueError("seq_lens and position_offsets must have the same length")

    if case.mode == "prefill":
        tokens_per_seq = list(case.seq_lens)
        write_positions = [list(range(length)) for length in case.seq_lens]
        logical_blocks_per_seq = [(length + case.block_size - 1) // case.block_size for length in case.seq_lens]
    elif case.mode == "decode":
        tokens_per_seq = [1 for _ in case.seq_lens]
        write_positions = [[length] for length in case.seq_lens]
        logical_blocks_per_seq = [(length + 1 + case.block_size - 1) // case.block_size for length in case.seq_lens]
    else:
        raise ValueError(f"Unknown mode: {case.mode}")

    total_logical_blocks = sum(logical_blocks_per_seq)
    physical_blocks = _permute_blocks(total_logical_blocks, case.fragmented_blocks)

    block_table: list[list[int]] = []
    cursor = 0
    for nblocks in logical_blocks_per_seq:
        block_table.append(physical_blocks[cursor : cursor + nblocks])
        cursor += nblocks

    slots: list[int] = []
    positions: list[int] = []
    sequence_ids: list[int] = []
    token_indices: list[int] = []
    for seq_id, seq_positions in enumerate(write_positions):
        for local_token_idx, pos in enumerate(seq_positions):
            logical_block = pos // case.block_size
            offset = pos % case.block_size
            slot = block_table[seq_id][logical_block] * case.block_size + offset
            slots.append(slot)
            positions.append(case.position_offsets[seq_id] + pos)
            sequence_ids.append(seq_id)
            token_indices.append(local_token_idx)

    max_position = max(positions) + 1 if positions else 1
    total_tokens = len(slots)
    return {
        "mode": case.mode,
        "tokens_per_seq": tokens_per_seq,
        "seq_lens": list(case.seq_lens),
        "position_offsets": list(case.position_offsets),
        "positions": positions,
        "slots": slots,
        "sequence_ids": sequence_ids,
        "token_indices": token_indices,
        "block_table": block_table,
        "block_size": case.block_size,
        "num_physical_blocks": total_logical_blocks,
        "total_tokens": total_tokens,
        "max_position": max_position,
        "fragmented_blocks": case.fragmented_blocks,
        "oracle": "contract_oracle",
        "layout": {
            "cache_shape": [total_logical_blocks, case.block_size, case.kv_heads, case.head_dim],
            "slot_formula": "cache[slot // block_size, slot % block_size, kv_head, head_dim]",
            "assumes_actual_vllm_cache_writer": False,
        },
    }


def summarize_contract(contract: dict[str, object]) -> dict[str, object]:
    block_table = contract["block_table"]
    return {
        "mode": contract["mode"],
        "tokens_per_seq": contract["tokens_per_seq"],
        "seq_lens": contract["seq_lens"],
        "position_offsets": contract["position_offsets"],
        "block_size": contract["block_size"],
        "num_physical_blocks": contract["num_physical_blocks"],
        "total_tokens": contract["total_tokens"],
        "max_position": contract["max_position"],
        "fragmented_blocks": contract["fragmented_blocks"],
        "oracle": contract["oracle"],
        "layout": contract["layout"],
        "blocks_per_seq": [len(seq_blocks) for seq_blocks in block_table],
        "sample_block_table": [seq_blocks[:8] for seq_blocks in block_table],
        "sample_slots": contract["slots"][:16],
        "sample_positions": contract["positions"][:16],
        "sample_sequence_ids": contract["sequence_ids"][:16],
        "sample_token_indices": contract["token_indices"][:16],
    }


def dry_run(output: Path) -> None:
    cases = build_cases()
    dry_cases = []
    for case in cases:
        contract = build_contract(case)
        dry_cases.append(
            {
                "case": case.__dict__,
                "contract_summary": summarize_contract(contract),
            }
        )
    payload = {
        "benchmark": "rope_vllm_cache_contract",
        "dry_run": True,
        "planned_cases": len(cases),
        "cases": dry_cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "planned_cases": len(cases)}, indent=2))


def run_case(case: ContractCase, warmup: int, repeats: int) -> dict[str, object]:
    dtype = DTYPES[case.dtype_name]
    contract = build_contract(case)
    total_tokens = int(contract["total_tokens"])
    max_position = int(contract["max_position"])
    slots = torch.tensor(contract["slots"], device="cuda", dtype=torch.long)
    positions = torch.tensor(contract["positions"], device="cuda", dtype=torch.long)

    k = torch.randn(total_tokens, case.kv_heads, case.head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cache_shape = contract["layout"]["cache_shape"]
    k_cache = torch.empty(*cache_shape, device="cuda", dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    emb = RotaryEmbedding(case.head_dim, case.head_dim, max_position, 10000, False, dtype).cuda()
    cos_sin = get_cos_sin_cache(emb, k).index_select(0, positions)
    cos, sin = [part.contiguous() for part in cos_sin.chunk(2, dim=-1)]
    expected_k = rope_ref(k, cos, sin)

    fused_contract_write(k, v, cos, sin, slots, k_cache, v_cache, case.triton_block)
    torch.cuda.synchronize()
    flat_k_cache = k_cache.reshape(-1, case.kv_heads, case.head_dim)
    flat_v_cache = v_cache.reshape_as(flat_k_cache)
    k_correct = torch.allclose(flat_k_cache[slots].float(), expected_k.float(), atol=2e-2, rtol=2e-2)
    v_correct = torch.allclose(flat_v_cache[slots].float(), v.float(), atol=2e-2, rtol=2e-2)
    k_diff = (flat_k_cache[slots].float() - expected_k.float()).abs().max().item()
    v_diff = (flat_v_cache[slots].float() - v.float()).abs().max().item()

    compiled_rope = torch.compile(rope_ref, mode="max-autotune-no-cudagraphs")
    _ = compiled_rope(k, cos, sin)
    torch.cuda.synchronize()

    def inductor_then_write():
        tmp = compiled_rope(k, cos, sin)
        contract_write(tmp, v, slots, k_cache, v_cache, case.triton_block)

    def vllm_then_write():
        tmp = k.clone()
        emb.forward_cuda(positions, tmp, None)
        contract_write(tmp, v, slots, k_cache, v_cache, case.triton_block)

    def fused():
        fused_contract_write(k, v, cos, sin, slots, k_cache, v_cache, case.triton_block)

    inductor_us = cuda_time_us(inductor_then_write, warmup, repeats)
    vllm_error = None
    try:
        vllm_us = cuda_time_us(vllm_then_write, warmup, repeats)
    except RuntimeError as exc:
        if "undefined Tensor" not in str(exc):
            raise
        vllm_us = None
        vllm_error = "vLLM key-only RoPE is unavailable in this installed vLLM API"
        torch.cuda.synchronize()
    fused_us = cuda_time_us(fused, warmup, repeats)
    fused_vs_vllm = None if vllm_us is None else round(vllm_us / fused_us, 4)

    return {
        "case": case.name,
        "mode": case.mode,
        "dtype": case.dtype_name,
        "seq_lens": list(case.seq_lens),
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "block_size": case.block_size,
        "triton_block": case.triton_block,
        "fragmented_blocks": case.fragmented_blocks,
        "total_tokens": total_tokens,
        "num_physical_blocks": contract["num_physical_blocks"],
        "cache_shape": contract["layout"]["cache_shape"],
        "oracle": contract["oracle"],
        "correct": bool(k_correct and v_correct),
        "k_correct": bool(k_correct),
        "v_correct": bool(v_correct),
        "k_max_abs_diff": k_diff,
        "v_max_abs_diff": v_diff,
        "inductor_then_contract_write_us": round(inductor_us, 3),
        "vllm_then_contract_write_us": None if vllm_us is None else round(vllm_us, 3),
        "vllm_error": vllm_error,
        "fused_contract_rope_kv_write_us": round(fused_us, 3),
        "fused_vs_inductor_contract_write": round(inductor_us / fused_us, 4),
        "fused_vs_vllm_contract_write": fused_vs_vllm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RoPE + KV write against a vLLM-style cache contract.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "rope_vllm_cache_contract.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if args.dry_run:
        dry_run(output)
        return
    if not HAS_TORCH_TRITON:
        raise RuntimeError("torch and triton are required. Use --dry-run for local contract planning.")
    if not HAS_VLLM:
        raise RuntimeError("vLLM is required for the runtime benchmark. Use --dry-run for local contract planning.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --dry-run for local contract planning.")

    torch.manual_seed(0)
    rows = []
    cases = build_cases()
    started = time.time()
    with vllm_config_context():
        for idx, case in enumerate(cases, 1):
            row = run_case(case, args.warmup, args.repeats)
            rows.append(row)
            print(
                f"PROGRESS {idx}/{len(cases)} {idx / len(cases):.0%} "
                f"{case.name} correct={row['correct']} "
                f"vs_vllm={format_ratio(row['fused_vs_vllm_contract_write'])} "
                f"oracle={row['oracle']}",
                flush=True,
            )

    payload = {
        "benchmark": "rope_vllm_cache_contract",
        "oracle": "contract_oracle",
        "elapsed_sec": round(time.time() - started, 3),
        "rows": rows,
        "non_claims": [
            "not a full vLLM serving benchmark",
            "not production vLLM cache writer unless oracle says real_vllm_oracle",
            "not an end-to-end inference speedup claim",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
