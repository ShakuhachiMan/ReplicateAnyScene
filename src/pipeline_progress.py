"""Pipeline progress logging and optional CUDA cache hints."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


def log(msg: str) -> None:
    print(msg, flush=True)


def cuda_mem_gib() -> str:
    if not torch.cuda.is_available():
        return "CUDA: N/A"
    alloc = torch.cuda.memory_allocated() / (1024**3)
    rsv = torch.cuda.memory_reserved() / (1024**3)
    return f"CUDA GiB alloc={alloc:.2f} reserved={rsv:.2f}"


def cuda_cache_clear(note: str = "") -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if note:
        log(f"[ReplicateAnyScene] empty_cache ({note}) | {cuda_mem_gib()}")


@contextmanager
def timed_stage(name: str, extra: str = "") -> Iterator[None]:
    suffix = f" {extra}" if extra else ""
    log(f"[ReplicateAnyScene] >>> 開始: {name}{suffix} | {cuda_mem_gib()}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"[ReplicateAnyScene] <<< 結束: {name} ({dt:.1f}s) | {cuda_mem_gib()}")


def frame_progress(
    step: int,
    total: int,
    label: str,
    *,
    every: int | None = None,
) -> None:
    """Print at first frame, last frame, and every `every` steps (default ~8 prints total)."""
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 8)
    if step == 0 or step + 1 == total or (step + 1) % every == 0:
        pct = 100.0 * (step + 1) / total
        log(
            f"[ReplicateAnyScene] {label} {step + 1}/{total} ({pct:.1f}%) | {cuda_mem_gib()}"
        )
