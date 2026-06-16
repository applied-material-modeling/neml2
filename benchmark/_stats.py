# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Timing harness + statistics for the benchmark sweep.

Pure stdlib + numpy + torch. No NEML2 imports -- this module also runs
under v2's Python env during the one-time v2 baseline.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def time_callable(fn: Callable[[], Any], *, device: str, n_warmup: int, n_runs: int) -> np.ndarray:
    """Run ``fn`` ``n_warmup + n_runs`` times; return per-call wall-time in seconds.

    For ``device='cuda'``, synchronize before each timer boundary so the
    measurement reflects on-device completion rather than kernel launch
    queueing. Warmup runs are discarded (kernel autotuning, AOTI loader
    state, allocator warmup all settle in the first few calls).
    """
    cuda_sync = device == "cuda" and torch.cuda.is_available()
    for _ in range(n_warmup):
        fn()
        if cuda_sync:
            torch.cuda.synchronize()
    times = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        if cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync:
            torch.cuda.synchronize()
        times[i] = time.perf_counter() - t0
    return times


def compute_stats(times_s: np.ndarray) -> dict[str, float | int]:
    """Reduce per-call seconds to a compact stats dict in milliseconds."""
    t_ms = times_s * 1000.0
    return {
        "n_runs": int(t_ms.size),
        "mean_ms": float(t_ms.mean()),
        "median_ms": float(np.median(t_ms)),
        "std_ms": float(t_ms.std(ddof=1)) if t_ms.size > 1 else 0.0,
        "min_ms": float(t_ms.min()),
        "max_ms": float(t_ms.max()),
        "p10_ms": float(np.percentile(t_ms, 10)),
        "p90_ms": float(np.percentile(t_ms, 90)),
    }


def reset_memory_peak(device: str) -> None:
    """Reset the per-batch CUDA peak counter so the next read covers this batch only.

    CPU has no per-batch reset (Linux's ``ru_maxrss`` is a process-lifetime
    high-water mark); we instead report current RSS via :func:`record_memory`
    after each batch and let the reader compare trajectories.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def record_memory(device: str) -> dict[str, float]:
    """Snapshot memory after a batch completes.

    Returns a dict with two columns:
    * ``cpu_rss_mb`` -- current resident-set size of this process via
      ``psutil`` (always populated).
    * ``cuda_peak_mb`` -- ``torch.cuda.max_memory_allocated()`` since the
      last :func:`reset_memory_peak`. ``nan`` when ``device != 'cuda'`` or
      when CUDA is unavailable.

    Designed to be called at the end of a per-batch measurement window so
    the trajectory across batch sizes reveals scaling behaviour. Cheap --
    no device sync beyond what ``max_memory_allocated`` already does
    (CUDA driver query, microsecond scale).
    """
    import psutil  # noqa: PLC0415

    rss_bytes = psutil.Process().memory_info().rss
    cpu_rss_mb = float(rss_bytes) / (1024.0 * 1024.0)
    if device == "cuda" and torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        cuda_peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    else:
        cuda_peak_mb = float("nan")
    return {"cpu_rss_mb": cpu_rss_mb, "cuda_peak_mb": cuda_peak_mb}


def loglog_slope(batches: list[int], medians_ms: list[float]) -> float:
    """Slope of log(t) vs log(B) over the given history.

    A slope of 0 = flat (batch parallelism still scaling), 1 = linear
    (device bandwidth saturated, each extra batch element costs its own
    time). Used as the stopping signal in the adaptive sweep.

    Uses an ordinary least-squares fit when len >= 2; returns ``nan`` when
    insufficient data.
    """
    if len(batches) < 2 or len(batches) != len(medians_ms):
        return float("nan")
    xs = np.log(np.asarray(batches, dtype=np.float64))
    ys = np.log(np.asarray(medians_ms, dtype=np.float64))
    xm = xs.mean()
    ym = ys.mean()
    denom = ((xs - xm) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(((xs - xm) * (ys - ym)).sum() / denom)


__all__ = [
    "time_callable",
    "compute_stats",
    "loglog_slope",
    "reset_memory_peak",
    "record_memory",
    "math",
]
