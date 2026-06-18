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

"""Compile + run one benchmark scenario.

For each scenario under ``benchmark/<name>/model.i`` this script
``neml2-compile``-s the model to an AOTI ``.pt2`` package, emits a drop-in
HIT stub, loads the stub through the native factory, and runs its
``[Drivers]/driver`` once. The benchmark inputs use ``${nbatch}`` and
``${device}`` HIT substitution variables -- those are bound via ``pre`` args
to both the compile step and the runtime load.

Usage:
    python -m benchmark.run_benchmark <scenario> [--batch N] [--device cpu|cuda]
                                                 [--output-dir DIR]

Exit codes: 0 on success, 77 if ``--device cuda`` was requested but CUDA is
unavailable (CTest's skip code), non-zero on any other failure.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_benchmark")
    parser.add_argument(
        "scenario",
        help="Scenario name (directory under benchmark/ containing model.i).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help=(
            "Compile-time batch size for ${nbatch} (default 2). Drives the "
            "AOTI example shape -- the .pt2 is traced with the dynamic batch "
            "dim sized at this value."
        ),
    )
    parser.add_argument(
        "--run-batch",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional separate runtime batch size for ${nbatch}. When set, "
            "the model is COMPILED at --batch and then LOADED + run at this "
            "value -- exercising the .pt2's dynamic-batch generalisability "
            "(``torch.export``'s dynamic-Dim survives the trace, runtime "
            "must accept any batch). Defaults to --batch (compile == run, "
            "the historical behaviour). For sub-batched scenarios where "
            "${nbatch} ALSO drives a static sub-batch axis (e.g. mxpc), "
            "this must equal --batch -- the sub-batch dim is baked at "
            "compile time and the runtime tensor must match."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for ${device} and the AOTI artifact (default cpu).",
    )
    parser.add_argument(
        "--driver",
        default="driver",
        metavar="NAME",
        help=(
            "Name of the [Drivers] block to compile and run (default: 'driver'). "
            "The compiler reads the driver block's `model = '...'` field to figure "
            "out which model to compile -- there's no separate --model override to "
            "keep in sync."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write .pt2 + meta + stub. Defaults to a fresh tmpdir.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Instead of a single timed run, warm up then capture the driver run "
            "under torch.profiler (CPU+CUDA) and print the per-kernel breakdown "
            "(flat + grouped by input shape)."
        ),
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of driver runs to capture under the profiler (default 3).",
    )
    return parser


def run_one(
    scenario: str,
    batch: int,
    device: str,
    output_dir: Path,
    driver_name: str = "driver",
    run_batch: int | None = None,
) -> tuple[float, float]:
    """Compile + run one scenario; return (compile_seconds, run_seconds).

    ``batch`` is the compile-time example shape; ``run_batch`` (when
    different) is the runtime batch used to load the stub and drive the
    Newton solve. A ``run_batch != batch`` invocation exercises the
    .pt2's dynamic-batch generalisability -- the AOTI graph is
    compile-pinned to a specific (device, dtype) but the dynamic batch
    dim survives the trace, so any positive batch should run.
    """
    from neml2 import load_input
    from neml2.cli.aoti_compile import compile_and_emit_stub

    model_i = _BENCHMARK_DIR / scenario / "model.i"
    if not model_i.exists():
        raise FileNotFoundError(f"no model.i under benchmark/{scenario}/")

    compile_pre = (f"nbatch={batch}", f"device={device}")
    runtime_batch = run_batch if run_batch is not None else batch
    runtime_pre = (f"nbatch={runtime_batch}", f"device={device}")

    t0 = time.perf_counter()
    # Compile shape comes from each scenario's ``[Settings]
    # example_batch_shape`` block parameterised by ``${nbatch}``; passing
    # a CLI ``example_batch_shape`` here would be a UNIFORM override that
    # silently collapses per-input declarations (e.g. mxpc's
    # ``elastic_strain~1 = '(2; ${nbatch}:grain)'`` would become ``(B,)``
    # for every input, treating per-grain sub-batch as dynamic batch).
    # The Inductor autotune block-size hint is workload-dependent
    # (scpdecoup CUDA at B=8192 measured 22% faster with autotune
    # seeded by the runtime batch), so each scenario's [Settings] should
    # use ``${nbatch}`` to scale the trace example with the compile
    # batch.
    stub_path = compile_and_emit_stub(
        model_i,
        output_dir,
        driver=driver_name,
        device=device,
        pre=compile_pre,
    )
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    factory = load_input(stub_path, pre=runtime_pre)
    factory.get_driver(driver_name).run()
    t_run = time.perf_counter() - t0

    return t_compile, t_run


def profile_one(
    scenario: str,
    batch: int,
    device: str,
    output_dir: Path,
    driver_name: str = "driver",
    warmup: int = 3,
    profile_runs: int = 3,
) -> None:
    """Compile + load one scenario, then ``torch.profiler`` the driver run.

    Reuses the same compile/load path as :func:`run_one` (so the device /
    dtype defaults and the AOTI stub flow are identical), but instead of a
    single timed run it warms up, then captures ``profile_runs`` driver runs
    under the CPU+CUDA profiler and prints the per-kernel breakdown -- both a
    flat ``self_cuda_time_total`` ranking and one grouped by input shape, so
    a blow-up along a sub-batch axis (e.g. 12 slip systems materialised where
    a size-1 broadcast would do) shows up directly in the shape column.
    """
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from neml2 import load_input
    from neml2.cli.aoti_compile import compile_and_emit_stub

    model_i = _BENCHMARK_DIR / scenario / "model.i"
    if not model_i.exists():
        raise FileNotFoundError(f"no model.i under benchmark/{scenario}/")

    pre = (f"nbatch={batch}", f"device={device}")
    t0 = time.perf_counter()
    stub_path = compile_and_emit_stub(
        model_i, output_dir, driver=driver_name, device=device, pre=pre
    )
    print(f"[{scenario}] compile: {time.perf_counter() - t0:.1f}s", flush=True)

    driver = load_input(stub_path, pre=pre).get_driver(driver_name)

    for _ in range(warmup):
        driver.run()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(profile_runs):
        driver.run()
    if device == "cuda":
        torch.cuda.synchronize()
    print(
        f"[{scenario}] batch={batch} device={device} "
        f"avg run: {(time.perf_counter() - t0) / profile_runs * 1000:.1f} ms",
        flush=True,
    )

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(profile_runs):
            driver.run()
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"\n================ TOP KERNELS by {sort_key} ================")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    print(f"\n================ TOP by {sort_key}, GROUPED BY INPUT SHAPE ================")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=25))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but torch.cuda.is_available() is False; skipping.")
        return 77
    # Mirror sweep.py: pin process-wide defaults so the [Tensors] block's
    # Python expressions (``torch.tensor([...])``, ``torch.linspace(...)``)
    # build their initial conditions on the target device/dtype. Without
    # this, the AOTI runtime's input validator (correctly) refuses to
    # silently coerce a CPU-built ``orientation~1`` against a CUDA-pinned
    # model and the run fails before reporting any timing.
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)

    if args.output_dir is None:
        ctx = tempfile.TemporaryDirectory()
        out_dir = Path(ctx.name)
    else:
        ctx = None
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.profile:
            profile_one(
                args.scenario,
                args.batch,
                args.device,
                out_dir,
                driver_name=args.driver,
                profile_runs=args.profile_runs,
            )
            return 0
        t_compile, t_run = run_one(
            args.scenario,
            args.batch,
            args.device,
            out_dir,
            driver_name=args.driver,
            run_batch=args.run_batch,
        )
    finally:
        if ctx is not None:
            ctx.cleanup()

    run_batch_str = (
        f"batch={args.batch}"
        if args.run_batch is None or args.run_batch == args.batch
        else f"batch={args.batch} run_batch={args.run_batch}"
    )
    print(f"[{args.scenario}] {run_batch_str} device={args.device}")
    print(f"  compile: {t_compile:.2f}s")
    print(f"  run:     {t_run:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
