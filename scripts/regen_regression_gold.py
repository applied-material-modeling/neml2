#!/usr/bin/env python3

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

"""Regenerate the gold ``.pt`` reference files under ``tests/regression/``.

Each scenario has a paired ``model.i`` + ``gold/result.pt``. Running this
script:

1. Walks every ``tests/regression/**/model.i``.
2. For each input file, locates the ``[Drivers]`` sub-block whose paired
   ``TransientRegression`` block points at ``gold/result.pt`` (the
   convention used everywhere in the suite).
3. Runs the underlying ``TransientDriver`` standalone.
4. Overwrites ``gold/result.pt`` via :meth:`TransientDriver.save_gold` —
   the new plain-``torch.save`` dict format.

Run from the project root:

    python scripts/regen_regression_gold.py [--dry-run] [<path-filter>]

Without ``<path-filter>``, every scenario is regenerated. With one or
more filters, only inputs whose relative path contains *any* of them are
regenerated (substring match; case-sensitive).

The script is idempotent in the sense that running it again on an
already-regenerated suite produces the same content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nmhit


def _find_transient_driver_name(input_file: Path) -> str | None:
    """Find the [Drivers] block whose type is TransientDriver."""
    root = nmhit.parse_file(input_file, [], [])
    for top in root.children(nmhit.NodeType.Section):
        if top.path() != "Drivers":
            continue
        for child in top.children(nmhit.NodeType.Section):
            if child.param_optional_str("type", "") == "TransientDriver":
                return child.path().rsplit("/", 1)[-1]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filters", nargs="*", help="substrings; regen only matching paths")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be regenerated, don't write"
    )
    parser.add_argument(
        "--regression-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "regression",
        help="root of the regression suite (default: tests/regression)",
    )
    args = parser.parse_args(argv)

    # Import neml2 after argparse so --help doesn't pay the import cost.
    import torch  # noqa: PLC0415

    from neml2 import load_input  # noqa: PLC0415

    # Match the regression suite's tests/conftest.py — the v2 C++ goldens were
    # written in float64, and the in-process loader assumes everything stays
    # in float64 (tolerances are tuned to that precision).
    torch.set_default_dtype(torch.float64)

    # Make the test-only _fixtures package importable so @register_native fires
    # for TabulatedPolynomialModel / TorchScriptFlowRate. Mirrors
    # tests/regression/conftest.py's side-effect import — without this, the
    # ``misc/polynomial`` and ``misc/torch_script`` scenarios fail to load.
    tests_dir = args.regression_dir.parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    try:
        import _fixtures  # noqa: F401, PLC0415  (side-effect: @register_native)
    except ImportError:
        pass  # tolerate older layouts that don't have the _fixtures package

    # A scenario is any input file (*.i) that has a gold/result.pt next to it.
    # Most are named model.i but a few use scenario-specific names (advection.i,
    # small_deformation.i, ...) — pair by directory, not by filename.
    inputs = sorted(
        p for p in args.regression_dir.rglob("*.i") if (p.parent / "gold" / "result.pt").exists()
    )
    if args.filters:
        rel = [p.relative_to(args.regression_dir) for p in inputs]
        inputs = [
            p for p, r in zip(inputs, rel, strict=True) if any(f in str(r) for f in args.filters)
        ]
    if not inputs:
        print("No matching inputs found.", file=sys.stderr)
        return 1

    failed: list[tuple[Path, str]] = []
    for input_file in inputs:
        rel = input_file.relative_to(args.regression_dir)
        gold_path = input_file.parent / "gold" / "result.pt"
        if not gold_path.parent.exists():
            print(f"SKIP {rel}: no gold/ directory", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"DRY  {rel}  ->  {gold_path.relative_to(args.regression_dir)}")
            continue
        try:
            driver_name = _find_transient_driver_name(input_file)
            if driver_name is None:
                print(f"SKIP {rel}: no [Drivers] of type TransientDriver", file=sys.stderr)
                continue
            factory = load_input(input_file)
            driver = factory.get_driver(driver_name)
            driver.run()
            driver.save_gold(gold_path)
            print(f"OK   {rel}  ->  {gold_path.name}")
        except Exception as exc:  # noqa: BLE001 — keep going through failures
            failed.append((rel, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)} scenario(s) failed:", file=sys.stderr)
        for rel, msg in failed:
            print(f"  {rel}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
