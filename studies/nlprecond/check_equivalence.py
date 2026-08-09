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

"""Check that a reformulated case solves the *same problem* as its parent.

A residual reformulation is only interesting if it is an exact rewrite. This
walks two cases through the identical load history and compares every shared
output at every step. A reformulation that converges faster but to a different
answer is a bug, not a result.

Usage::

    python -m studies.nlprecond.check_equivalence vp_isoharden vp_isoharden_inverted
"""

from __future__ import annotations

import argparse
import sys

import torch

from . import cases as case_registry
from .harness import BASE_NPOINT, GDOT_CUTOFF, GDOT_SEED, LS_ITERS


def _run(name: str, *, nsteps: int, nbatch: int, flow_n: float, max_its: int) -> dict:
    from neml2 import load_input, log  # noqa: PLC0415

    # Importing `.harness` turns the per-iteration newton debug stream on; this
    # tool only cares about the converged answer, so swallow it.
    log.set_sink(lambda _level, _line: None)

    case = case_registry.get(name)
    tfrac = nsteps / (BASE_NPOINT - 1)
    f = load_input(
        case.input_file,
        pre=[
            f"nbatch={nbatch}",
            f"npoint={nsteps + 1}",
            f"tfrac={tfrac!r}",
            f"flow_n={flow_n}",
            f"ls_iters={LS_ITERS[False]}",
            f"max_its={max_its}",
            f"gdot_cutoff={GDOT_CUTOFF!r}",
            f"gdot_seed={GDOT_SEED!r}",
        ],
    )
    driver = f.get_driver("driver")
    driver.run()
    return driver.result()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", choices=list(case_registry.CASES))
    ap.add_argument("candidate", choices=list(case_registry.CASES))
    ap.add_argument("--nsteps", type=int, default=6)
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("--flow-n", type=float, default=2.0)
    ap.add_argument("--max-its", type=int, default=50)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    kw = dict(nsteps=args.nsteps, nbatch=args.nbatch, flow_n=args.flow_n, max_its=args.max_its)
    ref = _run(args.reference, **kw)  # type: ignore[arg-type]
    cand = _run(args.candidate, **kw)  # type: ignore[arg-type]

    # Only outputs; inputs are prescribed identically by construction. The
    # candidate carries extra unknowns (e.g. flow_rate) the reference does not,
    # so compare the intersection and report what was skipped.
    shared = sorted(k for k in ref if k in cand and k.startswith("output."))
    only_cand = sorted(k for k in cand if k not in ref and k.startswith("output."))
    if not shared:
        raise SystemExit("no shared outputs to compare -- are these the same scenario?")

    worst_key, worst = "", 0.0
    n_bad = 0
    for k in shared:
        a, b = ref[k], cand[k]
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            n_bad += 1
            continue
        denom = a.abs().max().clamp_min(1.0)
        rel = ((a - b).abs().max() / denom).item()
        if rel > worst:
            worst_key, worst = k, rel
        if not torch.allclose(a, b, rtol=args.rtol, atol=args.atol):
            n_bad += 1
            if n_bad <= 5:
                print(f"  MISMATCH {k}: max rel {rel:.3e}")

    print(f"{args.reference}  vs  {args.candidate}")
    print(f"  nsteps={args.nsteps} nbatch={args.nbatch} flow_n={args.flow_n}")
    print(f"  compared {len(shared)} shared output series; {n_bad} outside tolerance")
    print(f"  worst relative difference: {worst:.3e}  ({worst_key})")
    if only_cand:
        uniq = sorted({k.split(".", 2)[-1] for k in only_cand})
        print(f"  candidate-only outputs (not compared): {', '.join(uniq)}")
    if n_bad:
        print("  RESULT: NOT equivalent")
        return 1
    print(f"  RESULT: equivalent within rtol={args.rtol:g} atol={args.atol:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
