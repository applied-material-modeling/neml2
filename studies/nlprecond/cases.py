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

"""Registry of the curated nonlinear-preconditioning testbed cases.

Each case is a self-contained copy of a regression scenario under
``tests/regression/solid_mechanics/``, edited only to expose the knobs the
harness varies (see the banner comment at the top of every ``model.i``). The
parents stay untouched -- they are pinned against a ``gold/result.pt``.

The registry carries the metadata the harness cannot infer from the input file:
how many nonlinear solves happen per driver step, and how big the implicit
system is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"


@dataclass(frozen=True)
class Case:
    """One testbed scenario."""

    #: Registry key; also the directory name under ``cases/``.
    name: str
    #: The regression scenario this was copied from, for provenance.
    parent: str
    #: Number of ``ImplicitUpdate`` solves the driver performs per time step.
    #: Used to attribute captured Newton solves back to driver steps; asserted
    #: against the captured count, so a wiring change fails loudly.
    solves_per_step: int
    #: Total scalar unknowns in the implicit system(s), for reporting.
    unknowns: int
    #: Which flow law ``flow_n`` feeds, for the report's theory column.
    flow_law: str
    #: One-line reason this case is in the testbed.
    why: str
    #: Whether the ``nopred`` arms can run. Set False for a case with a
    #: **sub-batched unknown** (e.g. a per-slip internal variable): with no
    #: predictor the unknown becomes an ordinary model input, and
    #: ``TransientDriver`` zero-fills unmatched inputs at *base* shape, dropping
    #: the sub-batch axis. The unknown vector then has fewer rows than the
    #: residual and the linear solve dies with ``linalg.solve: A must be batches
    #: of square matrices``. Setting this False turns that cryptic crash into a
    #: recorded ``skipped`` row.
    supports_nopred: bool = True
    #: Why ``supports_nopred`` is False, surfaced in the skip record.
    no_pred_blocker: str = ""

    @property
    def input_file(self) -> Path:
        return CASES_DIR / self.name / "model.i"


CASES: dict[str, Case] = {
    c.name: c
    for c in (
        Case(
            name="vp_isoharden",
            parent="tests/regression/solid_mechanics/viscoplasticity/isoharden",
            solves_per_step=1,
            unknowns=7,
            flow_law="perzyna",
            why="minimal reproducer; the fastest grid point",
        ),
        Case(
            name="vp_chaboche",
            parent="tests/regression/solid_mechanics/viscoplasticity/chaboche",
            solves_per_step=1,
            unknowns=19,
            flow_law="perzyna",
            why="nonlinearity spread across backstress groups",
        ),
        Case(
            name="cp_coupled",
            parent="tests/regression/solid_mechanics/crystal_plasticity/single_crystal_coupled",
            solves_per_step=1,
            unknowns=10,
            flow_law="powerlaw",
            why="canonical crystal plasticity; one fully-coupled group",
        ),
        Case(
            name="cp_decoupled",
            parent="tests/regression/solid_mechanics/crystal_plasticity/single_crystal_decoupled",
            solves_per_step=2,
            unknowns=10,
            flow_law="powerlaw",
            why="two sequentially-solved sub-systems (7 + 3)",
        ),
    )
}

#: Default stiffness exponent per flow law, matching the parent scenarios.
DEFAULT_FLOW_N = {"perzyna": 2.0, "powerlaw": 8.0}


def get(name: str) -> Case:
    """Look up a case by name, with a helpful error listing the valid keys."""
    try:
        return CASES[name]
    except KeyError:
        raise SystemExit(f"unknown case {name!r}; choose from {', '.join(CASES)}") from None


__all__ = ["CASES", "CASES_DIR", "DEFAULT_FLOW_N", "Case", "get"]
