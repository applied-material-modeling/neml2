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

"""Matrix-free restarted GMRES(m) linear solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, option

from ._krylov import _KrylovSolver

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


@register_neml2_object("GMRES")
class GMRES(_KrylovSolver):
    """Matrix-free restarted GMRES(m) linear solver for the implicit Newton step.

    Never assembles the Jacobian: each inner iteration applies ``J.v`` (the
    compiled/eager ``Matvec``). Best for the well-conditioned near-identity
    systems of implicit backward-Euler updates, where a handful of matvecs beat
    an O(N^3) dense factorization. Configure it as a ``Newton`` ``linear_solver``.
    """

    method = "gmres"

    hit = HitSchema(
        option("restart", int, "GMRES(m) restart width (Krylov subspace size)", default=40),
        option(
            "max_its",
            int,
            "Maximum inner (Krylov) iterations per Newton step",
            default=1000,
        ),
        option(
            "abs_tol",
            float,
            "Absolute inner-residual tolerance (0 disables the absolute test)",
            default=0.0,
        ),
        option(
            "rel_tol",
            float,
            "Relative inner-residual tolerance. Loose is fine (and faster): the "
            "outer Newton re-solves each step, so an inexact inner solve still "
            "converges -- tighten only for stiff/ill-conditioned systems",
            default=1.0e-4,
        ),
        option(
            "preconditioner",
            str,
            "Preconditioner: none | jacobi | block_jacobi | full",
            default="none",
        ),
        option(
            "cache_strategy",
            str,
            "Preconditioner rebuild policy across Newton iterations: "
            "none (rebuild each step) | chord (build once, reuse) | "
            "max_its (rebuild when a solve exceeds cache_max_its iterations)",
            default="none",
        ),
        option(
            "cache_max_its",
            int,
            "Iteration bar that triggers a preconditioner rebuild under the max_its cache strategy",
            default=10,
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> GMRES:
        del factory
        return cls(
            restart=node.param_optional_int("restart", 40),
            max_its=node.param_optional_int("max_its", 1000),
            abs_tol=node.param_optional_float("abs_tol", 0.0),
            rel_tol=node.param_optional_float("rel_tol", 1.0e-4),
            preconditioner=node.param_optional_str("preconditioner", "none"),
            cache_strategy=node.param_optional_str("cache_strategy", "none"),
            cache_max_its=node.param_optional_int("cache_max_its", 10),
        )


__all__ = ["GMRES"]
