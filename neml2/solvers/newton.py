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

"""Standard Newton-Raphson solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, dependency, option

from ._result import NonlinearResult, RetCode
from .dense_lu import DenseLU
from .schur_complement import SchurComplement

if TYPE_CHECKING:
    import nmhit

    from neml2.es import AxisLayout, ModelNonlinearSystem
    from neml2.factory import _NativeInputFile


def _group_layout_descriptor(layout: AxisLayout) -> list[tuple[str, list[int]]]:
    """Per-group ``(structure, sub_batch_shape)`` descriptors for the C++ solver.

    The shared C++ Newton loop needs only each group's ``SubBatchStructure``
    and ``sub_batch_shape`` to compute its per-group reductions -- it is
    deliberately independent of the Python ``AxisLayout``.
    """
    return [
        (layout.structure[gi], [int(s) for s in layout.group_sub_batch_shape(gi)])
        for gi in range(layout.ngroup)
    ]


@register_neml2_object("Newton")
class Newton:
    """The standard Newton-Raphson solver which always takes the 'full' Newton step."""

    #: Inherited by ``NewtonWithLineSearch``.
    SECTION = "Solvers"

    hit = HitSchema(
        dependency(
            "linear_solver",
            "get_solver",
            "The linear solver to use within the nonlinear solver",
            default=None,
        ),
        option("abs_tol", float, "Absolute tolerance in the convergence criteria", default=1.0e-10),
        option("rel_tol", float, "Relative tolerance in the convergence criteria", default=1.0e-8),
        option(
            "max_its",
            int,
            "Maximum number of iterations allowed before issuing an error/exception",
            default=25,
        ),
        option(
            "verbose",
            bool,
            "Print the Newton convergence history (per-iteration residual norms, "
            "including any line-search sub-iterations) after each solve",
            default=False,
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Newton:
        ls_name = node.param_optional_str("linear_solver", "")
        linear_solver = factory.get_solver(ls_name) if ls_name else None
        atol = node.param_optional_float("abs_tol", 1.0e-10)
        rtol = node.param_optional_float("rel_tol", 1.0e-8)
        miters = int(node.param_optional_int("max_its", 25))
        verbose = node.param_optional_bool("verbose", False)
        return cls(
            linear_solver=linear_solver, atol=atol, rtol=rtol, miters=miters, verbose=verbose
        )

    def __init__(
        self,
        *,
        linear_solver: DenseLU | SchurComplement | None = None,
        atol: float = 1.0e-10,
        rtol: float = 1.0e-8,
        miters: int = 25,
        verbose: bool = False,
    ) -> None:
        self.linear_solver = linear_solver if linear_solver is not None else DenseLU()
        self.atol = atol
        self.rtol = rtol
        self.miters = miters
        self.verbose = verbose

    def _solver_config(self) -> dict:
        """Per-group Newton config forwarded to the C++ solver.

        Base ``Newton`` takes the full step (``ls_max_iters=1`` disables line
        search). :class:`NewtonWithLineSearch` overrides this with its
        line-search options.
        """
        return {
            "atol": self.atol,
            "rtol": self.rtol,
            "miters": self.miters,
            "ls_type": "BACKTRACKING",
            "ls_max_iters": 1,
            "ls_cutback": 2.0,
            "ls_c": 1.0e-3,
        }

    def solve(self, system: ModelNonlinearSystem) -> NonlinearResult:
        """Solve the nonlinear system via the shared C++ Newton solver.

        The iteration control (convergence test, line search) lives in C++ and
        is shared with the AOTI runtime; this wrapper supplies the residual /
        Newton-step callbacks (the ``RHS`` residual + the ``Jacobian`` operator
        chained into ``LinearSolve``, the same modules the AOTI runtime compiles,
        run eagerly here) plus the per-group layouts, then commits the converged
        iterate back into the system.
        """
        # Local imports: the compiled extension + the residual/step export
        # modules, kept off the package import path so a partial build can still
        # import solvers.
        from neml2.aoti._aoti import newton_solve_eager  # noqa: PLC0415
        from neml2.es.implicit import (  # noqa: PLC0415
            RHS,
            Jacobian,
            LinearSolve,
            _vector_to_per_group_raws,
        )

        # The eager path runs the *same* operator + solve modules the AOTI runtime
        # compiles (RHS / Jacobian / LinearSolve), so the two cannot diverge --
        # only the iteration control differs (here C++, there C++ too). The linear
        # solve is un-baked from the Jacobian operator: ``Jacobian`` assembles
        # ``(A_blocks, b)``; ``LinearSolve`` reconstructs the typed operators and
        # applies the configured Python solver. The givens are bound once; the C++
        # loop drives the unknowns.
        rhs = RHS(system)
        jacobian = Jacobian(system)
        linear_solve = LinearSolve(system, self.linear_solver)
        n_a = system.blayout.ngroup * system.ulayout.ngroup  # A blocks (residual × unknown)

        # The Newton iterates are a fixed-point solve: gradients flow through
        # the converged state via the IFT (see ImplicitUpdate), never through
        # the iterations, so the solve runs detached.
        with torch.no_grad():
            g_raws = list(_vector_to_per_group_raws(system.g()))
            u0_raws = list(_vector_to_per_group_raws(system.u()))

            def residual_fn(u_raws: list[torch.Tensor]) -> list[torch.Tensor]:
                return list(rhs(*u_raws, *g_raws))

            def step_fn(
                u_raws: list[torch.Tensor],
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
                # Chain the operator + solve graphs: Jacobian -> (A_blocks, b),
                # LinearSolve(A_blocks, b) -> du. The b groups (the last
                # blayout.ngroup outputs of Jacobian) are reused for the
                # line-search residual, matching the compiled C++ step() path.
                jac_out = jacobian(*u_raws, *g_raws)
                du = linear_solve(*jac_out)
                b = jac_out[n_a:]
                return list(du), list(b)

            u_star, converged, iterations, log = newton_solve_eager(
                residual_fn=residual_fn,
                step_fn=step_fn,
                unknown_layout=_group_layout_descriptor(system.ulayout),
                residual_layout=_group_layout_descriptor(system.blayout),
                u0=u0_raws,
                # collect_log is eager-only (it controls returning the
                # convergence history to Python); the shared ``_solver_config``
                # also feeds the compiled path's ``set_solver_config``, which
                # has no such option.
                collect_log=self.verbose,
                **self._solver_config(),
            )
            system.set_u_from_group_raws(u_star)

        # ``verbose`` surfaces the convergence history the C++ loop collected
        # (``collect_log`` above). Printing from Python keeps it in the
        # notebook-captured stream, unlike the C++ ``NEML2_AOTI_TRACE_NEWTON``
        # stderr trace.
        if self.verbose and log:
            print("\n".join(log))

        ret = RetCode.SUCCESS if converged else RetCode.MAXITER
        return NonlinearResult(ret, iterations, log=tuple(log))


__all__ = ["Newton"]
