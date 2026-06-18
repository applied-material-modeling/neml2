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
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Newton:
        ls_name = node.param_optional_str("linear_solver", "")
        linear_solver = factory.get_solver(ls_name) if ls_name else None
        atol = node.param_optional_float("abs_tol", 1.0e-10)
        rtol = node.param_optional_float("rel_tol", 1.0e-8)
        miters = int(node.param_optional_int("max_its", 25))
        return cls(linear_solver=linear_solver, atol=atol, rtol=rtol, miters=miters)

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
        Newton-step callbacks (the ``RHS`` / ``NewtonStep`` export modules, run
        eagerly) plus the per-group layouts, then commits the converged iterate
        back into the system.
        """
        # Local imports: the compiled extension + the per-group raw boundary
        # helper, kept off the package import path so a partial build can still
        # import solvers.
        from neml2.aoti._aoti import newton_solve_eager  # noqa: PLC0415
        from neml2.es.implicit import _vector_to_per_group_raws  # noqa: PLC0415

        # The Newton iterates are a fixed-point solve: gradients flow through
        # the converged state via the IFT (see ImplicitUpdate), never through
        # the iterations, so the solve runs detached.
        with torch.no_grad():
            u0_raws = list(_vector_to_per_group_raws(system.u()))

            # residual / step delegate to the system's native assemble (the same
            # path the old eager Newton.update used) -- only the per-group raw
            # marshalling is new. The C++ loop owns convergence + line search.
            def residual_fn(u_raws: list[torch.Tensor]) -> list[torch.Tensor]:
                system.set_u_from_group_raws(u_raws)
                return list(_vector_to_per_group_raws(system.b()))

            def step_fn(
                u_raws: list[torch.Tensor],
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
                system.set_u_from_group_raws(u_raws)
                A, b = system.A_and_b()
                du = self.linear_solver.solve(A, b)
                return (
                    list(_vector_to_per_group_raws(du)),
                    list(_vector_to_per_group_raws(b)),
                )

            u_star, converged, iterations = newton_solve_eager(
                residual_fn=residual_fn,
                step_fn=step_fn,
                unknown_layout=_group_layout_descriptor(system.ulayout),
                residual_layout=_group_layout_descriptor(system.blayout),
                u0=u0_raws,
                **self._solver_config(),
            )
            system.set_u_from_group_raws(u_star)

        ret = RetCode.SUCCESS if converged else RetCode.MAXITER
        return NonlinearResult(ret, iterations)


__all__ = ["Newton"]
