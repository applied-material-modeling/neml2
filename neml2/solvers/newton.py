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

from neml2.es import AssembledVector, NonlinearSystem, norm
from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, dependency, option

from ._result import NonlinearResult, RetCode
from .dense_lu import DenseLU
from .schur_complement import SchurComplement

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


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

    def converged(self, itr: int, nb: torch.Tensor, nb0: torch.Tensor) -> bool:
        if self.verbose:
            print(
                f"ITERATION {itr:3d}, |R| = {torch.max(nb).item():.6e}, "
                f"|R0| = {torch.max(nb0).item():.6e}"
            )
        rel = nb / torch.clamp(nb0, min=torch.finfo(nb.dtype).tiny)
        return bool(torch.all(torch.logical_or(nb < self.atol, rel < self.rtol)).item())

    def solve(self, system: NonlinearSystem) -> NonlinearResult:
        b = system.b()
        nb = norm(b)
        nb0 = nb.clone()

        if self.converged(0, nb, nb0):
            return NonlinearResult(RetCode.SUCCESS, 0)

        for itr in range(1, self.miters):
            self.update(system)
            b = system.b()
            nb = norm(b)
            if self.converged(itr, nb, nb0):
                return NonlinearResult(RetCode.SUCCESS, itr)

        return NonlinearResult(RetCode.MAXITER, self.miters)

    def update(self, system: NonlinearSystem) -> None:
        """Apply one Newton step: solve ``A du = b`` and update ``u``.

        Extracted so :class:`NewtonWithLineSearch` can override the
        per-iter update without re-implementing the convergence loop.
        Mirrors the C++ ``Newton::update`` / ``NewtonWithLineSearch::update``
        split.
        """
        A, b = system.A_and_b()
        du = self.linear_solver.solve(A, b)
        assert isinstance(du, AssembledVector)
        system.set_u(system.u() + du)


__all__ = ["Newton"]
