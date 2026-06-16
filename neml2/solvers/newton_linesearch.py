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

"""Newton-Raphson solver with backtracking / strong-Wolfe line search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from neml2.es import AssembledVector, NonlinearSystem, norm_sq
from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, option

from ._helpers import _dot, _scale_assembled
from .dense_lu import DenseLU
from .newton import Newton
from .schur_complement import SchurComplement

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


@register_neml2_object("NewtonWithLineSearch")
class NewtonWithLineSearch(Newton):
    """The Newton-Raphson solver with line search."""

    hit = HitSchema(
        *Newton.hit.fields,
        option(
            "linesearch_type",
            str,
            "The type of linesearch used. Options are BACKTRACKING and STRONG_WOLFE.",
            default="BACKTRACKING",
        ),
        option(
            "max_linesearch_iterations",
            int,
            "Maximum allowable linesearch iterations. No error is produced upon reaching the "
            "maximum number of iterations, and the scale factor in the last iteration is used "
            "to scale the step.",
            default=10,
        ),
        option(
            "linesearch_cutback",
            float,
            "Linesearch cut-back factor when the current scale factor cannot sufficiently "
            "reduce the residual.",
            default=2.0,
        ),
        option(
            "linesearch_stopping_criteria",
            float,
            "The linesearch tolerance slightly relaxing the definition of residual decrease",
            default=1.0e-3,
        ),
        option(
            "check_negative_criterion",
            bool,
            "Whether to check if the threshold used in the convergence criterion for line "
            "search becomes negative. If true, and a negative value is detected, a warning "
            "message is printed.",
            default=False,
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> NewtonWithLineSearch:  # type: ignore[override]
        ls_name = node.param_optional_str("linear_solver", "")
        linear_solver = factory.get_solver(ls_name) if ls_name else None
        atol = node.param_optional_float("abs_tol", 1.0e-10)
        rtol = node.param_optional_float("rel_tol", 1.0e-8)
        miters = int(node.param_optional_int("max_its", 25))
        ls_type = node.param_optional_str("linesearch_type", "BACKTRACKING")
        ls_miter = int(node.param_optional_int("max_linesearch_iterations", 10))
        ls_sigma = node.param_optional_float("linesearch_cutback", 2.0)
        ls_c = node.param_optional_float("linesearch_stopping_criteria", 1.0e-3)
        check_crit = bool(node.param_optional_int("check_negative_criterion", 0))
        return cls(
            linear_solver=linear_solver,
            atol=atol,
            rtol=rtol,
            miters=miters,
            linesearch_type=ls_type,
            max_linesearch_iterations=ls_miter,
            linesearch_cutback=ls_sigma,
            linesearch_stopping_criteria=ls_c,
            check_negative_criterion=check_crit,
        )

    def __init__(
        self,
        *,
        linear_solver: DenseLU | SchurComplement | None = None,
        atol: float = 1.0e-10,
        rtol: float = 1.0e-8,
        miters: int = 25,
        verbose: bool = False,
        linesearch_type: str = "BACKTRACKING",
        max_linesearch_iterations: int = 10,
        linesearch_cutback: float = 2.0,
        linesearch_stopping_criteria: float = 1.0e-3,
        check_negative_criterion: bool = False,
    ) -> None:
        super().__init__(
            linear_solver=linear_solver,
            atol=atol,
            rtol=rtol,
            miters=miters,
            verbose=verbose,
        )
        if linesearch_type not in ("BACKTRACKING", "STRONG_WOLFE"):
            raise ValueError(
                f"NewtonWithLineSearch: linesearch_type={linesearch_type!r} must be "
                "'BACKTRACKING' or 'STRONG_WOLFE'."
            )
        self._ls_type = linesearch_type
        self._ls_miter = int(max_linesearch_iterations)
        self._ls_sigma = float(linesearch_cutback)
        self._ls_c = float(linesearch_stopping_criteria)
        self._check_crit = bool(check_negative_criterion)

    def update(self, system: NonlinearSystem) -> None:
        A, b = system.A_and_b()
        du = self.linear_solver.solve(A, b)
        assert isinstance(du, AssembledVector)

        u = system.u()
        b0 = system.b()
        nb0_sq = norm_sq(b0)  # (*B,)
        b0_dot_du = _dot(b0, du)

        # Per-element alpha -- broadcast over all assembled vector groups.
        alpha = torch.ones_like(nb0_sq)
        for _ in range(1, self._ls_miter):
            up = u + _scale_assembled(du, alpha)
            system.set_u(up)
            b_curr = system.b()
            nb_sq = norm_sq(b_curr)

            if self._ls_type == "BACKTRACKING":
                crit = nb0_sq - 2.0 * self._ls_c * alpha * b0_dot_du
            else:  # STRONG_WOLFE
                crit = (1.0 - self._ls_c * alpha) * nb0_sq

            if self._check_crit and bool((crit < 0).any().item()):
                import warnings  # noqa: PLC0415

                warnings.warn(
                    "NewtonWithLineSearch: negative stopping criterion encountered; "
                    "consider increasing linesearch_cutback or lowering "
                    "linesearch_stopping_criteria.",
                    stacklevel=2,
                )

            stop = (nb_sq <= crit) | (nb_sq <= self.atol * self.atol)
            if bool(stop.all().item()):
                break
            # Halve alpha (by cutback factor) for elements that still fail.
            alpha = torch.where(stop, alpha, alpha / self._ls_sigma)


__all__ = ["NewtonWithLineSearch"]
