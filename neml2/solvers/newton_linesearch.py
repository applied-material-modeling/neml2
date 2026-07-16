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

from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, option

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
        substep_del_tol = node.param_optional_float("substep_del_tol", 1.0e-6)
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
            substep_del_tol=substep_del_tol,
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
        substep_del_tol: float = 1.0e-6,
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
            substep_del_tol=substep_del_tol,
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

    def _solver_config(self) -> dict:
        """Forward the line-search options to the shared C++ Newton solver.

        The iteration + backtracking loop itself lives in C++ (shared with the
        AOTI runtime); this only differs from the base ``Newton`` by the
        line-search config it sends. Note: ``check_negative_criterion`` is a
        diagnostic-only option and is currently not honored by the C++ loop.
        """
        cfg = super()._solver_config()
        cfg.update(
            {
                "ls_type": self._ls_type,
                "ls_max_iters": self._ls_miter,
                "ls_cutback": self._ls_sigma,
                "ls_c": self._ls_c,
            }
        )
        return cfg


__all__ = ["NewtonWithLineSearch"]
