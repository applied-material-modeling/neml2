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

"""Python-native linear and nonlinear solvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, overload

import torch

from .equation_systems import (
    AssembledMatrix,
    AssembledVector,
    IStructure,
    NonlinearSystem,
    norm,
    norm_sq,
)
from .factory import register_neml2_object
from .schema import HitSchema, dependency, option

if TYPE_CHECKING:
    import nmhit

    from .factory import _NativeInputFile


class RetCode(Enum):
    """Nonlinear solver return code."""

    SUCCESS = 0
    MAXITER = 1
    FAILURE = 2


@dataclass(frozen=True)
class NonlinearResult:
    """Result metadata returned by a nonlinear solve."""

    ret: RetCode
    iterations: int


@register_neml2_object("DenseLU")
class DenseLU:
    """Dense LU linear solver. This solver assembles the (possibly) sparse matrix into a
    dense one and uses a standard LU decomposition to solve the system of equations.
    """

    SECTION = "Solvers"

    # No tunable options; the empty schema documents that in the syntax catalog.
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> DenseLU:
        del node, factory
        return cls()

    @overload
    def solve(self, A: AssembledMatrix, b: AssembledVector) -> AssembledVector: ...
    @overload
    def solve(self, A: AssembledMatrix, b: AssembledMatrix) -> AssembledMatrix: ...
    def solve(self, A, b):
        if A.row_layout.ngroup != 1 or A.col_layout.ngroup != 1:
            raise ValueError("DenseLU only supports single-group layouts")

        A00 = A.tensors[0][0]
        if isinstance(b, AssembledVector):
            if b.layout.ngroup != 1:
                raise ValueError("DenseLU only supports single-group vector RHS")
            x = torch.linalg.solve(A00, b.tensors[0].unsqueeze(-1)).squeeze(-1)
            return AssembledVector(A.col_layout, [x])

        if b.row_layout.ngroup != 1 or b.col_layout.ngroup != 1:
            raise ValueError("DenseLU only supports single-group matrix RHS")
        x = torch.linalg.solve(A00, b.tensors[0][0])
        return AssembledMatrix(A.col_layout, b.col_layout, [[x]])


@register_neml2_object("SchurComplement")
class SchurComplement:
    """Schur complement linear solver. Solves a block-partitioned system A x = b
    by forming and solving the Schur complement of the primary block.
    """

    SECTION = "Solvers"

    hit = HitSchema(
        option(
            "residual_primary_group",
            int,
            "Row (residual) group index of the primary block. The system must have exactly 2 "
            "residual groups; the other group is automatically the Schur complement residual "
            "group.",
            default=0,
        ),
        option(
            "unknown_primary_group",
            int,
            "Column (unknown) group index of the primary block. The system must have exactly 2 "
            "unknown groups; the other group is automatically the Schur complement unknown group.",
            default=0,
        ),
        dependency("primary_solver", "get_solver", "Linear solver used for the primary block A_pp"),
        dependency(
            "schur_solver", "get_solver", "Linear solver used for the Schur complement block S"
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> SchurComplement:
        return cls(
            residual_primary_group=node.param_optional_int("residual_primary_group", 0),
            unknown_primary_group=node.param_optional_int("unknown_primary_group", 0),
            primary_solver=factory.get_solver(node.param_str("primary_solver")),
            schur_solver=factory.get_solver(node.param_str("schur_solver")),
        )

    def __init__(
        self,
        *,
        residual_primary_group: int = 0,
        unknown_primary_group: int = 0,
        primary_solver=None,
        schur_solver=None,
    ) -> None:
        if residual_primary_group not in (0, 1):
            raise ValueError(f"residual_primary_group must be 0 or 1, got {residual_primary_group}")
        if unknown_primary_group not in (0, 1):
            raise ValueError(f"unknown_primary_group must be 0 or 1, got {unknown_primary_group}")
        self._rp = residual_primary_group
        self._rs = 1 - residual_primary_group
        self._up = unknown_primary_group
        self._us = 1 - unknown_primary_group
        self._primary_solver = primary_solver if primary_solver is not None else DenseLU()
        self._schur_solver = schur_solver if schur_solver is not None else DenseLU()

    @overload
    def solve(self, A: AssembledMatrix, b: AssembledVector) -> AssembledVector: ...
    @overload
    def solve(self, A: AssembledMatrix, b: AssembledMatrix) -> AssembledMatrix: ...
    def solve(self, A, b):
        self._validate_A(A)
        if isinstance(b, AssembledVector):
            return self._solve_vector(A, b)
        return self._solve_matrix(A, b)

    def _solve_vector(self, A: AssembledMatrix, b: AssembledVector) -> AssembledVector:
        if b.layout.ngroup != 2:
            raise ValueError(
                f"SchurComplement requires b to have exactly 2 groups (got {b.layout.ngroup})"
            )

        A_pp = A.group(self._rp, self._up)
        A_ps = A.group(self._rp, self._us)
        A_sp = A.group(self._rs, self._up)
        A_ss = A.group(self._rs, self._us)
        b_p = b.group(self._rp)
        b_s = b.group(self._rs)

        Y = self._primary_solver.solve(A_pp, A_ps)  # 1: Y = A_pp⁻¹ A_ps
        z = self._primary_solver.solve(A_pp, b_p)  # 2: z = A_pp⁻¹ b_p
        S = A_ss - A_sp @ Y  # 3: S = A_ss - A_sp Y
        d = b_s - A_sp @ z  # 4: d = b_s - A_sp z
        x_s = self._schur_solver.solve(S, d)  # 5: x_s = S⁻¹ d
        x_p = z - Y @ x_s  # 6: x_p = z - Y x_s

        # Reassemble in the original column-group order: primary block at
        # index `_up`, Schur block at `_us`. Since `_up + _us == 1`, only
        # two possible orderings exist.
        tensors = (
            [x_p.tensors[0], x_s.tensors[0]] if self._up == 0 else [x_s.tensors[0], x_p.tensors[0]]
        )
        return AssembledVector(A.col_layout, tensors)

    def _solve_matrix(self, A: AssembledMatrix, B: AssembledMatrix) -> AssembledMatrix:
        # Mirror the C++ assertion: matrix-RHS Schur is only well-defined when
        # B has a single column group (the sensitivity matrix in the IFT path
        # always satisfies this). Allowing more would force the reassembly to
        # iterate over column groups for no real caller's benefit.
        if B.row_layout.ngroup != 2 or B.col_layout.ngroup != 1:
            raise ValueError(
                f"SchurComplement requires B to have exactly 2 row groups and 1 column "
                f"group (got {B.row_layout.ngroup}, {B.col_layout.ngroup})"
            )

        A_pp = A.group(self._rp, self._up)
        A_ps = A.group(self._rp, self._us)
        A_sp = A.group(self._rs, self._up)
        A_ss = A.group(self._rs, self._us)
        B_p = B.group(self._rp, 0)
        B_s = B.group(self._rs, 0)

        Y = self._primary_solver.solve(A_pp, A_ps)  # 1: Y = A_pp⁻¹ A_ps
        Z = self._primary_solver.solve(A_pp, B_p)  # 2: Z = A_pp⁻¹ B_p
        S = A_ss - A_sp @ Y  # 3: S = A_ss - A_sp Y
        D = B_s - A_sp @ Z  # 4: D = B_s - A_sp Z
        X_s = self._schur_solver.solve(S, D)  # 5: X_s = S⁻¹ D
        X_p = Z - Y @ X_s  # 6: X_p = Z - Y X_s

        tensors = (
            [[X_p.tensors[0][0]], [X_s.tensors[0][0]]]
            if self._up == 0
            else [[X_s.tensors[0][0]], [X_p.tensors[0][0]]]
        )
        return AssembledMatrix(A.col_layout, B.col_layout, tensors)

    def _validate_A(self, A: AssembledMatrix) -> None:
        if A.row_layout.ngroup != 2 or A.col_layout.ngroup != 2:
            raise ValueError(
                "SchurComplement requires A to have exactly 2 row groups and 2 column "
                f"groups (got {A.row_layout.ngroup}, {A.col_layout.ngroup})"
            )
        # A_pp must be square for `primary_solver` to invert it. Surface this
        # at the SchurComplement boundary so the error names the algorithmic
        # constraint instead of bubbling up from inside DenseLU as
        # `torch.linalg.solve: A must be batches of square matrices`.
        rp_size = A.row_layout.group_size(self._rp)
        up_size = A.col_layout.group_size(self._up)
        if rp_size != up_size:
            raise ValueError(
                f"SchurComplement primary block A_pp must be square: "
                f"residual group {self._rp} has size {rp_size}, "
                f"unknown group {self._up} has size {up_size}. "
                f"Re-check (residual_primary_group, unknown_primary_group) "
                f"against the layout's group sizes."
            )


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

        Extracted so :class:`NewtonWithLineSearch` can override the per-iter
        update without re-implementing the convergence loop. Mirrors the C++
        ``Newton::update`` / ``NewtonWithLineSearch::update`` split.
        """
        A, b = system.A_and_b()
        du = self.linear_solver.solve(A, b)
        assert isinstance(du, AssembledVector)
        system.set_u(system.u() + du)


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
        # b0 · du as a single scalar per batch element (sum over all groups).
        b0_dot_du = _dot(b0, du)

        # Per-element alpha — broadcast over all assembled vector groups.
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


def _dot(a: AssembledVector, b: AssembledVector) -> torch.Tensor:
    """Batched dot product across all assembled-vector groups."""
    total: torch.Tensor | None = None
    for ta, tb in zip(a.tensors, b.tensors, strict=True):
        contribution = (ta * tb).sum(dim=-1)
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("Cannot dot empty AssembledVectors.")
    return total


def _scale_assembled(v: AssembledVector, alpha: torch.Tensor) -> AssembledVector:
    """Scale each group of an ``AssembledVector`` by per-element ``alpha``.

    ``alpha`` has shape ``(*B,)`` — broadcast over each group's trailing
    base dim.
    """
    scaled = [t * alpha.unsqueeze(-1) for t in v.tensors]
    return AssembledVector(v.layout, scaled)


__all__ = [
    "RetCode",
    "NonlinearResult",
    "DenseLU",
    "SchurComplement",
    "Newton",
    "NewtonWithLineSearch",
    "IStructure",
]
