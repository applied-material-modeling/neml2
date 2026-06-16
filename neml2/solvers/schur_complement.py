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

"""Schur-complement linear solver for block-partitioned systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from neml2.es import AssembledMatrix, AssembledVector
from neml2.factory import register_neml2_object
from neml2.schema import HitSchema, dependency, option

from .dense_lu import DenseLU

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


@register_neml2_object("SchurComplement")
class SchurComplement:
    """Schur complement linear solver. Solves a block-partitioned system
    ``A x = b`` by forming and solving the Schur complement of the primary
    block.

    The six-step factorisation reads naturally on top of the
    :class:`~neml2.types.Tensor`-backed :class:`AssembledMatrix` /
    :class:`AssembledVector` arithmetic -- ``A_pp.solve(...)``,
    ``A_sp @ Y``, ``A_ss - A_sp @ Y`` all forward to the typed primitive.
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

        Y = self._primary_solver.solve(A_pp, A_ps)  # 1: Y = A_pp^-1 A_ps
        z = self._primary_solver.solve(A_pp, b_p)  # 2: z = A_pp^-1 b_p
        S = A_ss - A_sp @ Y  # 3: S = A_ss - A_sp Y
        d = b_s - A_sp @ z  # 4: d = b_s - A_sp z
        x_s = self._schur_solver.solve(S, d)  # 5: x_s = S^-1 d
        x_p = z - Y @ x_s  # 6: x_p = z - Y x_s

        # Reassemble in the original column-group order: primary block at
        # index `_up`, Schur block at `_us`. Since `_up + _us == 1`, only
        # two possible orderings exist.
        tensors = (
            [x_p.tensors[0], x_s.tensors[0]] if self._up == 0 else [x_s.tensors[0], x_p.tensors[0]]
        )
        return AssembledVector(A.col_layout, tensors)

    def _solve_matrix(self, A: AssembledMatrix, B: AssembledMatrix) -> AssembledMatrix:
        # B must have 2 row groups (matches A's primary/Schur split).
        # B's col groups index independent right-hand-sides; solve each
        # col group separately and concat to assemble the full result.
        # The single-col-group restriction is only an algorithmic
        # convenience -- the Schur identities are linear in B and apply
        # per-col-group unchanged. Multi-col-group B is produced when
        # the equation system's glayout splits givens into per-structure
        # groups (e.g. mxpc with a BLOCK per-grain history group + a
        # DENSE global force group); a flat single-group B would force
        # the per-grain history cross-block at O(N^2).
        if B.row_layout.ngroup != 2:
            raise ValueError(
                f"SchurComplement requires B to have exactly 2 row groups "
                f"(got {B.row_layout.ngroup})"
            )

        A_pp = A.group(self._rp, self._up)
        A_ps = A.group(self._rp, self._us)
        A_sp = A.group(self._rs, self._up)
        A_ss = A.group(self._rs, self._us)

        # Y and S do not depend on B; compute once and reuse across col groups.
        Y = self._primary_solver.solve(A_pp, A_ps)  # 1: Y = A_pp^-1 A_ps
        S = A_ss - A_sp @ Y  # 3: S = A_ss - A_sp Y

        # Per-col-group: solve (A^-1 @ B[:, col_group]) via Schur.
        per_col_X_p: list = []
        per_col_X_s: list = []
        for j in range(B.col_layout.ngroup):
            B_p = B.group(self._rp, j)
            B_s = B.group(self._rs, j)
            Z = self._primary_solver.solve(A_pp, B_p)  # 2: Z = A_pp^-1 B_p
            D = B_s - A_sp @ Z  # 4: D = B_s - A_sp Z
            X_s = self._schur_solver.solve(S, D)  # 5: X_s = S^-1 D
            X_p = Z - Y @ X_s  # 6: X_p = Z - Y X_s
            per_col_X_p.append(X_p.tensors[0][0])
            per_col_X_s.append(X_s.tensors[0][0])

        tensors = [per_col_X_p, per_col_X_s] if self._up == 0 else [per_col_X_s, per_col_X_p]
        return AssembledMatrix(A.col_layout, B.col_layout, tensors)

    def _validate_A(self, A: AssembledMatrix) -> None:
        if A.row_layout.ngroup != 2 or A.col_layout.ngroup != 2:
            raise ValueError(
                "SchurComplement requires A to have exactly 2 row groups and 2 column "
                f"groups (got {A.row_layout.ngroup}, {A.col_layout.ngroup})"
            )
        # A_pp must be square for `primary_solver` to invert it. Surface
        # this at the SchurComplement boundary so the error names the
        # algorithmic constraint instead of bubbling up from inside
        # DenseLU as `torch.linalg.solve: A must be batches of square
        # matrices`.
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


__all__ = ["SchurComplement"]
