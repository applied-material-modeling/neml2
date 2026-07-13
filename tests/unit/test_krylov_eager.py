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

"""Eager parity for the matrix-free iterative (Krylov) linear solvers.

The iterative ``GMRES`` / ``BiCGStab`` linear solvers drive the *same* shared C++
Newton loop as the direct ``DenseLU``, but each step's linear solve is a Krylov
iteration over the ``Matvec`` (``J.v``) callback (``Newton._solve_iterative`` ->
``krylov_solve_eager``). Since the converged Newton iterate is solver-independent,
every iterative configuration must land on the same root as ``DenseLU``. This
exercises the full eager Krylov data path (flatten -> C++ krylov_solve ->
unflatten, preconditioner setup/apply, cache policy) end to end.
"""

from __future__ import annotations

import pytest
import torch

from neml2.es import ModelNonlinearSystem, SparseVector
from neml2.models.model import Model
from neml2.solvers import (
    GMRES,
    BiCGStab,
    BlockJacobiPreconditioner,
    DenseLU,
    FullPreconditioner,
    JacobiPreconditioner,
    Newton,
    RetCode,
)
from neml2.types import Scalar
from neml2.types import allclose as _allclose


class ScalarResidual(Model):
    """Single-unknown ``x^2 - c = 0`` -- a 1x1 system (plumbing smoke)."""

    input_spec = {"x": Scalar, "c": Scalar}
    output_spec = {"x_residual": Scalar}

    def forward(self, x: Scalar, c: Scalar, v=None):
        r = x * x - c
        if v is None:
            return r
        return r, self.apply_chain_rule(
            v, "x_residual", {"x": lambda V: 2.0 * x * V, "c": lambda V: -V}, output=r
        )


class CoupledResidual(Model):
    """Two coupled unknowns with a genuinely dense 2x2 Jacobian:

        r_x = x^2 + 0.5 y - c
        r_y = y^2 + 0.5 x - d

    J = [[2x, 0.5], [0.5, 2y]] -- off-diagonal coupling makes the Krylov + the
    Block-Jacobi / Full preconditioners do real work (a diagonal system would be
    trivial for any of them).
    """

    input_spec = {"x": Scalar, "y": Scalar, "c": Scalar, "d": Scalar}
    output_spec = {"x_residual": Scalar, "y_residual": Scalar}

    def forward(self, x: Scalar, y: Scalar, c: Scalar, d: Scalar, v=None):
        rx = x * x + 0.5 * y - c
        ry = y * y + 0.5 * x - d
        if v is None:
            return rx, ry
        vx = self.apply_chain_rule(
            v, "x_residual", {"x": lambda V: 2.0 * x * V, "y": lambda V: 0.5 * V}, output=rx
        )
        vy = self.apply_chain_rule(
            v, "y_residual", {"x": lambda V: 0.5 * V, "y": lambda V: 2.0 * y * V}, output=ry
        )
        return rx, ry, {**vx, **vy}


def _scalar_system() -> ModelNonlinearSystem:
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )
    return sys


def _coupled_system() -> ModelNonlinearSystem:
    sys = ModelNonlinearSystem(CoupledResidual(), unknowns=[["x", "y"]])
    sys.initialize(
        u=SparseVector(
            sys.ulayout,
            {
                "x": Scalar(torch.tensor([1.0, 1.5], dtype=torch.float64)),
                "y": Scalar(torch.tensor([1.0, 2.5], dtype=torch.float64)),
            },
        ),
        g=SparseVector(
            sys.glayout,
            {
                "c": Scalar(torch.tensor([4.5, 6.0], dtype=torch.float64)),
                "d": Scalar(torch.tensor([9.25, 12.0], dtype=torch.float64)),
            },
        ),
    )
    return sys


def _solve(system_factory, linear_solver) -> dict[str, Scalar]:
    # Library-default Newton tolerances (rtol 1e-8): the inner Krylov solve is
    # accurate enough that the iterative and direct routes take identical Newton
    # steps and land on the same root. (A far tighter outer rtol than the inner
    # ``rel_tol`` would expose the usual inexact-Newton floor for a
    # genuinely inexact inner solver like BiCGStab -- not a parity concern.)
    sys = system_factory()
    result = Newton(atol=1e-10, rtol=1e-8, miters=50, linear_solver=linear_solver).solve(sys)
    assert result.ret is RetCode.SUCCESS
    return sys.u().disassemble()


def test_gmres_bicgstab_match_dense_lu_scalar():
    gold = _solve(_scalar_system, DenseLU())
    for solver in (GMRES(), BiCGStab()):
        got = _solve(_scalar_system, solver)
        assert _allclose(got["x"], gold["x"], atol=1e-8), f"{type(solver).__name__} scalar mismatch"


# GMRES/BiCGStab x preconditioner x cache-strategy, all must hit the DenseLU root.
_CONFIGS = [
    GMRES(),
    GMRES(preconditioner=JacobiPreconditioner()),
    GMRES(preconditioner=BlockJacobiPreconditioner(), cache_strategy="chord"),
    GMRES(preconditioner=FullPreconditioner(), cache_strategy="chord"),
    GMRES(preconditioner=FullPreconditioner(), cache_strategy="max_its", cache_max_its=2),
    GMRES(restart=1),  # forces multiple restarts on the 2-D system
    BiCGStab(),
    BiCGStab(preconditioner=BlockJacobiPreconditioner(), cache_strategy="chord"),
]


def _config_id(s) -> str:
    return f"{type(s).__name__}-{s.preconditioner.kind}-{s.cache_strategy}"


@pytest.mark.parametrize("solver", _CONFIGS, ids=_config_id)
def test_iterative_matches_dense_lu_coupled(solver):
    gold = _solve(_coupled_system, DenseLU())
    got = _solve(_coupled_system, solver)
    assert _allclose(got["x"], gold["x"], atol=1e-8)
    assert _allclose(got["y"], gold["y"], atol=1e-8)


def _coupled_system_unbatched_u0() -> ModelNonlinearSystem:
    """Batched givens but an UNBATCHED initial guess -- a broadcast u0 (shape
    ``(base,)`` not ``(*B, base)``), the common case for a predictor/scalar IC.
    The direct path handles this by broadcasting on ``u + du``; the Krylov path
    must take its vector batch from the (batched) residual, not from u0.
    """
    sys = ModelNonlinearSystem(CoupledResidual(), unknowns=[["x", "y"]])
    sys.initialize(
        u=SparseVector(
            sys.ulayout,
            {  # unbatched scalars -- no leading batch axis
                "x": Scalar(torch.tensor(1.0, dtype=torch.float64)),
                "y": Scalar(torch.tensor(1.0, dtype=torch.float64)),
            },
        ),
        g=SparseVector(
            sys.glayout,
            {  # batched givens (batch 2)
                "c": Scalar(torch.tensor([4.5, 6.0], dtype=torch.float64)),
                "d": Scalar(torch.tensor([9.25, 12.0], dtype=torch.float64)),
            },
        ),
    )
    return sys


@pytest.mark.parametrize("solver", _CONFIGS, ids=_config_id)
def test_iterative_handles_unbatched_initial_guess(solver):
    """Regression: an unbatched u0 with batched givens must solve (and match the
    direct route) rather than crashing in the Krylov flatten/unflatten -- the
    Krylov batch comes from the residual, not u0. (chaboche12's driver hits this;
    the earlier batched-u0 tests did not.)"""
    gold = _solve(_coupled_system_unbatched_u0, DenseLU())
    got = _solve(_coupled_system_unbatched_u0, solver)
    assert _allclose(got["x"], gold["x"], atol=1e-8)
    assert _allclose(got["y"], gold["y"], atol=1e-8)
