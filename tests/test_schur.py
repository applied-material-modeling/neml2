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

"""Python-native SchurComplement correctness coverage.

Mirrors the C++ ``TEST_CASE("NonlinearSolver/SchurComplement")`` pattern in
``tests/unit/solvers/test_NonlinearSolvers.cxx`` for the Python eager path.
Three layers:

1. Direct linear-solver tests against ``torch.linalg.solve`` on the flattened
   system — pins each step of the Schur factorisation independently of the
   Newton outer loop.
2. End-to-end Newton-on-Schur tests against Power / Rosenbrock 2-group
   residuals with closed-form roots, cross-checked against Newton-on-DenseLU
   on the flattened single-group reformulation.
3. HIT factory parsing — ``type = SchurComplement`` in ``tests/unit/solvers/
   solvers.i`` flows through ``_NativeInputFile`` and wires up the nested
   sub-solvers + group indices correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.chain_rule import ChainRuleDict
from neml2.equation_systems import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    IStructure,
    ModelNonlinearSystem,
)
from neml2.factory import load_input
from neml2.model import Model
from neml2.solvers import DenseLU, Newton, RetCode, SchurComplement
from neml2.types import Scalar

torch.set_default_dtype(torch.float64)

SOLVERS_I = Path(__file__).parent / "solvers" / "solvers.i"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assembled(
    A_full: torch.Tensor,
    b_full: torch.Tensor,
    sizes: tuple[int, int],
) -> tuple[AssembledMatrix, AssembledVector]:
    """Split a flat ``(*B, n, n)`` / ``(*B, n)`` pair into a 2-group assembled view.

    The variable names per group are ``g0v0..g0v{s0-1}`` and ``g1v0..g1v{s1-1}``
    so the layout's group sizes match ``sizes``. Each variable is a Scalar; the
    block tensors are pulled from ``A_full`` / ``b_full`` by index slicing.
    """
    s0, s1 = sizes
    group0 = [f"g0v{i}" for i in range(s0)]
    group1 = [f"g1v{i}" for i in range(s1)]
    specs: dict[str, type] = {name: Scalar for name in (*group0, *group1)}
    row_layout = AxisLayout([group0, group1], specs)
    col_layout = AxisLayout([group0, group1], specs)

    A_blocks: list[list[torch.Tensor]] = [
        [A_full[..., :s0, :s0], A_full[..., :s0, s0:]],
        [A_full[..., s0:, :s0], A_full[..., s0:, s0:]],
    ]
    b_blocks = [b_full[..., :s0], b_full[..., s0:]]
    return (
        AssembledMatrix(row_layout, col_layout, A_blocks),
        AssembledVector(row_layout, b_blocks),
    )


def _spd_matrix(n: int, batch: torch.Size, seed: int) -> torch.Tensor:
    """Generate a random symmetric positive-definite matrix for stable solves."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(*batch, n, n, generator=g)
    return M @ M.transpose(-1, -2) + n * torch.eye(n).expand(*batch, n, n)


# ---------------------------------------------------------------------------
# Direct linear-solver tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rp,up,sizes",
    [
        # Mirror C++ `tests/unit/solvers/test_NonlinearSolvers.cxx:97-99` —
        # asymmetric group sizes (5, 3) with the default (0, 0) indices.
        (0, 0, (5, 3)),
        # `_rp == _up == 1` swaps which block is the "primary" but still
        # picks square A_pp (the (1, 1) block, size 3×3). Both off-diagonal
        # cases (rp != up) are invalid for asymmetric group sizes because
        # A_pp would not be square — see `test_schur_rejects_*` below for the
        # symmetric-size cross-index coverage.
        (1, 1, (5, 3)),
        # Equal group sizes (4, 4) let us exercise the rp != up cross-index
        # paths where the algorithm picks an off-diagonal A_pp block.
        (0, 1, (4, 4)),
        (1, 0, (4, 4)),
    ],
)
def test_schur_vector_solve_matches_dense_lu(rp: int, up: int, sizes: tuple[int, int]):
    """SchurComplement(A, b) == torch.linalg.solve(A, b) for valid (rp, up) pairs."""
    batch = torch.Size([2])
    s0, s1 = sizes
    n = s0 + s1
    A_full = _spd_matrix(n, batch, seed=42)
    b_full = torch.randn(*batch, n, generator=torch.Generator().manual_seed(7))

    A, b = _make_assembled(A_full, b_full, sizes)
    schur = SchurComplement(
        residual_primary_group=rp,
        unknown_primary_group=up,
        primary_solver=DenseLU(),
        schur_solver=DenseLU(),
    )
    x = schur.solve(A, b)
    assert isinstance(x, AssembledVector)
    x_flat = torch.cat(x.tensors, dim=-1)
    x_ref = torch.linalg.solve(A_full, b_full.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x_flat, x_ref, rtol=1e-10, atol=1e-10)


def test_schur_matrix_solve_matches_dense_lu():
    """SchurComplement(A, B) == torch.linalg.solve(A, B) for matrix RHS."""
    batch = torch.Size([2])
    s0, s1 = 5, 3
    n = s0 + s1
    k = 4  # number of given-side columns
    A_full = _spd_matrix(n, batch, seed=11)
    B_full = torch.randn(*batch, n, k, generator=torch.Generator().manual_seed(13))

    # Build A with 2-group row/col layout and B with 2-group row + 1-group col.
    group0 = [f"g0v{i}" for i in range(s0)]
    group1 = [f"g1v{i}" for i in range(s1)]
    g_names = [f"gv{i}" for i in range(k)]
    A_specs: dict[str, type] = {name: Scalar for name in (*group0, *group1)}
    B_col_specs: dict[str, type] = {name: Scalar for name in g_names}
    row_layout = AxisLayout([group0, group1], A_specs)
    A = AssembledMatrix(
        row_layout,
        row_layout,
        [
            [A_full[..., :s0, :s0], A_full[..., :s0, s0:]],
            [A_full[..., s0:, :s0], A_full[..., s0:, s0:]],
        ],
    )
    B_col_layout = AxisLayout([g_names], B_col_specs)
    B = AssembledMatrix(
        row_layout,
        B_col_layout,
        [[B_full[..., :s0, :]], [B_full[..., s0:, :]]],
    )

    schur = SchurComplement(primary_solver=DenseLU(), schur_solver=DenseLU())
    X = schur.solve(A, B)
    assert isinstance(X, AssembledMatrix)
    # Stack the result rows back together for comparison.
    X_full = torch.cat([X.tensors[0][0], X.tensors[1][0]], dim=-2)
    X_ref = torch.linalg.solve(A_full, B_full)
    assert torch.allclose(X_full, X_ref, rtol=1e-10, atol=1e-10)


def test_schur_rejects_single_group_inputs():
    """A single-group A surfaces a clear error before any Schur arithmetic."""
    layout = AxisLayout([["x"]], {"x": Scalar})
    A = AssembledMatrix(layout, layout, [[torch.eye(1)]])
    b = AssembledVector.from_dict(layout, {"x": torch.tensor([1.0])})
    schur = SchurComplement(primary_solver=DenseLU(), schur_solver=DenseLU())
    with pytest.raises(ValueError, match="exactly 2 row groups"):
        schur.solve(A, b)


def test_schur_invalid_primary_group_index_rejected():
    """``residual_primary_group`` outside {0, 1} is caught at construction."""
    with pytest.raises(ValueError, match="residual_primary_group must be 0 or 1"):
        SchurComplement(residual_primary_group=2)


def test_schur_defaults_both_sub_solvers_to_dense_lu():
    """Programmatic construction without explicit sub-solvers picks DenseLU.

    HIT still requires both names explicitly — the C++ side has no defaults,
    so cross-stack inputs always spell them out — but the Python API matches
    the dominant pattern (Newton's `linear_solver` already defaults to
    DenseLU; SchurComplement now follows suit).
    """
    schur = SchurComplement()
    assert isinstance(schur._primary_solver, DenseLU)
    assert isinstance(schur._schur_solver, DenseLU)


def test_schur_non_square_primary_block_rejected_with_actionable_message():
    """A_pp must be square — surface the constraint at SchurComplement, not deep in DenseLU.

    Without this check, the error bubbles up from torch.linalg.solve as the
    cryptic "A must be batches of square matrices, but they are X by Y
    matrices", which doesn't tell the user *which* algorithmic constraint is
    violated. The named-group error lets them re-check
    `(residual_primary_group, unknown_primary_group)` against the layout.
    """
    # Asymmetric group sizes (5, 3) + cross-index (rp=0, up=1) ⇒
    # A_pp is the (0, 1) block of shape (5, 3) — not square.
    batch = torch.Size([1])
    A_full = _spd_matrix(8, batch, seed=99)
    b_full = torch.zeros(*batch, 8)
    A, b = _make_assembled(A_full, b_full, (5, 3))
    schur = SchurComplement(residual_primary_group=0, unknown_primary_group=1)
    with pytest.raises(ValueError, match="A_pp must be square"):
        schur.solve(A, b)


# ---------------------------------------------------------------------------
# End-to-end Newton tests on 2-group residuals
# ---------------------------------------------------------------------------


class _PowerResidual(Model):
    """``r_p = u_p^2 - c_p``, ``r_s = u_s^3 - c_s`` — closed-form root per group."""

    input_spec = {"u_p": Scalar, "u_s": Scalar, "c_p": Scalar, "c_s": Scalar}
    output_spec = {"u_p_residual": Scalar, "u_s_residual": Scalar}

    def forward(self, u_p: Scalar, u_s: Scalar, c_p: Scalar, c_s: Scalar, v=None):  # type: ignore[override]
        r_p = u_p * u_p - c_p
        r_s = u_s * u_s * u_s - c_s
        if v is None:
            return r_p, r_s
        # Diagonal Jacobian blocks: d(r_p)/d(u_p) = 2u_p, d(r_s)/d(u_s) = 3u_s^2
        # off-diagonals are zero (each residual depends only on its own unknown).
        v_out_p = self.apply_chain_rule(
            v,
            "u_p_residual",
            {
                "u_p": lambda V, up=u_p: 2.0 * up * V,
                "c_p": lambda V: -V,
            },
            output=r_p,
        )
        v_out_s = self.apply_chain_rule(
            v,
            "u_s_residual",
            {
                "u_s": lambda V, us=u_s: 3.0 * us * us * V,
                "c_s": lambda V: -V,
            },
            output=r_s,
        )
        # Merge per-output dicts.
        merged: ChainRuleDict = {**v_out_p, **v_out_s}
        return r_p, r_s, merged


class _CoupledResidual(Model):
    """2x2 coupled residual with nonzero off-diagonal Jacobian blocks.

    ``r_p = u_p^2 + u_s - c_p``, ``r_s = u_p + u_s - c_s``. Closed form:
    ``u_p = ((c_s - 1)/2)`` solves the linear elimination… Actually we don't
    rely on a closed form — instead we cross-check the Schur solution against
    DenseLU on the flattened reformulation (same linear operator, different
    factorisation, must agree to machine precision). The off-diagonal coupling
    exercises ``A_ps`` and ``A_sp`` non-trivially.
    """

    input_spec = {"u_p": Scalar, "u_s": Scalar, "c_p": Scalar, "c_s": Scalar}
    output_spec = {"u_p_residual": Scalar, "u_s_residual": Scalar}

    def forward(self, u_p: Scalar, u_s: Scalar, c_p: Scalar, c_s: Scalar, v=None):  # type: ignore[override]
        r_p = u_p * u_p + u_s - c_p
        r_s = u_p + u_s - c_s
        if v is None:
            return r_p, r_s
        v_out_p = self.apply_chain_rule(
            v,
            "u_p_residual",
            {
                "u_p": lambda V, up=u_p: 2.0 * up * V,
                "u_s": lambda V: V,
                "c_p": lambda V: -V,
            },
            output=r_p,
        )
        v_out_s = self.apply_chain_rule(
            v,
            "u_s_residual",
            {
                "u_p": lambda V: V,
                "u_s": lambda V: V,
                "c_s": lambda V: -V,
            },
            output=r_s,
        )
        merged: ChainRuleDict = {**v_out_p, **v_out_s}
        return r_p, r_s, merged


def _build_two_group_system(model: Model) -> ModelNonlinearSystem:
    return ModelNonlinearSystem(
        model, unknowns=[["u_p"], ["u_s"]], residuals=[["u_p_residual"], ["u_s_residual"]]
    )


def _build_one_group_system(model: Model) -> ModelNonlinearSystem:
    return ModelNonlinearSystem(
        model,
        unknowns=[["u_p", "u_s"]],
        residuals=[["u_p_residual", "u_s_residual"]],
    )


def test_newton_schur_power_converges_to_closed_form_root():
    """SchurComplement-driven Newton recovers u_p = sqrt(c_p), u_s = c_s^(1/3)."""
    sys = _build_two_group_system(_PowerResidual())
    c_p = torch.tensor([4.0, 9.0])  # batch 2
    c_s = torch.tensor([8.0, 27.0])
    sys.initialize(
        u={"u_p": torch.tensor([0.5, 0.5]), "u_s": torch.tensor([0.5, 0.5])},
        g={"c_p": c_p, "c_s": c_s},
    )
    newton = Newton(
        linear_solver=SchurComplement(primary_solver=DenseLU(), schur_solver=DenseLU()),
        atol=1e-12,
        rtol=1e-12,
    )
    res = newton.solve(sys)
    assert res.ret == RetCode.SUCCESS
    sol = sys.u().disassemble()
    assert torch.allclose(sol["u_p"], torch.sqrt(c_p), atol=1e-10)
    assert torch.allclose(sol["u_s"], c_s ** (1.0 / 3.0), atol=1e-10)


def test_newton_schur_coupled_matches_dense_lu_solution():
    """Schur and DenseLU paths converge to identical unknowns on a coupled residual.

    Closed-form analysis of ``_CoupledResidual``:
    ``u_p² - u_p + (c_s - c_p) = 0`` ⇒ real roots iff ``4(c_s - c_p) ≤ 1``.
    The chosen ``(c_p, c_s)`` pairs satisfy this with a well-conditioned
    Jacobian at the chosen initial guess.
    """
    sys_schur = _build_two_group_system(_CoupledResidual())
    sys_dense = _build_one_group_system(_CoupledResidual())
    c_p = torch.tensor([4.0, 6.0])
    c_s = torch.tensor([2.0, 3.0])
    init_u = {"u_p": torch.tensor([1.5, 1.5]), "u_s": torch.tensor([0.5, 0.5])}
    givens = {"c_p": c_p, "c_s": c_s}

    sys_schur.initialize(u=init_u, g=givens)
    sys_dense.initialize(u=init_u, g=givens)

    newton_schur = Newton(
        linear_solver=SchurComplement(primary_solver=DenseLU(), schur_solver=DenseLU()),
        atol=1e-12,
        rtol=1e-12,
    )
    newton_dense = Newton(linear_solver=DenseLU(), atol=1e-12, rtol=1e-12)

    assert newton_schur.solve(sys_schur).ret == RetCode.SUCCESS
    assert newton_dense.solve(sys_dense).ret == RetCode.SUCCESS

    sol_schur = sys_schur.u().disassemble()
    sol_dense = sys_dense.u().disassemble()
    assert torch.allclose(sol_schur["u_p"], sol_dense["u_p"], rtol=1e-10, atol=1e-10)
    assert torch.allclose(sol_schur["u_s"], sol_dense["u_s"], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("rp,up", [(0, 0), (1, 1)])
def test_newton_schur_alternate_group_indices(rp: int, up: int):
    """Non-default (rp, up) reassembles into the correct unknown slots.

    Restricted to ``rp == up`` because the unknowns have unequal "sizes" only
    in the sense of position (both groups carry a single Scalar each here, so
    every combo would work, but mirroring the C++ test we restrict to the
    diagonal pairs).
    """
    sys = _build_two_group_system(_PowerResidual())
    c_p = torch.tensor([4.0])
    c_s = torch.tensor([8.0])
    sys.initialize(
        u={"u_p": torch.tensor([0.5]), "u_s": torch.tensor([0.5])},
        g={"c_p": c_p, "c_s": c_s},
    )
    newton = Newton(
        linear_solver=SchurComplement(
            residual_primary_group=rp,
            unknown_primary_group=up,
            primary_solver=DenseLU(),
            schur_solver=DenseLU(),
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    assert newton.solve(sys).ret == RetCode.SUCCESS
    sol = sys.u().disassemble()
    assert torch.allclose(sol["u_p"], torch.sqrt(c_p), atol=1e-10)
    assert torch.allclose(sol["u_s"], c_s ** (1.0 / 3.0), atol=1e-10)


# ---------------------------------------------------------------------------
# Sub-batch awareness
# ---------------------------------------------------------------------------


def test_schur_solve_broadcasts_over_sub_batch_axes():
    """A BLOCK layout with per-site systems solves correctly via Schur.

    introduced the sub-batch concept (per-cell / per-bin axes that
    broadcast like a batch dim but encode structured per-site information).
    The Schur algorithm doesn't know about it explicitly — but every
    primitive it composes (``torch.matmul`` for ``@``, ``torch.linalg.solve``
    inside DenseLU, blockwise subtraction) broadcasts over leading batch
    axes naturally. This test pins that property end-to-end: a layout with
    sub-batch shape ``(L,)`` carries through the full 6-step factorisation
    and produces per-site solutions that agree with per-site DenseLU on the
    flattened system.
    """
    L = 4  # sub-batch length
    batch = torch.Size([2, L])  # dynamic batch + sub-batch
    s0, s1 = 3, 2
    n = s0 + s1
    A_full = _spd_matrix(n, batch, seed=23)
    b_full = torch.randn(*batch, n, generator=torch.Generator().manual_seed(29))

    # Manually attach the sub-batch metadata to the layout so the BLOCK
    # property holds. _make_assembled doesn't know about sub-batch, so we
    # build the layout ourselves.
    group0 = [f"g0v{i}" for i in range(s0)]
    group1 = [f"g1v{i}" for i in range(s1)]
    specs: dict[str, type] = {name: Scalar for name in (*group0, *group1)}
    sub_batch = {name: torch.Size([L]) for name in (*group0, *group1)}
    layout = AxisLayout([group0, group1], specs, sub_batch_shapes=sub_batch)
    assert layout.istructure == IStructure.BLOCK
    A = AssembledMatrix(
        layout,
        layout,
        [
            [A_full[..., :s0, :s0], A_full[..., :s0, s0:]],
            [A_full[..., s0:, :s0], A_full[..., s0:, s0:]],
        ],
    )
    b = AssembledVector(layout, [b_full[..., :s0], b_full[..., s0:]])

    x = SchurComplement().solve(A, b)
    assert isinstance(x, AssembledVector)
    # Schur preserves the leading (2, L) dims via broadcasting.
    assert x.tensors[0].shape == torch.Size([2, L, s0])
    assert x.tensors[1].shape == torch.Size([2, L, s1])

    x_flat = torch.cat(x.tensors, dim=-1)
    x_ref = torch.linalg.solve(A_full, b_full.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(x_flat, x_ref, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Per-group istructure plumbing
# ---------------------------------------------------------------------------


def test_per_group_istructure_propagates_through_layout():
    """``istructure = (BLOCK, DENSE)`` lands per-group on ulayout/blayout."""
    sys = ModelNonlinearSystem(
        _PowerResidual(),
        unknowns=[["u_p"], ["u_s"]],
        residuals=[["u_p_residual"], ["u_s_residual"]],
        istructure=[IStructure.BLOCK, IStructure.DENSE],
    )
    assert sys.ulayout.group_istructures == (IStructure.BLOCK, IStructure.DENSE)
    assert sys.blayout.group_istructures == (IStructure.BLOCK, IStructure.DENSE)
    # Aggregate single-value view: BLOCK iff every group is BLOCK.
    assert sys.ulayout.istructure == IStructure.DENSE


def test_istructure_length_mismatch_rejected():
    """One istructure entry per unknown group, no more, no less."""
    with pytest.raises(ValueError, match="one entry per unknown group"):
        ModelNonlinearSystem(
            _PowerResidual(),
            unknowns=[["u_p"], ["u_s"]],
            residuals=[["u_p_residual"], ["u_s_residual"]],
            istructure=[IStructure.BLOCK],
        )


# ---------------------------------------------------------------------------
# HIT factory parsing
# ---------------------------------------------------------------------------


def test_schur_complement_loads_from_solvers_hit_file():
    """``type = SchurComplement`` in tests/unit/solvers/solvers.i wires up correctly."""
    f = load_input(SOLVERS_I)
    schur = f.get_solver("schur")
    assert isinstance(schur, SchurComplement)
    # Group indices both 0 per the HIT file.
    assert schur._rp == 0
    assert schur._rs == 1
    assert schur._up == 0
    assert schur._us == 1
    # Nested sub-solvers are both DenseLU (HIT name `lu`).
    assert isinstance(schur._primary_solver, DenseLU)
    assert isinstance(schur._schur_solver, DenseLU)


def test_newton_with_schur_loads_from_solvers_hit_file():
    """``[newton_sc]`` constructs Newton wrapping SchurComplement via the factory."""
    f = load_input(SOLVERS_I)
    newton = f.get_solver("newton_sc")
    assert isinstance(newton, Newton)
    assert isinstance(newton.linear_solver, SchurComplement)
