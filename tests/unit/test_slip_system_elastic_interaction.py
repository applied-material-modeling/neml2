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

"""Tests for :class:`SlipSystemElasticInteraction`.

The coupling matrix $A_{ij} = \\Delta t\\, M_i : \\mathbb{C} : M_j$ is contracted
in the lab frame, which means both Schmid operands *and* the stiffness must be
rotated. Rotating some but not all of them produces a matrix that is symmetric,
positive semi-definite and entirely plausible, but wrong -- so the frame
convention is pinned here rather than left to inspection.

The sharp check is :func:`test_isotropic_coupling_is_orientation_independent`:
an isotropic stiffness commutes with rotation and the Schmid inner products are
invariant under a common rotation, so with isotropic elasticity $A$ cannot
depend on the orientation. It does depend on it if the frames are mismatched.
"""

from __future__ import annotations

import pytest
import torch

from neml2.data import CrystalGeometry
from neml2.data.CubicCrystal import cubic_symmetry_operators
from neml2.models.solid_mechanics.crystal_plasticity import SlipSystemElasticInteraction
from neml2.types import (
    MRP,
    SR2,
    SSR4,
    WR2,
    MillerIndex,
    Scalar,
    euler_rodrigues,
    r2_from_sr2,
    r2_from_wr2,
    rotate,
    sym,
)

MU = 40000.0
LAM = 60000.0
DT = 0.1
NSLIP = 12


@pytest.fixture
def fcc() -> CrystalGeometry:
    return CrystalGeometry(
        sym_ops=cubic_symmetry_operators(),
        lattice_vectors=torch.eye(3, dtype=torch.float64),
        slip_directions=MillerIndex(torch.tensor([1.0, 1.0, 0.0])),
        slip_planes=MillerIndex(torch.tensor([1.0, 1.0, 1.0])),
    )


def _isotropic() -> SSR4:
    """Isotropic stiffness in Mandel form: ``lam I(x)I + 2 mu I_sym``."""
    c = torch.zeros(6, 6, dtype=torch.float64)
    c[:3, :3] = LAM
    c += 2.0 * MU * torch.eye(6, dtype=torch.float64)
    return SSR4(c)


def _coupling(fcc: CrystalGeometry, stiffness: SSR4, orientation: torch.Tensor) -> torch.Tensor:
    model = SlipSystemElasticInteraction(crystal_geometry=fcc, elastic_stiffness_tensor=stiffness)
    # Called positionally, as the ResolvedShear tests do: the schema's variable
    # names are irrelevant to what is being checked here.
    A = model(
        MRP(orientation),
        Scalar(torch.tensor(DT, dtype=torch.float64)),
        Scalar(torch.tensor(0.0, dtype=torch.float64)),
    )
    return A.data.detach()  # data-ok: test assertion on the numeric result


def test_shape_and_symmetry(fcc):
    A = _coupling(fcc, _isotropic(), torch.zeros(3, dtype=torch.float64))
    assert A.shape == (NSLIP, NSLIP)
    assert torch.allclose(A, A.T, rtol=0, atol=1e-9)


def test_positive_semi_definite(fcc):
    """PSD is what brackets each coordinate solve downstream, so it is load-bearing."""
    A = _coupling(fcc, _isotropic(), torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    eig = torch.linalg.eigvalsh(0.5 * (A + A.T))
    assert eig.min() > -1e-6 * A.abs().max()
    assert (torch.diagonal(A) >= 0).all()


def test_isotropic_coupling_is_orientation_independent(fcc):
    """The frame check.

    An isotropic stiffness commutes with rotation, and the Schmid inner products
    are invariant under a common rotation, so ``A`` must be identical at every
    orientation. If the stiffness were left in the crystal frame while the Schmid
    tensors were rotated (or vice versa), this would fail.
    """
    C = _isotropic()
    ref = _coupling(fcc, C, torch.zeros(3, dtype=torch.float64))
    for r in (
        torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([-0.2, 0.5, 0.1], dtype=torch.float64),
        torch.tensor([0.7, -0.4, 0.9], dtype=torch.float64),
    ):
        assert torch.allclose(_coupling(fcc, C, r), ref, rtol=1e-10, atol=1e-8)


def test_isotropic_matches_the_closed_form(fcc):
    """With isotropic elasticity and deviatoric Schmid tensors, ``A = 2 mu dt M:M``."""
    A = _coupling(fcc, _isotropic(), torch.zeros(3, dtype=torch.float64))
    M = fcc.M.data.detach()  # data-ok: building the reference by hand
    # Mandel form already carries the sqrt(2) weights, so this inner product is
    # the full double contraction.
    expected = 2.0 * MU * DT * (M @ M.T)
    assert torch.allclose(A, expected, rtol=1e-9, atol=1e-8)


def test_a_promoted_stiffness_survives_construction(tmp_path):
    """Construction contract: a promoted parameter must stay in ``input_spec``.

    This model's ``__init__`` names only some of its schema fields, so
    ``Model.from_hit`` applies the rest through ``_store_schema_values`` *after*
    construction -- and that rebuilds ``input_spec`` from the class-level spec.
    ``declare_typed_parameter`` (mode 3) had already added an input keyed by the
    provider's output name, which the schema knows nothing about, so the rebuild
    used to drop it while ``_promoted_params`` went on expecting it. The model
    then died at forward time inside ``_get_param`` with a bare ``IndexError``.

    Kept as a real-model test rather than a synthetic one because the shape that
    triggers it -- custom ``__init__`` + unconsumed name-bearing fields +
    promotable parameter -- is what this class actually has.
    """
    inp = tmp_path / "promoted.i"
    inp.write_text("""
[Tensors]
  [a]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []
[]
[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]
[Models]
  [stiffness_provider]
    type = IsotropicElasticityTensor
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [coupling]
    type = SlipSystemElasticInteraction
    orientation = 'orientation~1'
    elastic_stiffness_tensor = 'stiffness_provider'
  []
[]
""")
    from neml2 import load_input  # noqa: PLC0415

    model = load_input(inp).get_model("coupling")
    promoted = [p.input_name for p in model._promoted_params.values()]
    assert promoted, "the stiffness should have been promoted to an input"
    for name in promoted:
        assert name in model.input_spec, (
            f"promoted parameter input {name!r} was dropped from input_spec; "
            f"input_spec is {list(model.input_spec)}"
        )


def test_omits_the_spin_convection_term(fcc):
    r"""Pins the documented approximation against the exact condensed coupling.

    ``ElasticStrainRate`` carries $\dot\varepsilon^e = d - d^p + \Omega[\varepsilon^e]$
    with $\Omega[V] = [W, V]$, so the elastic block of the Jacobian is
    $I - \Delta t\,\Omega$ and the exact coupling is
    $\Delta t\,M^{\mathsf T}\mathbb{C}(I - \Delta t\,\Omega)^{-1}M$. Because
    $\Omega$ generates a rotation it is skew-adjoint, which splits the deviation
    cleanly: **skew** at $O(\Delta t\lVert\Omega\rVert)$ and **symmetric** at
    $O((\Delta t\lVert\Omega\rVert)^2)$.

    This model computes the symmetric part on purpose -- coordinate descent needs
    a symmetric $A$ to have a potential to descend, and the exact coupling is not
    symmetric. The test checks both sides of that claim: the symmetric agreement
    is second order, and the discarded skew part is genuinely first order and
    genuinely there (a lower bound, so this cannot pass by the term vanishing).
    """
    w = torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64)
    eps = DT * 2.0 * w.norm().item()  # dt * spectral radius of Omega

    # Omega as a 6x6 Mandel matrix, built through the ops the leaf itself uses.
    W = r2_from_wr2(WR2(w))
    cols = []
    for k in range(6):
        e = torch.zeros(6, dtype=torch.float64)
        e[k] = 1.0
        V = r2_from_sr2(SR2(e))
        cols.append(sym(W @ V - V @ W).data)  # data-ok: building a reference matrix
    Omega = torch.stack(cols, dim=-1)
    assert torch.allclose(Omega, -Omega.T, rtol=0, atol=1e-12), "Omega must be skew-adjoint"

    r = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    A = _coupling(fcc, _isotropic(), r)

    # Exact coupling. Schmid tensors are traceless and Omega preserves that, so
    # the isotropic stiffness acts as 2*mu throughout.
    R = euler_rodrigues(MRP(r))
    M = rotate(fcc.M, R.sub_batch.unsqueeze(-1)).data  # data-ok
    Jxx_inv = torch.linalg.inv(torch.eye(6, dtype=torch.float64) - DT * Omega)
    A_exact = 2.0 * MU * DT * (M @ Jxx_inv @ M.T)

    D = (A_exact - A) / A.abs().max()
    sym_part = (0.5 * (D + D.T)).abs().max().item()
    skew_part = (0.5 * (D - D.T)).abs().max().item()

    assert sym_part < 2.0 * eps**2, f"symmetric deviation {sym_part:.2e} is not second order"
    assert 0.25 * eps < skew_part < eps, f"skew deviation {skew_part:.2e} is not first order"
    # A skew matrix has zero diagonal, so the bracket precondition A_ii >= 0 is
    # unaffected by the omission at first order -- worth pinning separately.
    assert torch.diagonal(D).abs().max().item() < 2.0 * eps**2


def test_scales_with_the_time_step(fcc):
    """``A`` is linear in dt -- it is an increment, not a rate."""
    C = _isotropic()
    r = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    model = SlipSystemElasticInteraction(crystal_geometry=fcc, elastic_stiffness_tensor=C)

    def at(dt: float) -> torch.Tensor:
        A = model(
            MRP(r),
            Scalar(torch.tensor(dt, dtype=torch.float64)),
            Scalar(torch.tensor(0.0, dtype=torch.float64)),
        )
        return A.data.detach()  # data-ok

    assert torch.allclose(at(0.2), 2.0 * at(0.1), rtol=1e-12, atol=1e-12)
