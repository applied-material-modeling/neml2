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
from neml2.types import MRP, SSR4, MillerIndex, Scalar

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
