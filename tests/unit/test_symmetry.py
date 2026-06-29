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

"""``neml2.ops.symmetry`` — crystal point-group operators in orbifold notation.
Self-contained group-theory invariants (no dependency on the v2 build, which is
used separately as a one-off parity oracle)."""

import pytest
import torch

from neml2.data import cubic_symmetry_operators
from neml2.ops import symmetry
from neml2.ops.symmetry import ORBIFOLDS

_EXPECTED_COUNTS = {
    "432": 24,
    "23": 12,
    "622": 12,
    "32": 6,
    "6": 6,
    "3": 3,
    "42": 8,
    "4": 4,
    "222": 4,
    "2": 2,
    "1": 1,
}


def _f64():
    torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("orbifold", ORBIFOLDS)
def test_operator_count(orbifold):
    _f64()
    S = symmetry(orbifold)
    assert S.sub_batch_ndim == 1
    assert S.data.shape == torch.Size([_EXPECTED_COUNTS[orbifold], 3, 3])


@pytest.mark.parametrize("orbifold", ORBIFOLDS)
def test_operators_are_proper_rotations(orbifold):
    """Every operator is orthonormal with det +1 (a pure rotation)."""
    _f64()
    R = symmetry(orbifold).data
    rrt = torch.einsum("nij,nkj->nik", R, R)
    assert torch.allclose(rrt, torch.eye(3).expand_as(rrt), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-12)


@pytest.mark.parametrize("orbifold", ORBIFOLDS)
def test_identity_is_present(orbifold):
    _f64()
    R = symmetry(orbifold).data
    eye = torch.eye(3)
    assert bool(((R - eye).abs().amax(dim=(-1, -2)) < 1e-12).any())


def test_cubic_is_closed_under_composition():
    """The 24 cubic operators form a group: every product is again an operator."""
    _f64()
    R = symmetry("432").data  # (24, 3, 3)
    prods = torch.einsum("aij,bjk->abik", R, R).reshape(-1, 3, 3)
    for p in prods:
        match = (R - p).abs().amax(dim=(-1, -2)) < 1e-10
        assert bool(match.any()), "product fell outside the operator set"


def test_432_matches_data_cubic_symmetry_operators():
    """symmetry('432') is the same operator set as neml2.data.cubic_symmetry_operators."""
    _f64()
    A = symmetry("432").data
    B = cubic_symmetry_operators().data
    assert A.shape == B.shape
    for a in A:
        match = (B - a).abs().amax(dim=(-1, -2)) < 1e-12
        assert bool(match.any()), "operator missing from cubic_symmetry_operators"


def test_unknown_orbifold_raises():
    with pytest.raises(ValueError, match="Unknown crystal class"):
        symmetry("777")
