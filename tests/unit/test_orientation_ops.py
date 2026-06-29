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

"""Orientation gap ops added for the crystallography port: the Quaternion type,
MRP <-> Quaternion <-> rotation-matrix conversions, the SO(3) distance / volume
element, and Vec cross / dot. Checked by mathematical invariants (no dependency
on the v2 build)."""

import math

import torch

from neml2.types import (
    MRP,
    R2,
    Quaternion,
    Scalar,
    Vec,
    cross,
    dist,
    dV,
    euler_rodrigues,
    gdist,
    inverse,
    quaternion_dist,
    quaternion_rotation_matrix,
    shadow,
    to_quaternion,
    vdot,
)


def _f64():
    torch.set_default_dtype(torch.float64)


def _rand_mrp(n=8):
    # MRPs with modest magnitude so we stay well inside one shadow chart.
    return MRP(0.3 * torch.randn(n, 3, dtype=torch.float64))


# ---------- Vec cross / dot ----------


def test_cross_right_handed_basis():
    _f64()
    e1 = Vec(torch.tensor([1.0, 0.0, 0.0]))
    e2 = Vec(torch.tensor([0.0, 1.0, 0.0]))
    e3 = Vec(torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(cross(e1, e2).data, e3.data)
    assert torch.allclose(cross(e2, e1).data, -e3.data)  # antisymmetry


def test_vdot_orthonormal():
    _f64()
    e1 = Vec(torch.tensor([1.0, 0.0, 0.0]))
    e2 = Vec(torch.tensor([0.0, 1.0, 0.0]))
    assert float(vdot(e1, e1).data) == 1.0
    assert float(vdot(e1, e2).data) == 0.0


def test_cross_vdot_preserve_sub_batch():
    _f64()
    b = Vec(torch.randn(4, 3), sub_batch_ndim=1)
    # the per-cell sub-batch axis must survive both products
    assert cross(b, b).sub_batch_ndim == 1
    assert vdot(b, b).sub_batch_ndim == 1


# ---------- MRP <-> Quaternion <-> R2 consistency ----------


def test_quaternion_matrix_matches_euler_rodrigues():
    """quaternion_rotation_matrix(to_quaternion(r)) must equal euler_rodrigues(r)
    -- two independent derivations of the same rotation matrix."""
    _f64()
    r = _rand_mrp()
    R_euler = euler_rodrigues(r).data
    R_quat = quaternion_rotation_matrix(to_quaternion(r)).data
    assert torch.allclose(R_euler, R_quat, atol=1e-12)


def test_from_matrix_round_trips_euler_rodrigues():
    """``MRP.from_matrix`` inverts the rotation-matrix map: feeding it
    ``euler_rodrigues(r)`` recovers ``r`` (modest magnitudes stay inside one
    shadow chart, ``theta < pi``)."""
    _f64()
    r = _rand_mrp()
    R = euler_rodrigues(r)
    r_back = MRP.from_matrix(R)
    assert torch.allclose(r_back.data, r.data, atol=1e-10)


def test_from_matrix_identity_is_zero():
    _f64()
    r = MRP.from_matrix(R2(torch.eye(3)))
    assert torch.allclose(r.data, torch.zeros(3))


def test_from_matrix_preserves_sub_batch():
    _f64()
    r = MRP(0.3 * torch.randn(4, 3, dtype=torch.float64), sub_batch_ndim=1)
    R = euler_rodrigues(r)
    assert MRP.from_matrix(R).sub_batch_ndim == 1


def test_identity_conversions():
    _f64()
    q = to_quaternion(MRP.identity())
    assert torch.allclose(q.data, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    R = quaternion_rotation_matrix(Quaternion.identity()).data
    assert torch.allclose(R, torch.eye(3))


def test_rotation_matrices_are_orthonormal():
    _f64()
    R = quaternion_rotation_matrix(to_quaternion(_rand_mrp())).data
    eye = torch.einsum("...ij,...kj->...ik", R, R)
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-12)


# ---------- inverse / shadow ----------


def test_inverse_is_negation_and_matrix_transpose():
    _f64()
    r = _rand_mrp()
    assert torch.allclose(inverse(r).data, -r.data)
    R = euler_rodrigues(r).data
    Rinv = euler_rodrigues(inverse(r)).data
    assert torch.allclose(Rinv, R.transpose(-1, -2), atol=1e-12)


def test_shadow_is_same_orientation():
    _f64()
    r = _rand_mrp()
    # The shadow MRP encodes the same physical rotation.
    assert torch.allclose(euler_rodrigues(shadow(r)).data, euler_rodrigues(r).data, atol=1e-10)


# ---------- distances / volume element ----------


def test_dist_zero_to_self_and_shadow():
    _f64()
    r = _rand_mrp()
    other = MRP(0.3 * torch.randn(8, 3))
    assert torch.allclose(dist(r, r).data, torch.zeros(8), atol=1e-9)
    # dist accounts for the shadow chart, so a rotation and its shadow are 0 apart
    assert torch.allclose(dist(r, shadow(r)).data, torch.zeros(8), atol=1e-9)
    # gdist (raw) is symmetric and non-negative
    assert torch.allclose(gdist(r, other).data, gdist(other, r).data, atol=1e-12)
    assert bool((dist(r, other).data >= -1e-12).all())


def test_dV_identity_and_positive():
    _f64()
    assert math.isclose(float(dV(MRP.identity()).data), 8.0 / math.pi, rel_tol=1e-12)
    assert bool((dV(_rand_mrp()).data > 0).all())


def test_quaternion_dist_zero_to_self():
    _f64()
    q = to_quaternion(_rand_mrp())
    assert torch.allclose(quaternion_dist(q, q).data, torch.zeros(8), atol=1e-7)


# ---------- constructors ----------


def test_from_axis_angle_z_rotation():
    _f64()
    theta = 0.7
    r = MRP.from_axis_angle(Vec(torch.tensor([0.0, 0.0, 1.0])), Scalar(torch.tensor(theta)))
    R = euler_rodrigues(r).data
    c, s = math.cos(theta), math.sin(theta)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(R, Rz, atol=1e-12)


def test_rotation_from_to_maps_v1_to_v2():
    _f64()
    v1 = Vec(torch.tensor([1.0, 0.0, 0.0]))
    v2 = Vec(torch.tensor([0.0, 1.0, 0.0]))
    r = MRP.rotation_from_to(v1, v2)
    R = euler_rodrigues(r).data
    mapped = R @ v1.data
    assert torch.allclose(mapped, v2.data, atol=1e-12)


def test_rotate_vec_matvec_and_broadcast():
    from neml2.types import R2, rotate  # noqa: PLC0415

    _f64()
    R = euler_rodrigues(_rand_mrp(1))  # (1,3,3)
    v = Vec(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(rotate(v, R).data, torch.einsum("nij,nj->ni", R.data, v.data))
    # a batch of operators (sub_batch) applied to a single direction broadcasts
    ops = R2(torch.stack([torch.eye(3), torch.eye(3)[[1, 0, 2]]]), sub_batch_ndim=1)
    out = rotate(Vec(torch.tensor([0.0, 0.0, 1.0])), ops)
    assert out.data.shape == torch.Size([2, 3])
    assert out.sub_batch_ndim == 1


def test_rand_is_uniform_orientation_shape_and_valid():
    _f64()
    r = MRP.rand(16)
    assert r.data.shape == torch.Size([16, 3])
    R = euler_rodrigues(r).data
    eye = torch.einsum("...ij,...kj->...ik", R, R)
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-10)
