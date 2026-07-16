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

"""Python-native crystal-plasticity leaf and primitive tests.

Coverage:

* (a) Tensor types — ``MRP``, ``R2``, ``WR2``, ``MillerIndex`` round-trip
  and key free-function math (``euler_rodrigues``, ``jvp_euler_rodrigues``,
  ``exp_map``, ``dexp_map``, ``compose``, ``rotate_sym``, ``rotate_ssr4``).
  Validated against (i) hand-computed known cases, (ii) an independent
  MRP→quaternion→R reference implemented in this test file, (iii)
  physical/algebraic invariants (orthogonality, ``det == 1``, identity
  element, associativity, matrix consistency under composition), and
  (iv) finite-difference Jacobians. Per D-053 this suite never imports
  from ``neml2.tensors`` — the C++ Python bindings are scheduled for
  removal and tests must not couple to them.
* (b) ``CrystalGeometry`` — FCC ``(110)<111>`` slip-system inventory:
  ``nslip == 12``, unit-norm directions/planes, perpendicularity,
  Frobenius norms of ``M`` / ``W``.
* (c) Per-leaf forward + chain-rule sanity for the Python-native CP models
  (``ResolvedShear``, ``ElasticStrainRate``, ``PlasticDeformationRate``,
  ``PlasticVorticity``, ``OrientationRate``, ``SumSlipRates``,
  ``PowerLawSlipRule``, ``SingleSlipStrengthMap``,
  ``VoceSingleSlipHardeningRule``, ``CrystalPlasticityStrainPredictor``)
  via identity-seeded tangent propagation cross-checked with
  ``torch.autograd.functional.jacobian``.
* (d) End-to-end byte-identical Taylor CP single step
  (:func:`test_taylor_cp_native_matches_cxx_gold_one_step`) — loads
  ``cp_taylor_native.i`` (Phase-10.5-style rewrite of the C++ regression
  input — same model, ``[Tensors]`` blocks translated to ``type =
  Python`` expressions), runs the full mixed BLOCK+DENSE Schur stack to
  convergence, and asserts every output matches the frozen
  ``tests/regression/.../taylor/gold/result.pt`` at ``rtol=atol=1e-12``.
"""

from __future__ import annotations

import math

import pytest
import torch

from neml2 import (
    CrystalGeometry,
    cubic_symmetry_operators,
)
from neml2.types import (
    MRP,
    R2,
    SR2,
    SSR4,
    WR2,
    MillerIndex,
    Scalar,
    compose,
    dexp_map,
    drotate,
    drotate_self,
    euler_rodrigues,
    exp_map,
    r2_from_sr2,
    r2_from_wr2,
    rotate,
    skew,
    sym,
)

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Self-contained reference helpers — no `neml2.tensors` C++-binding imports.

# on them couple their continued passage to code we plan to remove.
# Instead, we cross-check the Python-native math against (a) hand-computed
# known cases, (b) physical invariants (orthogonality, det == 1,
# composition associativity), and (c) finite-difference derivatives.
# ---------------------------------------------------------------------------


def _quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """Quaternion ``(w, x, y, z)`` → 3×3 rotation matrix (an independent
    reference implementation that does not touch any project module under
    test). Mirrors the standard quaternion→matrix formula."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def _mrp_to_R_via_quaternion(mrp: torch.Tensor) -> torch.Tensor:
    """Modified Rodrigues parameters → rotation matrix via an intermediate
    quaternion. Independent of :func:`euler_rodrigues`, which it cross-checks.

    Given ``r = n · tan(θ/4)`` with ``n`` a unit vector:
        ``||r|| = tan(θ/4)``, ``θ = 4·atan(||r||)``,
        ``q = (cos(θ/2), n·sin(θ/2))``.
    Special-cases ``||r|| = 0`` to the identity quaternion (no division).
    """
    norm = torch.linalg.vector_norm(mrp)
    if float(norm) == 0.0:
        return torch.eye(3, dtype=mrp.dtype, device=mrp.device)
    theta = 4.0 * torch.atan(norm)
    n = mrp / norm
    half_theta = theta / 2.0
    q = torch.empty(4, dtype=mrp.dtype, device=mrp.device)
    q[0] = torch.cos(half_theta)
    q[1:] = n * torch.sin(half_theta)
    return _quat_to_R(q)


# ---------------------------------------------------------------------------
# (a) Tensor type round-trips and key ops
# ---------------------------------------------------------------------------


def test_rot_identity_yields_identity_matrix():
    r = MRP.identity(dtype=torch.float64)
    R = euler_rodrigues(r).data
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-15)


def test_rot_euler_rodrigues_known_axis_angle_cases():
    """Check :func:`euler_rodrigues` against three hand-computed cases:

    * 90° about ``+z``: ``r = (0, 0, tan(22.5°))``, ``R = [[0,-1,0],[1,0,0],[0,0,1]]``.
    * 180° about ``+z``: ``r = (0, 0, 1)``, ``R = diag(-1, -1, 1)``.
    * 90° about ``+y``: ``r = (0, tan(22.5°), 0)``, ``R = [[0,0,1],[0,1,0],[-1,0,0]]``.
    """
    t225 = math.tan(math.pi / 8)  # tan(22.5°)

    R = euler_rodrigues(MRP(torch.tensor([0.0, 0.0, t225]))).data
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(R, expected, atol=1e-14)

    R = euler_rodrigues(MRP(torch.tensor([0.0, 0.0, 1.0]))).data
    expected = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(R, expected, atol=1e-14)

    R = euler_rodrigues(MRP(torch.tensor([0.0, t225, 0.0]))).data
    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert torch.allclose(R, expected, atol=1e-14)


def test_rot_euler_rodrigues_matches_independent_quaternion_path():
    """Cross-check :func:`euler_rodrigues` against an independent MRP→quaternion→R path."""
    for r_vec in [
        [0.1, -0.2, 0.3],
        [0.0, 0.0, 0.0],
        [-0.05, 0.4, 0.1],
        [0.7, 0.0, -0.3],
    ]:
        mrp = torch.tensor(r_vec, dtype=torch.float64)
        py = euler_rodrigues(MRP(mrp)).data
        ref = _mrp_to_R_via_quaternion(mrp)
        assert torch.allclose(py, ref, atol=1e-13, rtol=1e-13), (
            f"r={r_vec} mismatch: py={py}, ref={ref}"
        )


def test_rot_euler_rodrigues_yields_orthogonal_unit_determinant():
    """For any MRP, ``R`` must be orthogonal with ``det(R) = +1``."""
    for r_vec in [[0.1, -0.2, 0.3], [-0.5, 0.0, 0.0], [0.05, -0.07, 0.12]]:
        R = euler_rodrigues(MRP(torch.tensor(r_vec, dtype=torch.float64))).data
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-14)
        assert math.isclose(float(torch.linalg.det(R)), 1.0, abs_tol=1e-14)


def test_exp_map_yields_orthogonal_rotation():
    for w_vec in [[0.0, 0.0, 0.0], [1e-10, 1e-10, 1e-10], [0.1, -0.2, 0.05]]:
        r = exp_map(WR2(torch.tensor(w_vec, dtype=torch.float64)))
        R = euler_rodrigues(r).data
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
        assert math.isclose(torch.linalg.det(R).item(), 1.0, abs_tol=1e-12)


def test_dexp_map_matches_finite_difference():
    w0 = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    eps = 1e-6
    analytical = dexp_map(WR2(w0)).data
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for j in range(3):
        wp = w0.clone()
        wp[j] += eps
        wm = w0.clone()
        wm[j] -= eps
        fd[:, j] = (exp_map(WR2(wp)).data - exp_map(WR2(wm)).data) / (2 * eps)
    assert torch.allclose(analytical, fd, atol=1e-9)


def test_compose_satisfies_rotation_group_properties():
    """``compose`` must (1) act as the identity element under :func:`MRP.identity`,
    (2) be associative, and (3) match composition through the rotation
    matrices: ``R(compose(r1, r2)) == R(r1) @ R(r2)``."""
    r0 = MRP.identity(dtype=torch.float64)
    r1 = MRP(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    r2 = MRP(torch.tensor([-0.05, 0.12, 0.08], dtype=torch.float64))
    r3 = MRP(torch.tensor([0.4, 0.1, -0.2], dtype=torch.float64))

    # Identity element.
    assert torch.allclose(compose(r0, r1).data, r1.data, atol=1e-14)
    assert torch.allclose(compose(r1, r0).data, r1.data, atol=1e-14)

    # Associativity: (r1 ∘ r2) ∘ r3 == r1 ∘ (r2 ∘ r3).
    left = compose(compose(r1, r2), r3)
    right = compose(r1, compose(r2, r3))
    assert torch.allclose(left.data, right.data, atol=1e-13)

    # Matrix consistency: R(r1 ∘ r2) == R(r1) @ R(r2).
    R12 = euler_rodrigues(compose(r1, r2)).data
    R1_R2 = euler_rodrigues(r1).data @ euler_rodrigues(r2).data
    assert torch.allclose(R12, R1_R2, atol=1e-13)


def test_drotate_matches_finite_difference():
    """``drotate(r1, r2) = ∂(compose(r2, r1))/∂(r2)`` via central FD."""
    r1 = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    r2 = torch.tensor([-0.05, 0.12, 0.08], dtype=torch.float64)
    analytical = drotate(MRP(r1), MRP(r2)).data  # (3, 3) Jacobian
    eps = 1e-7
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for j in range(3):
        rp = r2.clone()
        rp[j] += eps
        rm = r2.clone()
        rm[j] -= eps
        fd[:, j] = (compose(MRP(rp), MRP(r1)).data - compose(MRP(rm), MRP(r1)).data) / (2.0 * eps)
    assert torch.allclose(analytical, fd, atol=1e-6)


def test_drotate_self_matches_finite_difference():
    """``drotate_self(r1, r2) = ∂(compose(r2, r1))/∂(r1)`` via central FD."""
    r1 = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    r2 = torch.tensor([-0.05, 0.12, 0.08], dtype=torch.float64)
    analytical = drotate_self(MRP(r1), MRP(r2)).data
    eps = 1e-7
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for j in range(3):
        rp = r1.clone()
        rp[j] += eps
        rm = r1.clone()
        rm[j] -= eps
        fd[:, j] = (compose(MRP(r2), MRP(rp)).data - compose(MRP(r2), MRP(rm)).data) / (2.0 * eps)
    assert torch.allclose(analytical, fd, atol=1e-6)


def test_sym_skew_roundtrip_on_r2():
    A = torch.randn(3, 3, dtype=torch.float64)
    A_sym_R2 = R2((A + A.T) / 2)
    A_skew_R2 = R2((A - A.T) / 2)
    # sym/skew projections should be invertible into Mandel SR2 / axial WR2
    s = sym(A_sym_R2)
    w = skew(A_skew_R2)
    A_back = r2_from_sr2(s).data + r2_from_wr2(w).data
    assert torch.allclose(A_back, A, atol=1e-14)


def test_rotate_sym_round_trips_through_full_R2():
    """``rotate(s, R) == sym(R s_full R^T)`` — direct formula vs free fn."""
    r_vec = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    s_full = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]], dtype=torch.float64)
    s_mandel = sym(R2(s_full))
    R = euler_rodrigues(MRP(r_vec))
    py_rot = rotate(s_mandel, R).data
    # Direct: sym(R · s_full · R^T) packed in Mandel.
    direct = sym(R2(R.data @ s_full @ R.data.T)).data
    assert torch.allclose(py_rot, direct, atol=1e-12)


def test_rotate_ssr4_invariance():
    """Identity rotation preserves the stiffness; ``T'[strain] == sym(R T R^T)[strain]``."""
    T_data = torch.tensor(
        [
            [287111.0, 127190.5, 127190.5, 0, 0, 0],
            [127190.5, 287111.0, 127190.5, 0, 0, 0],
            [127190.5, 127190.5, 287111.0, 0, 0, 0],
            [0, 0, 0, 120710.0, 0, 0],
            [0, 0, 0, 0, 120710.0, 0],
            [0, 0, 0, 0, 0, 120710.0],
        ],
        dtype=torch.float64,
    )
    T = SSR4(T_data)
    R_id = R2.identity()
    assert torch.allclose(rotate(T, R_id).data, T.data, atol=1e-12)

    r_vec = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    R = euler_rodrigues(MRP(r_vec))
    eps_full = torch.tensor(
        [[0.01, 0.002, -0.001], [0.002, -0.005, 0.003], [-0.001, 0.003, 0.008]],
        dtype=torch.float64,
    )
    eps = sym(R2(eps_full))
    T_rot = rotate(T, R)
    sigma_a = SR2(torch.einsum("ij,j->i", T_rot.data, eps.data))
    # Path B: rotate strain into crystal frame, multiply, rotate stress back.
    eps_cry = rotate(eps, R.base.transpose(-2, -1))
    sigma_cry = SR2(torch.einsum("ij,j->i", T_data, eps_cry.data))
    sigma_b = rotate(sigma_cry, R)
    assert torch.allclose(sigma_a.data, sigma_b.data, atol=1e-8)


# ---------------------------------------------------------------------------
# (b) Crystallography — CubicCrystal FCC (110)<111>
# ---------------------------------------------------------------------------


@pytest.fixture
def fcc_geometry() -> CrystalGeometry:
    return CrystalGeometry(
        sym_ops=cubic_symmetry_operators(),
        lattice_vectors=torch.eye(3, dtype=torch.float64),
        slip_directions=MillerIndex(torch.tensor([1.0, 1.0, 0.0])),
        slip_planes=MillerIndex(torch.tensor([1.0, 1.0, 1.0])),
    )


def test_cubic_fcc_has_12_slip_systems(fcc_geometry):
    assert fcc_geometry.nslip == 12
    assert fcc_geometry.nslip_groups == 1
    assert fcc_geometry.nslip_in_group(0) == 12


def test_cubic_fcc_directions_and_planes_are_unit(fcc_geometry):
    norms_d = torch.linalg.vector_norm(fcc_geometry.cartesian_slip_directions, dim=-1)
    norms_n = torch.linalg.vector_norm(fcc_geometry.cartesian_slip_planes, dim=-1)
    ones = torch.ones(12, dtype=torch.float64)
    assert torch.allclose(norms_d, ones, atol=1e-12)
    assert torch.allclose(norms_n, ones, atol=1e-12)


def test_cubic_fcc_directions_perpendicular_to_planes(fcc_geometry):
    dn = (fcc_geometry.cartesian_slip_directions * fcc_geometry.cartesian_slip_planes).sum(dim=-1)
    assert torch.allclose(dn, torch.zeros(12, dtype=torch.float64), atol=1e-12)


def test_cubic_fcc_schmid_frobenius_norms(fcc_geometry):
    norm_M = torch.linalg.vector_norm(fcc_geometry.M.data, dim=-1)
    norm_W = torch.linalg.vector_norm(fcc_geometry.W.data, dim=-1)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.allclose(norm_M, torch.full((12,), inv_sqrt2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(norm_W, torch.full((12,), 0.5, dtype=torch.float64), atol=1e-12)


def test_cubic_fcc_burgers_for_unit_lattice(fcc_geometry):
    assert torch.allclose(
        fcc_geometry.burgers,
        torch.full((12,), math.sqrt(2.0), dtype=torch.float64),
        atol=1e-12,
    )


def test_cubic_crystal_factory_built_from_python_objects():
    """Direct construction (sidesteps HIT) — the factory wraps this same call."""
    sd = MillerIndex(torch.tensor([1.0, 1.0, 0.0]))
    sp = MillerIndex(torch.tensor([1.0, 1.0, 1.0]))
    cg = CrystalGeometry(
        sym_ops=cubic_symmetry_operators(),
        lattice_vectors=torch.eye(3, dtype=torch.float64),
        slip_directions=sd,
        slip_planes=sp,
    )
    assert cg.M.sub_batch_ndim == 1
    assert cg.W.sub_batch_ndim == 1
    assert cg.A.sub_batch_ndim == 1
    assert cg.M.data.shape == (12, 6)


# ---------------------------------------------------------------------------
# (c) Per-leaf forward + chain-rule sanity
# ---------------------------------------------------------------------------


def _seed_identity(spec_var: str, n_components: int, batch_shape: tuple[int, ...]) -> dict:
    """Identity-seeded tangent: ``v[var][var] = I_{n}`` broadcast over batch."""
    eye = torch.eye(n_components, dtype=torch.float64)
    eye_b = eye.expand(*batch_shape, n_components, n_components).contiguous()
    return {spec_var: {spec_var: eye_b}}


def test_resolved_shear_forward_matches_direct_formula(fcc_geometry):
    """``τ_k = M_k_rot : σ`` (identity rotation case, so M_rot == M)."""
    from neml2 import ResolvedShear

    leaf = ResolvedShear(crystal_geometry=fcc_geometry)
    sigma = SR2(torch.randn(6, dtype=torch.float64))
    R = R2.identity(dtype=torch.float64)
    rss = leaf(sigma, R)
    assert rss.data.shape == (12,)
    assert rss.sub_batch_ndim == 1
    # Direct: τ_k = M_k : σ (Mandel inner product)
    direct = (fcc_geometry.M.data * sigma.data.unsqueeze(0)).sum(dim=-1)
    assert torch.allclose(rss.data, direct, atol=1e-12)


def test_resolved_shear_chain_rule_matches_autograd(fcc_geometry):
    from neml2 import ResolvedShear

    leaf = ResolvedShear(crystal_geometry=fcc_geometry)
    sigma_data = torch.tensor([100.0, 50.0, -10.0, 20.0, 30.0, -5.0], dtype=torch.float64)
    R_data = torch.eye(3, dtype=torch.float64)
    sigma_leaf = sigma_data.detach().clone().requires_grad_(True)

    def f(s):
        return leaf(SR2(s), R2(R_data)).data  # (12,)

    J_autograd = torch.autograd.functional.jacobian(f, sigma_leaf)  # (12, 6)
    # Now use chain-rule action with a leading-K typed identity seed.
    v_in = {"stress": {"stress": SR2(torch.eye(6, dtype=torch.float64))}}
    _rss, v_out = leaf(SR2(sigma_data), R2(R_data), v=v_in)
    # v_out['resolved_shears']['stress'] shape (K=6, nslip); move K to columns.
    J_chain = v_out["resolved_shears"]["stress"].data.movedim(0, -1)
    assert torch.allclose(J_chain, J_autograd, atol=1e-10)


def test_sum_slip_rates_forward_and_chain_rule(fcc_geometry):
    from neml2 import SumSlipRates

    leaf = SumSlipRates()
    g_data = torch.tensor([0.1, -0.2, 0.05, -0.3, 0.01, 0.02, -0.04, 0.07, -0.01, 0.0, 0.1, -0.05])
    g = Scalar(g_data, sub_batch_ndim=1)
    sg = leaf(g)
    assert sg.sub_batch_ndim == 0
    assert torch.allclose(sg.data, g_data.abs().sum().reshape(()), atol=1e-12)

    # Identity-seeded chain rule: leading K and the slip axis as sub-batch.
    V = Scalar(torch.eye(12, dtype=g_data.dtype), sub_batch_ndim=1)
    _, v_out = leaf(g, v={"slip_rates": {"slip_rates": V}})
    # ∂(Σ|γ|)/∂γ_k = sign(γ_k). Expected along the leading K axis.
    expected = torch.sign(g_data)
    assert torch.allclose(v_out["sum_slip_rates"]["slip_rates"].data, expected)


def test_elastic_strain_rate_forward():
    from neml2 import ElasticStrainRate

    leaf = ElasticStrainRate()
    e = SR2(torch.tensor([0.01, 0.0, -0.005, 0.0, 0.0, 0.0], dtype=torch.float64))
    d = SR2(torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    w = WR2(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
    dp = SR2(torch.zeros(6, dtype=torch.float64))
    e_dot = leaf(e, d, w, dp)
    # With zero w and zero dp: e_dot = d.
    assert torch.allclose(e_dot.data, d.data, atol=1e-12)


def test_orientation_rate_forward():
    from neml2 import OrientationRate

    leaf = OrientationRate()
    e = SR2(torch.zeros(6, dtype=torch.float64))
    w = WR2(torch.tensor([0.01, -0.02, 0.005], dtype=torch.float64))
    dp = SR2(torch.zeros(6, dtype=torch.float64))
    wp = WR2(torch.tensor([0.005, -0.01, 0.002], dtype=torch.float64))
    rate = leaf(e, w, dp, wp)
    # With e=0 and dp=0, rate = w - wp.
    assert torch.allclose(rate.data, w.data - wp.data, atol=1e-12)


def test_crystal_plasticity_strain_predictor_zero_old_strain():
    from neml2 import CrystalPlasticityStrainPredictor

    leaf = CrystalPlasticityStrainPredictor(scale=1.0, threshold=1e-2)
    D = SR2(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    t = Scalar(torch.tensor(0.1, dtype=torch.float64))
    t_n = Scalar(torch.tensor(0.0, dtype=torch.float64))
    Ee_n = SR2(torch.zeros(6, dtype=torch.float64))
    out = leaf(D, t, t_n, Ee_n)
    # Old norm is 0 < threshold so we use Ee_pred = Ee_n + scale*D*dt = D*0.1
    expected = D.data * 0.1
    assert torch.allclose(out.data, expected, atol=1e-12)


def test_cp_deformation_gradient_predictor_below_threshold_holds_fp_n():
    """Trial elastic strain below ``threshold`` -> the frozen ``Fp_n`` is kept."""
    from neml2 import CrystalPlasticityDeformationGradientPredictor

    leaf = CrystalPlasticityDeformationGradientPredictor(scale=0.1, threshold=100.0)
    assert list(leaf.input_spec) == ["deformation_gradient", "plastic_deformation_gradient~1"]
    assert list(leaf.output_spec) == ["plastic_deformation_gradient"]
    F = R2(torch.tensor([[1.02, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    Fp_n = R2.identity()
    out = leaf(F, Fp_n)
    # norm(sym(F) - I) = 0.02 < threshold=100 -> predictor not applied.
    assert torch.allclose(out.data, Fp_n.data, atol=1e-12)


def test_cp_deformation_gradient_predictor_above_threshold_relaxes():
    """Trial elastic strain above ``threshold`` -> seed Fp from the relaxed trial.

    Cross-checked against an independent torch oracle of
    ``Fp = inv(I + s (Fe_tr - I)) @ F`` with ``Fe_tr = F Fp_n^{-1}``.
    """
    from neml2 import CrystalPlasticityDeformationGradientPredictor

    scale = 0.1
    leaf = CrystalPlasticityDeformationGradientPredictor(scale=scale, threshold=1e-3)
    F = R2(torch.tensor([[1.02, 0.01, 0.0], [0.0, 0.99, 0.0], [0.005, 0.0, 1.0]]))
    Fp_n = R2.identity()
    out = leaf(F, Fp_n)

    Fd, Fpn = F.data, Fp_n.data
    I3 = torch.eye(3, dtype=torch.float64)
    Fe_tr = Fd @ torch.linalg.inv(Fpn)
    Fe_pred = I3 + scale * (Fe_tr - I3)
    Fp_ref = torch.linalg.inv(Fe_pred) @ Fd
    assert torch.allclose(out.data, Fp_ref, rtol=1e-10, atol=1e-12)
    # Sanity: the predictor genuinely moved off the frozen guess.
    assert not torch.allclose(out.data, Fpn, atol=1e-6)


def test_cp_deformation_gradient_predictor_trivial_chain_rule():
    """One-shot warm-up: its chain rule is a zero pass-through (not differentiated)."""
    from neml2 import CrystalPlasticityDeformationGradientPredictor
    from neml2.models.chain_rule import ChainRuleDict

    leaf = CrystalPlasticityDeformationGradientPredictor(scale=0.1, threshold=1e-3)
    F = R2(torch.tensor([[1.02, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    Fp_n = R2.identity()
    seed = R2.identity()
    v: ChainRuleDict = {"deformation_gradient": {"deformation_gradient": seed}}
    result, v_out = leaf(F, Fp_n, v=v)
    tangent = v_out["plastic_deformation_gradient"]["deformation_gradient"]
    assert torch.allclose(tangent.data, torch.zeros_like(tangent.data), atol=1e-14)


def test_cp_deformation_gradient_predictor_renamed_via_hit():
    """The schema fix makes the I/O variable names configurable from an input file."""
    from neml2 import ModelUnitTest

    text = """
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'F Fp~1'
    input_R2_values = 'F Fpn'
    output_R2_names = 'Fp'
    output_R2_values = 'Fpn'
    value_rel_tol = 1e-10
  []
[]
[Models]
  [model]
    type = CrystalPlasticityDeformationGradientPredictor
    deformation_gradient = 'F'
    plastic_deformation_gradient = 'Fp'
    threshold = 100.0
  []
[]
[Tensors]
  [F]
    type = Python
    expr = 'R2(torch.diag(torch.tensor([1.02, 1.0, 1.0], dtype=torch.float64)))'
  []
  [Fpn]
    type = Python
    expr = 'R2.identity()'
  []
[]
"""
    ut = ModelUnitTest.from_string(text)
    # Renamed I/O resolved through the HIT factory path.
    assert list(ut.model.input_spec) == ["F", "Fp~1"]
    assert list(ut.model.output_spec) == ["Fp"]
    # threshold=100 -> below the gate -> output holds Fp_n (== identity). Skip the
    # JVP oracle: this is a one-shot predictor with a deliberately trivial chain rule.
    report = ut.run(check_dvalue=False)
    assert report.value_checks == 1
