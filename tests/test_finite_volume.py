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

"""Coverage for the four Python-native finite-volume leaves.

For each leaf:

* Forward path matches the canonical algebraic formula.
* Chain rule against an identity seed reproduces the full dense Jacobian
  (``torch.autograd.functional.jacobian``).
* ``list_deriv`` declares the right dense / diagonal flags.

Plus an end-to-end composition test stitching the four together (the
KWN advection chain shape) — ComposedModel's resolver should produce the
correct dense-end-to-end metadata and the cumulative Jacobian still matches
autograd.
"""

from __future__ import annotations

import torch

from neml2.models.common import ComposedModel
from neml2.models.finite_volume import (
    FiniteVolumeAppendBoundaryCondition,
    FiniteVolumeGradient,
    FiniteVolumeUpwindedAdvectiveFlux,
    LinearlyInterpolateToCellEdges,
)
from neml2.types import Scalar


def _seed(t: torch.Tensor) -> Scalar:
    """Leading-K Scalar identity seed shaped ``(numel(t), *t.shape)``."""
    N = t.numel()
    eye = torch.eye(N, dtype=t.dtype, device=t.device).reshape(N, *t.shape)
    return Scalar(eye, sub_batch_ndim=1)


def _autograd_jac(fn, x: torch.Tensor) -> torch.Tensor:
    """Dense Jacobian ``df/dx`` from ``torch.autograd.functional.jacobian``."""
    return torch.autograd.functional.jacobian(fn, x, create_graph=False)


# ---------------------------------------------------------------------------
# FiniteVolumeAppendBoundaryCondition
# ---------------------------------------------------------------------------


def test_append_bc_left_forward():
    m = FiniteVolumeAppendBoundaryCondition(side="left", bc_value=0.0, input="u", output="ext")
    u = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    out = m(Scalar(u))
    assert torch.equal(out.data, torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64))


def test_append_bc_right_forward():
    m = FiniteVolumeAppendBoundaryCondition(side="right", bc_value=7.0, input="u", output="ext")
    u = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    out = m(Scalar(u))
    assert torch.equal(out.data, torch.tensor([[1.0, 2.0, 3.0, 7.0]], dtype=torch.float64))


def test_append_bc_chain_rule_matches_autograd():
    """Identity-seeded chain rule reproduces the (N+1, N) selector."""
    for side, bc in (("left", 0.0), ("right", 2.5)):
        m = FiniteVolumeAppendBoundaryCondition(side=side, bc_value=bc, input="u", output="ext")
        u = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        v = {"u": {"s": _seed(u)}}
        _, v_out = m(Scalar(u, sub_batch_ndim=1), v=v)
        # Reference: jacobian of m.forward(u).data over u.
        jac = _autograd_jac(lambda t, m=m: m(Scalar(t)).data, u)
        # v_out["ext"]["s"] has shape (*B, N+1, 1, numel(u)). jac has shape
        # (*B_out, N+1, *B_in, N). Pull out the diagonal-batch slab and
        # reshape to compare.
        for L_out in range(jac.shape[1]):
            for L_in in range(jac.shape[3]):
                k = L_in  # one batch entry, so seed index k matches L_in
                expected = jac[0, L_out, 0, L_in]
                assert torch.isclose(v_out["ext"]["s"].data[k, 0, L_out], expected), (
                    f"side={side}, L_out={L_out}, L_in={L_in}: "
                    f"chain rule {v_out['ext']['s'].data[k, 0, L_out]} "
                    f"vs autograd {expected}"
                )


def test_append_bc_list_deriv_is_dense():
    m = FiniteVolumeAppendBoundaryCondition(side="left", bc_value=0.0, input="u", output="ext")
    assert m.list_deriv == {("ext", "u"): "dense"}


# ---------------------------------------------------------------------------
# FiniteVolumeGradient
# ---------------------------------------------------------------------------


def test_gradient_forward_default_prefactor():
    dx = Scalar(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))
    m = FiniteVolumeGradient(dx=dx)
    u = torch.tensor([[1.0, 3.0, 6.0, 10.0]], dtype=torch.float64)
    # Expected: -prefactor*(u[1:]-u[:-1])/dx = -[2, 3, 4]/0.5 = [-4, -6, -8]
    out = m(Scalar(u))
    assert torch.allclose(out.data, torch.tensor([[-4.0, -6.0, -8.0]], dtype=torch.float64))


def test_gradient_forward_with_prefactor():
    dx = Scalar(torch.tensor([0.5, 0.5], dtype=torch.float64))
    pref = Scalar(torch.tensor([2.0, 3.0], dtype=torch.float64))
    m = FiniteVolumeGradient(dx=dx, prefactor=pref)
    u = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    # -2*(2-1)/0.5 = -4, -3*(4-2)/0.5 = -12
    out = m(Scalar(u))
    assert torch.allclose(out.data, torch.tensor([[-4.0, -12.0]], dtype=torch.float64))


def test_gradient_chain_rule_matches_autograd():
    dx = Scalar(torch.tensor([0.5, 0.7, 0.3], dtype=torch.float64))
    m = FiniteVolumeGradient(dx=dx)
    u = torch.tensor([[1.0, 3.0, 6.0, 10.0]], dtype=torch.float64)
    v = {"u": {"s": _seed(u)}}
    _, v_out = m(Scalar(u, sub_batch_ndim=1), v=v)
    jac = _autograd_jac(lambda t: m(Scalar(t)).data, u)
    for L_out in range(jac.shape[1]):
        for L_in in range(jac.shape[3]):
            assert torch.isclose(
                v_out["grad_u"]["s"].data[L_in, 0, L_out], jac[0, L_out, 0, L_in]
            ), f"L_out={L_out}, L_in={L_in}"


def test_gradient_list_deriv_is_dense():
    dx = Scalar(torch.tensor([0.5], dtype=torch.float64))
    m = FiniteVolumeGradient(dx=dx)
    assert m.list_deriv == {("grad_u", "u"): "dense"}


# ---------------------------------------------------------------------------
# LinearlyInterpolateToCellEdges
# ---------------------------------------------------------------------------


def test_interpolate_forward_uniform_grid():
    """Uniform grid: midpoint interpolation is the average of adjacent cells."""
    N = 4
    centers = Scalar(torch.linspace(0.5, N - 0.5, N, dtype=torch.float64))
    edges = Scalar(torch.linspace(0.0, float(N), N + 1, dtype=torch.float64))
    # Use a literal cell_values as a HIT-style string so it becomes a static buffer.
    vals = Scalar(torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64))
    m = LinearlyInterpolateToCellEdges(cell_values=vals, cell_centers=centers, cell_edges=edges)
    out = m()
    # Interior edges at x=1, 2, 3: averages of (1,2), (2,4), (4,8).
    assert torch.allclose(out.data, torch.tensor([1.5, 3.0, 6.0], dtype=torch.float64))


def test_interpolate_chain_rule_matches_autograd_for_nl_cell_values():
    """When cell_values is wired as an nl input (string spec promoted to an
    input variable via mode 4), chain rule reproduces the bidiagonal weight
    matrix exactly."""
    N = 5
    centers = Scalar(torch.linspace(0.5, N - 0.5, N, dtype=torch.float64))
    edges = Scalar(torch.linspace(0.0, float(N), N + 1, dtype=torch.float64))
    m = LinearlyInterpolateToCellEdges(
        cell_values="qbin",  # bare-variable promotion → input named 'qbin'
        cell_centers=centers,
        cell_edges=edges,
    )
    assert "qbin" in m.input_spec  # promoted to nl input
    assert m.list_deriv == {("edge_values", "qbin"): "dense"}

    q = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float64)
    seed = _seed(q)
    v = {"qbin": {"s": seed}}
    _, v_out = m(Scalar(q, sub_batch_ndim=1), v=v)

    # Reference: dense jacobian of m(Scalar(q)) over q. Need a closure that
    # rebuilds the forward purely as a function of q.
    def fwd_of_q(t: torch.Tensor) -> torch.Tensor:
        return m(Scalar(t)).data

    jac = _autograd_jac(fwd_of_q, q)
    for L_out in range(jac.shape[0]):
        for L_in in range(jac.shape[1]):
            assert torch.isclose(v_out["edge_values"]["s"].data[L_in, L_out], jac[L_out, L_in]), (
                f"L_out={L_out}, L_in={L_in}"
            )


def test_interpolate_with_static_cell_values_has_no_chain_rule_action():
    """When cell_values is a static buffer, the chain rule emits an empty
    v_out (no inputs to propagate against)."""
    N = 4
    centers = Scalar(torch.linspace(0.5, N - 0.5, N, dtype=torch.float64))
    edges = Scalar(torch.linspace(0.0, float(N), N + 1, dtype=torch.float64))
    vals = Scalar(torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64))
    m = LinearlyInterpolateToCellEdges(cell_values=vals, cell_centers=centers, cell_edges=edges)
    # No inputs to seed; v_out has the output key but the inner dict is empty.
    out, v_out = m(v={})
    assert v_out == {"edge_values": {}}
    assert torch.allclose(out.data, torch.tensor([1.5, 3.0, 6.0], dtype=torch.float64))


# ---------------------------------------------------------------------------
# FiniteVolumeUpwindedAdvectiveFlux
# ---------------------------------------------------------------------------


def test_advective_flux_forward_positive_velocity_upwinds_left():
    m = FiniteVolumeUpwindedAdvectiveFlux()
    u = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    v = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    out = m(Scalar(u), Scalar(v))
    # All v positive ⇒ flux[i] = v[i]*u[i] = [0.5, 1.0, 2.0]
    assert torch.allclose(out.data, torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))


def test_advective_flux_forward_negative_velocity_upwinds_right():
    m = FiniteVolumeUpwindedAdvectiveFlux()
    u = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    v = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
    out = m(Scalar(u), Scalar(v))
    # All v negative ⇒ flux[i] = v[i]*u[i+1] = [-2, -4, -8]
    assert torch.allclose(out.data, torch.tensor([-2.0, -4.0, -8.0], dtype=torch.float64))


def test_advective_flux_chain_rule_u_dense_matches_autograd():
    m = FiniteVolumeUpwindedAdvectiveFlux()
    u = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    ve = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float64)  # mixed sign
    seed_u = _seed(u)
    v = {"u": {"s": seed_u}}
    _, v_out = m(Scalar(u, sub_batch_ndim=1), Scalar(ve, sub_batch_ndim=1), v=v)

    def fwd_of_u(t: torch.Tensor) -> torch.Tensor:
        return m(Scalar(t), Scalar(ve)).data

    jac = _autograd_jac(fwd_of_u, u)
    for L_out in range(jac.shape[0]):
        for L_in in range(jac.shape[1]):
            assert torch.isclose(v_out["flux"]["s"].data[L_in, L_out], jac[L_out, L_in])


def test_advective_flux_chain_rule_v_edge_diagonal_matches_autograd():
    m = FiniteVolumeUpwindedAdvectiveFlux()
    u = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    ve = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float64)
    seed_v = _seed(ve)
    v = {"v_edge": {"s": seed_v}}
    _, v_out = m(Scalar(u, sub_batch_ndim=1), Scalar(ve, sub_batch_ndim=1), v=v)

    def fwd_of_v(t: torch.Tensor) -> torch.Tensor:
        return m(Scalar(u), Scalar(t)).data

    jac = _autograd_jac(fwd_of_v, ve)
    for L_out in range(jac.shape[0]):
        for L_in in range(jac.shape[1]):
            assert torch.isclose(v_out["flux"]["s"].data[L_in, L_out], jac[L_out, L_in])


def test_advective_flux_list_deriv():
    m = FiniteVolumeUpwindedAdvectiveFlux()
    # u is dense, v_edge defaults to diagonal (absent from list_deriv).
    assert m.list_deriv == {("flux", "u"): "dense"}


# ---------------------------------------------------------------------------
# End-to-end composition — KWN-style advection chain
# ---------------------------------------------------------------------------


def test_kwn_advection_chain_metadata_and_jacobian():
    """Compose flux → left-BC → right-BC → gradient. Expect:

    * ComposedModel.list_deriv flags ``(grad_u, u_in)`` and
      ``(grad_u, v_edge)`` as dense (every edge along the path is dense
      for u; v_edge enters via the dense flux step).
    * Cumulative Jacobian over the master inputs matches autograd.
    """
    N = 5  # cells
    # Step 1: flux at interior edges (size M = N-1 = 4).
    flux = FiniteVolumeUpwindedAdvectiveFlux(u="u_in", v_edge="ve_in", flux="J")
    # Step 2: left BC append → size M+1 = N.
    lbc = FiniteVolumeAppendBoundaryCondition(side="left", bc_value=0.0, input="J", output="J_l")
    # Step 3: right BC append → size N+1.
    rbc = FiniteVolumeAppendBoundaryCondition(
        side="right", bc_value=0.0, input="J_l", output="J_lr"
    )
    # Step 4: gradient on J_lr (size N+1) → grad (size N).
    dx = Scalar(torch.full((N,), 0.5, dtype=torch.float64))
    grad = FiniteVolumeGradient(dx=dx, u="J_lr", grad_u="grad")

    c = ComposedModel([flux, lbc, rbc, grad])
    assert c.list_deriv == {
        ("grad", "u_in"): "dense",
        ("grad", "ve_in"): "dense",
    }
    assert c.input_spec == {"u_in": Scalar, "ve_in": Scalar}
    assert c.output_spec == {"grad": Scalar}

    u = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float64)
    ve = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # Forward only — sanity.
    (out,) = c(u, ve)
    assert out.data.shape == torch.Size([5])

    # Chain rule: seed both master inputs.
    v = {"u_in": {"u_seed": _seed(u)}, "ve_in": {"v_seed": _seed(ve)}}
    _, v_out = c(Scalar(u, sub_batch_ndim=1), Scalar(ve, sub_batch_ndim=1), v=v)

    def fwd_pair(u_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        # torch.autograd.functional.jacobian needs Tensor outputs; unwrap.
        (out,) = c(u_t, v_t)
        return out.data

    jac_u, jac_v = torch.autograd.functional.jacobian(fwd_pair, (u, ve))
    # Cross-check both seeds against the dense Jacobian.
    for L_out in range(jac_u.shape[0]):
        for L_in in range(jac_u.shape[1]):
            assert torch.isclose(v_out["grad"]["u_seed"].data[L_in, L_out], jac_u[L_out, L_in]), (
                f"u: L_out={L_out}, L_in={L_in}"
            )
        for L_in in range(jac_v.shape[1]):
            assert torch.isclose(v_out["grad"]["v_seed"].data[L_in, L_out], jac_v[L_out, L_in]), (
                f"v: L_out={L_out}, L_in={L_in}"
            )
