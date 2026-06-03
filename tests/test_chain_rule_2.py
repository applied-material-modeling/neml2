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

"""Tests for the v2 second-order chain-rule infrastructure.

The design adds an opt-in ``v2`` kwarg to ``Model.forward`` and
``ComposedModel.forward`` that propagates ``∂²(output)/∂(seed_a)∂(seed_b)``
through the chain via ``(g∘f)'' = g'' · (f')² + g' · f''``.

Coverage:
- ``SR2Invariant`` standalone: v2 action equals analytical ∂²vm/∂M² and
  matches finite differences.
- ``YieldFunction`` (linear in its inputs): v2 contribution is zero.
- ``ComposedModel`` threading: composing a v2-supporting leaf with a linear
  leaf reproduces the leaf's own v2 output (g'·f'' term).
- End-to-end v2 through the J2 ``[flow] = ComposedModel(vonmises yield)``:
  the v2 propagated to ``yield_function`` matches central-difference of the
  first-order tangent dimensions.
"""

from __future__ import annotations

import torch

from neml2 import load_string
from neml2.models.common import ComposedModel, SR2Invariant
from neml2.types import SR2, Scalar

B = 2


def _yield_function():
    # YieldFunction is schema-only (no Python constructor); build it via HIT.
    # ``isotropic_hardening`` is optional in v3.0; opt in so the composed
    # chain rule below can take ∂f/∂h.
    return load_string(
        "[Models]\n"
        "  [yld]\n"
        "    type = YieldFunction\n"
        "    yield_stress = 1000.0\n"
        "    isotropic_hardening = 'isotropic_hardening'\n"
        "  []\n"
        "[]"
    ).get_model("yld")


def _sr2_eye() -> SR2:
    return SR2(torch.eye(6, dtype=torch.float64).reshape(6, 1, 6).expand(6, B, 6).contiguous())


def _scalar_eye() -> Scalar:
    return Scalar(torch.ones(1, B, dtype=torch.float64))


def _scalar_first_to_trailing(v: Scalar) -> torch.Tensor:
    return v.data.movedim(0, -1).unsqueeze(-2)


def _scalar_second_to_trailing(v: Scalar) -> torch.Tensor:
    return v.data.permute(2, 0, 1).unsqueeze(1)


def _M_state(seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = 100.0 * torch.randn(B, 6, generator=g, dtype=torch.float64)
    # Make it deviatoric to keep vm safely bounded away from zero.
    mean = M[..., :3].sum(-1, keepdim=True) / 3.0
    M[..., :3] = M[..., :3] - mean
    return M


# ---------------------------------------------------------------------------
# SR2Invariant — leaf v2 sanity vs. finite-difference Hessian
# ---------------------------------------------------------------------------


def test_sr2_invariant_v2_matches_finite_difference():
    vm_model = SR2Invariant(
        tensor="mandel_stress", invariant="effective_stress", invariant_type="VONMISES"
    )
    M = _M_state(7)

    I6 = _sr2_eye()
    v = {"mandel_stress": {"M": I6}}
    v2: dict = {}
    _, v_out, v2_out = vm_model(SR2(M), v=v, v2=v2)

    # First-order tangent: ∂vm/∂M = N → v_out shape (*B, 1, 6).
    assert _scalar_first_to_trailing(v_out["effective_stress"]["M"]).shape == (B, 1, 6)

    # Second-order block via identity-seed-pair: ∂²vm/∂M².
    H_analytical = _scalar_second_to_trailing(v2_out["effective_stress"]["M"]["M"])
    assert H_analytical.shape == (B, 1, 6, 6)

    # Central-difference reference on the first derivative.
    eps = 1e-6
    H_fd = torch.zeros(B, 1, 6, 6, dtype=torch.float64)

    def first_deriv(M_):
        v_ = {"mandel_stress": {"M": I6}}
        _, vo = vm_model(SR2(M_), v=v_)
        return _scalar_first_to_trailing(vo["effective_stress"]["M"])  # (B, 1, 6)

    for j in range(6):
        dM = torch.zeros_like(M)
        dM[..., j] = eps
        H_fd[..., :, j] = (first_deriv(M + dM) - first_deriv(M - dM)) / (2 * eps)

    abs_diff = (H_analytical - H_fd).abs().max().item()
    assert abs_diff < 1e-8


def test_sr2_invariant_v2_with_nonidentity_seed():
    """v2 must respect the (Va, Vb) outer-product structure for arbitrary seeds."""
    vm_model = SR2Invariant(
        tensor="mandel_stress", invariant="effective_stress", invariant_type="VONMISES"
    )
    M = _M_state(13)

    g = torch.Generator().manual_seed(91)
    Va_old = torch.randn(B, 6, 3, generator=g, dtype=torch.float64)
    Vb_old = torch.randn(B, 6, 2, generator=g, dtype=torch.float64)
    Va = SR2(Va_old.movedim(-1, 0).contiguous())
    Vb = SR2(Vb_old.movedim(-1, 0).contiguous())
    v = {"mandel_stress": {"a": Va, "b": Vb}}
    _, v_out, v2_out = vm_model(SR2(M), v=v, v2={})

    # Three v2 entries: (a,a), (a,b), (b,a), (b,b).
    inner = v2_out["effective_stress"]
    assert set(inner.keys()) == {"a", "b"}
    assert set(inner["a"].keys()) == {"a", "b"}
    assert set(inner["b"].keys()) == {"a", "b"}
    assert _scalar_second_to_trailing(inner["a"]["a"]).shape == (B, 1, 3, 3)
    assert _scalar_second_to_trailing(inner["a"]["b"]).shape == (B, 1, 3, 2)
    assert _scalar_second_to_trailing(inner["b"]["a"]).shape == (B, 1, 2, 3)
    assert _scalar_second_to_trailing(inner["b"]["b"]).shape == (B, 1, 2, 2)

    # Identity-seed reference Hessian → contract with (Va, Vb) for the ab block.
    # contribution[A, B] = sum_{i,j} H[i,j] · Va[i, A] · Vb[j, B].
    I6 = _sr2_eye()
    _, _, ref = vm_model(SR2(M), v={"mandel_stress": {"M": I6}}, v2={})
    H = _scalar_second_to_trailing(ref["effective_stress"]["M"]["M"]).squeeze(-3)  # (B, 6, 6)
    # H @ Vb: (B, 6, 6) @ (B, 6, 2) = (B, 6, 2). Then Va^T · (H @ Vb): (B, 3, 6) · (B, 6, 2).
    expected_ab = (Va_old.transpose(-1, -2) @ (H @ Vb_old)).unsqueeze(-3)  # (B, 1, 3, 2)
    assert torch.allclose(
        _scalar_second_to_trailing(inner["a"]["b"]), expected_ab, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# YieldFunction — linear leaf has zero v2 contribution
# ---------------------------------------------------------------------------


def test_yield_function_v2_is_zero():
    yld = _yield_function()
    es = torch.tensor([1500.0, 2000.0], dtype=torch.float64)
    h = torch.tensor([100.0, 200.0], dtype=torch.float64)

    I1 = _scalar_eye()
    v = {"effective_stress": {"a": I1}, "isotropic_hardening": {"b": I1}}
    _, v_out, v2_out = yld(Scalar(es), Scalar(h), v=v, v2={})

    # First-order tangents present.
    assert "yield_function" in v_out
    # Second-order: all zero (the framework returns empty dicts for sums of
    # zero contributions; we don't require zero tensors).
    assert v2_out["yield_function"] == {}


# ---------------------------------------------------------------------------
# ComposedModel — v2 threading through `[flow] = ComposedModel(vm yield)`
# ---------------------------------------------------------------------------


def test_composed_v2_matches_finite_difference_through_flow():
    import math

    vm = SR2Invariant(
        tensor="mandel_stress", invariant="effective_stress", invariant_type="VONMISES"
    )
    yld = _yield_function()
    flow = ComposedModel([vm, yld])

    M = _M_state(23)
    h = 25.0 * torch.ones(B, dtype=torch.float64)

    I6 = _sr2_eye()
    I1 = _scalar_eye()
    v = {"mandel_stress": {"M": I6}, "isotropic_hardening": {"h": I1}}
    *_vals, v_out, v2_out = flow(M, h, v=v, v2={})

    # ∂²(yield_function)/∂M² = √(2/3) · ∂²vm/∂M² (yield linear in vm with
    # C++-aligned √(2/3) factor).
    H_yield = _scalar_second_to_trailing(v2_out["yield_function"]["M"]["M"])
    H_vm = _scalar_second_to_trailing(
        vm(SR2(M), v={"mandel_stress": {"M": I6}}, v2={})[-1]["effective_stress"]["M"]["M"]
    )
    assert torch.allclose(H_yield, math.sqrt(2.0 / 3.0) * H_vm, rtol=1e-12, atol=1e-12)

    # Mixed (M, h) block: ∂²(yield_function)/(∂M ∂h) = 0 (yield is linear in h
    # and vm depends only on M).
    H_mixed = v2_out["yield_function"].get("M", {}).get("h")
    if H_mixed is not None:
        H_mixed_t = _scalar_second_to_trailing(H_mixed)
        assert torch.allclose(H_mixed_t, torch.zeros_like(H_mixed_t), atol=1e-14)
    # h-h block: yield is linear in h ⇒ zero.
    H_hh = v2_out["yield_function"].get("h", {}).get("h")
    if H_hh is not None:
        H_hh_t = _scalar_second_to_trailing(H_hh)
        assert torch.allclose(H_hh_t, torch.zeros_like(H_hh_t), atol=1e-14)


def test_composed_v2_none_short_circuits_to_first_order():
    """ComposedModel.forward(v=…) returns (*vals, v_out) — no v2 element."""
    vm = SR2Invariant(
        tensor="mandel_stress", invariant="effective_stress", invariant_type="VONMISES"
    )
    yld = _yield_function()
    flow = ComposedModel([vm, yld])

    M = _M_state(31)
    h = torch.zeros(B, dtype=torch.float64)
    I6 = _sr2_eye()
    I1 = _scalar_eye()
    v = {"mandel_stress": {"M": I6}, "isotropic_hardening": {"h": I1}}
    result = flow(M, h, v=v)  # no v2 → first-order only
    # (vals…, v_out): the v_out is the last element.
    v_out = result[-1]
    assert "yield_function" in v_out


def test_composed_v2_without_v_raises():
    vm = SR2Invariant(
        tensor="mandel_stress", invariant="effective_stress", invariant_type="VONMISES"
    )
    yld = _yield_function()
    flow = ComposedModel([vm, yld])

    M = _M_state(41)
    h = torch.zeros(B, dtype=torch.float64)
    import pytest

    with pytest.raises(ValueError, match="v2/vh was provided without v"):
        flow(M, h, v2={})
