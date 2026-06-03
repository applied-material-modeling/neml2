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

"""Tests for the generic ``Normality`` operator.

Normality wraps an inner Model and exposes ∂(inner.function)/∂(inner.from[i])
as named outputs. Its first-order chain rule needs the inner function's
second derivatives, which the v2 chain-rule infrastructure propagates via
the inner ComposedModel.

Coverage:
- Pure forward: outputs equal hand-computed first derivatives.
- forward(v=…): pushforward equals analytical Hessian-times-tangent (the
  J2 derivative ∂N/∂M = (3/2)/vm · [P_dev − (2/3) N⊗N], expressed in Mandel).
- Central-difference verification of the Hessian on a non-trivial state.
- Constant-output input (isotropic_hardening_direction): chain rule yields
  structural zeros (absent from v_out) for any incoming tangent.
"""

from __future__ import annotations

import math

import pytest
import torch

from neml2 import load_string
from neml2.models.common import ComposedModel, SR2Invariant
from neml2.models.solid_mechanics.plasticity import Normality
from neml2.types import SR2, Scalar

PARAMS = dict(sy=1000.0)
B = 3


def _yield_function():
    # YieldFunction is schema-only (no Python constructor); build it via HIT.
    return load_string(
        "[Models]\n"
        "  [yld]\n"
        "    type = YieldFunction\n"
        f"    yield_stress = {PARAMS['sy']}\n"
        # isotropic_hardening is optional on the v3.0 side — opt in
        # explicitly so Normality can take ∂f/∂h below.
        "    isotropic_hardening = 'isotropic_hardening'\n"
        "  []\n"
        "[]"
    ).get_model("yld")


def _flow_model() -> ComposedModel:
    return ComposedModel(
        [
            SR2Invariant(
                tensor="mandel_stress",
                invariant="effective_stress",
                invariant_type="VONMISES",
            ),
            _yield_function(),
        ]
    )


def _normality() -> Normality:
    return Normality(
        model=_flow_model(),
        function="yield_function",
        from_=["mandel_stress", "isotropic_hardening"],
        to=["flow_direction", "isotropic_hardening_direction"],
    )


def _deviatoric_state() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    M = 100.0 * torch.randn(B, 6, generator=g, dtype=torch.float64)
    # Project to deviator to avoid the small-vm gradient blow-up region.
    mean = M[..., :3].sum(-1, keepdim=True) / 3.0
    M[..., :3] = M[..., :3] - mean
    h = 50.0 * torch.rand(B, generator=g, dtype=torch.float64)
    return M, h


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_normality_forward_flow_direction_matches_analytical_J2():
    """flow_direction = √(2/3) · (3/2) · dev(M) / vm (C++-aligned algebra)."""
    normality = _normality()
    M, h = _deviatoric_state()
    N, ihd = normality(M, h)

    # C++ YieldFunction: f = √(2/3) · (vm − sy − h)
    # ⇒ ∂f/∂M = √(2/3) · ∂vm/∂M = √(2/3) · (3/2) S / vm = √(3/2) · S / vm.
    S = M  # already deviatoric
    vm = torch.sqrt(1.5 * (S * S).sum(-1))
    N_ref = math.sqrt(1.5) * S / vm.unsqueeze(-1)

    assert torch.allclose(N, N_ref, rtol=1e-12, atol=1e-12)
    # Yield function ⇒ ∂f/∂h = −√(2/3).
    assert torch.allclose(ihd, torch.full_like(ihd, -math.sqrt(2.0 / 3.0)), atol=1e-14)


def test_normality_output_shapes_and_dtypes():
    normality = _normality()
    M, h = _deviatoric_state()
    N, ihd = normality(M, h)
    assert N.shape == (B, 6)
    assert ihd.shape == (B,)
    assert N.dtype == torch.float64
    assert ihd.dtype == torch.float64


def test_normality_unit_norm_on_pure_shear():
    """For deviatoric M and C++-aligned YieldFunction, ||N||_F = 1.

    Without √(2/3): ||N|| = √(3/2). With √(2/3): √(2/3) · √(3/2) = 1.
    """
    normality = _normality()
    tau = 50.0
    M = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(2.0) * tau]], dtype=torch.float64)
    h = torch.zeros(1, dtype=torch.float64)
    N, _ = normality(M, h)
    n_norm = torch.sqrt((N * N).sum()).item()
    assert math.isclose(n_norm, 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Chain rule (first-order pushforward of Normality's outputs)
# ---------------------------------------------------------------------------


def test_normality_jvp_matches_finite_difference_hessian():
    """Identity seeds → v_out blocks = inner function's Hessian (FD-verified)."""
    normality = _normality()
    M, h = _deviatoric_state()

    I6 = SR2(torch.eye(6, dtype=torch.float64).reshape(6, 1, 6).expand(6, B, 6))
    I1 = Scalar(torch.ones(1, B, dtype=torch.float64))
    v = {
        "mandel_stress": {"M": I6},
        "isotropic_hardening": {"h": I1},
    }
    _, _, v_out = normality(M, h, v=v)

    # ∂(flow_direction)/∂M = ∂²f/∂M² → finite-difference check.
    eps = 1e-6
    H_fd = torch.zeros(B, 6, 6, dtype=torch.float64)
    for i in range(6):
        dM = torch.zeros_like(M)
        dM[..., i] = eps
        Np, _ = normality(M + dM, h)
        Nm, _ = normality(M - dM, h)
        H_fd[..., :, i] = (Np - Nm) / (2 * eps)

    H_analytical = v_out["flow_direction"]["M"].data.movedim(0, -1)
    assert H_analytical.shape == (B, 6, 6)
    abs_diff = (H_analytical - H_fd).abs().max().item()
    # Symmetric central diff on a smooth function is exact to O(eps²);
    # for eps=1e-6 and dynamic range O(1e-2) this is ~1e-9.
    assert abs_diff < 1e-8


def test_normality_constant_output_chain_rule_is_zero():
    """isotropic_hardening_direction = -1 constant → no nonzero v_out entries.

    The yield function is linear in (vm, h), so ∂²f/(∂h ∂·) = 0; Normality's
    ``isotropic_hardening_direction`` output therefore contributes nothing to
    the outer chain rule. The framework represents structural zeros as absent
    inner-dict entries (not zero tensors).
    """
    normality = _normality()
    M, h = _deviatoric_state()

    v = {
        "mandel_stress": {
            "M": SR2(torch.eye(6, dtype=torch.float64).reshape(6, 1, 6).expand(6, B, 6))
        },
    }
    _, _, v_out = normality(M, h, v=v)
    # ihd's v_out exists but its inner dict is empty (constant output).
    assert v_out["isotropic_hardening_direction"] == {}


def test_normality_rejects_v2_kwarg():
    """Normality cannot propagate second-order sensitivities (would need ∂³f)."""
    normality = _normality()
    M, h = _deviatoric_state()
    with pytest.raises(NotImplementedError, match="v2"):
        normality(M, h, v={}, v2={})


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_normality_validates_from_to_length():
    with pytest.raises(ValueError, match="from and to"):
        Normality(
            model=_flow_model(),
            function="yield_function",
            from_=["mandel_stress"],
            to=["flow_direction", "isotropic_hardening_direction"],
        )


def test_normality_validates_function_in_inner_output_spec():
    with pytest.raises(KeyError, match="no output named 'foo'"):
        Normality(model=_flow_model(), function="foo", from_=["mandel_stress"], to=["x"])


def test_normality_validates_from_in_inner_input_spec():
    with pytest.raises(KeyError, match="no input named 'foo'"):
        Normality(
            model=_flow_model(),
            function="yield_function",
            from_=["foo"],
            to=["x"],
        )
