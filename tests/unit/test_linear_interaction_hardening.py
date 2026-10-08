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

"""Unit tests for the linear slip-system interaction leaves.

``LinearInteractionStrengthMap``: ``tau_i = tau0_i + sum_j h_ij phi_j``.
``LinearInteractionHardeningRule``: ``slip_hardening_rate_i = sum_j M_ij |gamma_dot_j|``.

Both couple slip systems linearly (constant Jacobian), reduce to independent per-system
hardening at ``h = I``, and are dense consumers of the grain x slip fold.
"""

from __future__ import annotations

import pytest
import torch

import neml2.user_tensors  # noqa: F401 (register SquareMatrix)
from neml2 import load_string
from neml2.types import Scalar

_TAU0 = 5.0


def _dense(n: int, off: float = 0.3) -> torch.Tensor:
    eye = torch.eye(n, dtype=torch.float64)
    return eye + off * (1.0 - eye)


def _rows(m: torch.Tensor) -> str:
    return ", ".join(repr(m[i].tolist()) for i in range(m.shape[0]))


def _strength_model(h: torch.Tensor):
    txt = f"""
    [Tensors]
      [hmat]
        type = Python
        expr = 'Scalar(torch.tensor([{_rows(h)}], dtype=torch.float64)).sub_batch.retag(2)'
      []
    []
    [Models]
      [model]
        type = LinearInteractionStrengthMap
        slip_hardening = 'slip_hardening'
        interaction_matrix = 'hmat'
        constant_strength = {_TAU0}
      []
    []
    """
    return load_string(txt).get_model("model")


def _rule_model(m: torch.Tensor):
    txt = f"""
    [Tensors]
      [mmat]
        type = Python
        expr = 'Scalar(torch.tensor([{_rows(m)}], dtype=torch.float64)).sub_batch.retag(2)'
      []
    []
    [Models]
      [model]
        type = LinearInteractionHardeningRule
        slip_hardening = 'slip_hardening'
        slip_rates = 'slip_rates'
        interaction_matrix = 'mmat'
      []
    []
    """
    return load_string(txt).get_model("model")


# ---------------------------------------------------------------- strength map


@pytest.mark.parametrize("nslip", [3, 5, 12])
def test_strength_value_matches_closed_form(nslip):
    """tau_i = tau0 + sum_j h_ij phi_j for a dense h, any n_slip."""
    torch.manual_seed(nslip)
    phi = (torch.rand(nslip, dtype=torch.float64) + 0.5) * 10.0
    h = _dense(nslip)
    model = _strength_model(h)
    tau = model.call_by_name({"slip_hardening": Scalar(phi, 1).with_sub_batch_ndim(1)})[
        "slip_strengths"
    ]
    torch.testing.assert_close(tau.data.reshape(nslip), _TAU0 + h @ phi, rtol=0, atol=1e-10)


def test_strength_identity_decouples():
    """With h = I the systems decouple: tau_i = tau0 + phi_i."""
    nslip = 5
    torch.manual_seed(1)
    phi = torch.rand(nslip, dtype=torch.float64) + 0.5
    model = _strength_model(torch.eye(nslip, dtype=torch.float64))
    tau = model.call_by_name({"slip_hardening": Scalar(phi, 1).with_sub_batch_ndim(1)})[
        "slip_strengths"
    ]
    torch.testing.assert_close(tau.data.reshape(nslip), _TAU0 + phi, rtol=0, atol=1e-12)


def test_strength_pushforward_matches_autograd():
    """The analytic d(tau)/d(phi) = h (dense, constant) matches autograd."""
    nslip = 6
    torch.manual_seed(7)
    phi0 = torch.rand(nslip, dtype=torch.float64) + 0.5
    h = _dense(nslip, 0.4)
    model = _strength_model(h)

    def tau_of(p):
        return model.call_by_name({"slip_hardening": Scalar(p, 1).with_sub_batch_ndim(1)})[
            "slip_strengths"
        ].data.reshape(nslip)

    torch.testing.assert_close(
        torch.autograd.functional.jacobian(tau_of, phi0), h, rtol=0, atol=1e-10
    )


def test_strength_non_square_matrix_raises():
    model = _strength_model(torch.ones(3, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="must be square"):
        model.call_by_name(
            {"slip_hardening": Scalar(torch.ones(4, dtype=torch.float64), 1).with_sub_batch_ndim(1)}
        )


# ---------------------------------------------------------------- hardening rate rule


@pytest.mark.parametrize("nslip", [3, 5, 12])
def test_rule_value_matches_closed_form(nslip):
    """slip_hardening_rate_i = sum_j M_ij |gamma_dot_j| for a dense M, any n_slip."""
    torch.manual_seed(nslip + 50)
    gdot = (torch.rand(nslip, dtype=torch.float64) - 0.5) * 20.0  # mixed signs
    m = _dense(nslip)
    model = _rule_model(m)
    rate = model.call_by_name(
        {
            "slip_hardening": Scalar(
                torch.zeros(nslip, dtype=torch.float64), 1
            ).with_sub_batch_ndim(1),
            "slip_rates": Scalar(gdot, 1).with_sub_batch_ndim(1),
        }
    )["slip_hardening_rate"]
    torch.testing.assert_close(rate.data.reshape(nslip), m @ gdot.abs(), rtol=0, atol=1e-10)


def test_rule_pushforward_matches_autograd():
    """The analytic d(rate)/d(gamma_dot) = M_ij sign(gamma_dot_j) matches autograd."""
    nslip = 6
    torch.manual_seed(11)
    gdot0 = (torch.rand(nslip, dtype=torch.float64) - 0.5) * 20.0
    m = _dense(nslip, 0.4)
    model = _rule_model(m)
    h0 = Scalar(torch.zeros(nslip, dtype=torch.float64), 1).with_sub_batch_ndim(1)

    def rate_of(g):
        return model.call_by_name(
            {"slip_hardening": h0, "slip_rates": Scalar(g, 1).with_sub_batch_ndim(1)}
        )["slip_hardening_rate"].data.reshape(nslip)

    jac = torch.autograd.functional.jacobian(rate_of, gdot0)
    expected = m * torch.sign(gdot0).unsqueeze(0)  # d rate_i / d gdot_j = M_ij sign(gdot_j)
    torch.testing.assert_close(jac, expected, rtol=0, atol=1e-10)


def test_rule_slip_count_mismatch_raises():
    model = _rule_model(torch.eye(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="does not match"):
        model.call_by_name(
            {
                "slip_hardening": Scalar(
                    torch.zeros(4, dtype=torch.float64), 1
                ).with_sub_batch_ndim(1),
                "slip_rates": Scalar(torch.ones(4, dtype=torch.float64), 1).with_sub_batch_ndim(1),
            }
        )
