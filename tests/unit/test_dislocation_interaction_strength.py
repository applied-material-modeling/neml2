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

"""Unit tests for :class:`DislocationInteractionStrengthMap`.

The interaction map generalizes :class:`DislocationObstacleStrengthMap` with a
slip-system latent-hardening matrix: ``tau_i = tau0 + alpha*mu*b*sqrt(sum_r h_ir
rho_r)``. These tests pin the two defining properties — exact reduction to the
diagonal model at ``h = I``, and the closed-form value for a general ``n_slip``
and a dense ``h`` — rather than merely that it runs.
"""

from __future__ import annotations

import pytest
import torch

from neml2 import load_string
from neml2.types import Scalar

# shared coefficients
_TAU0, _ALPHA, _MU, _B = 5.0, 0.5, 80.0, 0.25


def _interaction_model(nslip: int, h: torch.Tensor):
    """Build the interaction map with a baked ``n_slip x n_slip`` matrix ``h``."""
    rows = ", ".join(repr(h[i].tolist()) for i in range(nslip))
    txt = f"""
    [Tensors]
      [hmat]
        type = Python
        expr = 'Scalar(torch.tensor([{rows}], dtype=torch.float64)).sub_batch.retag(2)'
      []
    []
    [Models]
      [model]
        type = DislocationInteractionStrengthMap
        dislocation_density = 'dislocation_density'
        interaction_matrix = 'hmat'
        constant_strength = {_TAU0}
        alpha = {_ALPHA}
        mu = {_MU}
        b = {_B}
      []
    []
    """
    return load_string(txt).get_model("model")


def _obstacle_model():
    txt = f"""
    [Models]
      [model]
        type = DislocationObstacleStrengthMap
        constant_strength = {_TAU0}
        alpha = {_ALPHA}
        mu = {_MU}
        b = {_B}
      []
    []
    """
    return load_string(txt).get_model("model")


@pytest.mark.parametrize("nslip", [3, 5, 12])
def test_value_matches_closed_form(nslip):
    """tau_i = tau0 + alpha*mu*b*sqrt(sum_r h_ir rho_r) for a dense h, any n_slip."""
    torch.manual_seed(nslip)
    rho = (torch.rand(nslip, dtype=torch.float64) + 0.5) * 10.0
    eye = torch.eye(nslip, dtype=torch.float64)
    h = eye + 0.3 * (1.0 - eye)  # dense, symmetric
    model = _interaction_model(nslip, h)
    tau = model.call_by_name({"dislocation_density": Scalar(rho, 1).with_sub_batch_ndim(1)})[
        "slip_strengths"
    ]
    expected = _TAU0 + _ALPHA * _MU * _B * torch.sqrt(h @ rho)
    torch.testing.assert_close(tau.data.reshape(nslip), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("nslip", [3, 12])
def test_h_identity_reduces_to_obstacle_map(nslip):
    """With h = I the interaction map must equal DislocationObstacleStrengthMap exactly."""
    torch.manual_seed(nslip + 100)
    rho = (torch.rand(nslip, dtype=torch.float64) + 0.5) * 10.0
    rho_s = Scalar(rho, 1).with_sub_batch_ndim(1)
    interaction = _interaction_model(nslip, torch.eye(nslip, dtype=torch.float64))
    obstacle = _obstacle_model()
    tau_i = interaction.call_by_name({"dislocation_density": rho_s})["slip_strengths"]
    tau_o = obstacle.call_by_name({"dislocation_density": rho_s})["slip_strengths"]
    torch.testing.assert_close(tau_i.data, tau_o.data, rtol=0, atol=0)


def test_dense_pushforward_matches_autograd():
    """The analytic d(tau)/d(rho) (dense over the source slip axis) matches autograd."""
    nslip = 6
    torch.manual_seed(7)
    rho0 = (torch.rand(nslip, dtype=torch.float64) + 0.5) * 10.0
    eye = torch.eye(nslip, dtype=torch.float64)
    h = eye + 0.4 * (1.0 - eye)
    model = _interaction_model(nslip, h)

    def tau_of(r):
        return model.call_by_name({"dislocation_density": Scalar(r, 1).with_sub_batch_ndim(1)})[
            "slip_strengths"
        ].data.reshape(nslip)

    jac_autograd = torch.autograd.functional.jacobian(tau_of, rho0)
    forest = h @ rho0
    d_coeff = _ALPHA * _MU * _B * 0.5 / torch.sqrt(forest)
    jac_formula = d_coeff.unsqueeze(-1) * h  # d tau_i / d rho_r = d_coeff_i * h_ir
    torch.testing.assert_close(jac_autograd, jac_formula, rtol=0, atol=1e-10)


def _model_with_raw_matrix(expr: str):
    """Build the interaction map with an arbitrary interaction_matrix tensor expr."""
    txt = f"""
    [Tensors]
      [hmat]
        type = Python
        expr = '{expr}'
      []
    []
    [Models]
      [model]
        type = DislocationInteractionStrengthMap
        dislocation_density = 'dislocation_density'
        interaction_matrix = 'hmat'
        constant_strength = {_TAU0}
        alpha = {_ALPHA}
        mu = {_MU}
        b = {_B}
      []
    []
    """
    return load_string(txt).get_model("model")


def test_non_square_matrix_raises():
    """A non-square interaction_matrix gives a clear 'must be square' error."""
    model = _model_with_raw_matrix(
        "Scalar(torch.ones(3, 4, dtype=torch.float64)).sub_batch.retag(2)"
    )
    rho = Scalar(torch.ones(4, dtype=torch.float64), 1).with_sub_batch_ndim(1)
    with pytest.raises(ValueError, match="must be square"):
        model.call_by_name({"dislocation_density": rho})


def test_matrix_slip_count_mismatch_raises():
    """A square matrix whose size != dislocation_density slip count errors clearly."""
    model = _model_with_raw_matrix("Scalar(torch.eye(3, dtype=torch.float64)).sub_batch.retag(2)")
    rho = Scalar(torch.ones(4, dtype=torch.float64), 1).with_sub_batch_ndim(1)
    with pytest.raises(ValueError, match="does not match"):
        model.call_by_name({"dislocation_density": rho})
