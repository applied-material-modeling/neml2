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

"""Tests for Model: variable declarations, consumed/provided items,
call_by_name routing, and forward(v=...) Jacobian correctness via FD."""

import pytest
import torch

from neml2.model import Model
from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity
from neml2.types import SR2

E, NU = 100.0, 0.3
BATCH = 4


def _strain(batch=BATCH):
    return SR2(torch.randn(batch, 6))


# ---------------------------------------------------------------------------
# LinearIsotropicElasticity — variable declarations
# ---------------------------------------------------------------------------


def test_lie_input_spec():
    m = LinearIsotropicElasticity(E, NU)
    assert m.input_spec == {"strain": SR2}


def test_lie_output_spec_default():
    m = LinearIsotropicElasticity(E, NU)
    assert m.output_spec == {"stress": SR2}


def test_lie_consumed_provided():
    m = LinearIsotropicElasticity(E, NU)
    assert m.consumed_items == {"strain"}
    assert m.provided_items == {"stress"}


def test_lie_call_by_name_stress():
    m = LinearIsotropicElasticity(E, NU)
    strain = _strain()
    state = {"strain": strain.data}
    out = m.call_by_name(state)
    assert set(out.keys()) == {"stress"}
    direct = m(strain)
    assert torch.allclose(out["stress"], direct.data)


# ---------------------------------------------------------------------------
# LinearIsotropicElasticity — forward(v=...) Jacobian pushforward
# ---------------------------------------------------------------------------


def test_lie_forward_v_values_match_forward():
    """forward(v=...) must return the same value as forward()."""
    m = LinearIsotropicElasticity(E, NU)
    strain = _strain()
    V = {"strain": {"strain": torch.eye(6).expand(BATCH, 6, 6)}}
    stress_v, _ = m(strain, v=V)
    stress_fwd = m(strain)
    assert torch.allclose(stress_v.data, stress_fwd.data)


def test_lie_forward_v_jacobian_vs_fd():
    """∂stress/∂strain must match central-difference estimate."""
    m = LinearIsotropicElasticity(E, NU)
    strain = SR2(torch.randn(BATCH, 6, dtype=torch.float64))
    V = {"strain": {"strain": torch.eye(6, dtype=torch.float64).expand(BATCH, 6, 6)}}
    _, v_out = m(strain, v=V)
    J_analytic = v_out["stress"]["strain"].data  # (BATCH, 6, 6)

    h = 1e-7
    J_fd = torch.zeros(BATCH, 6, 6, dtype=torch.float64)
    for j in range(6):
        perturb = torch.zeros(BATCH, 6, dtype=torch.float64)
        perturb[:, j] = h
        s_plus = m(SR2(strain.data + perturb))
        s_minus = m(SR2(strain.data - perturb))
        J_fd[:, :, j] = (s_plus.data - s_minus.data) / (2 * h)

    assert torch.allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Model subclass edge cases
# ---------------------------------------------------------------------------


def test_model_no_forward_raises():
    class Bare(Model):
        input_spec = {}
        output_spec = {}

    with pytest.raises((TypeError, NotImplementedError)):
        Bare()()


def test_model_forward_v_raises_type_error_when_not_supported():
    """A model whose forward doesn't accept v raises TypeError when v is passed."""

    class NoDeriv(Model):
        input_spec = {"x": SR2}
        output_spec = {"y": SR2}

        def forward(self, x: SR2) -> SR2:
            return x

    x = SR2(torch.randn(BATCH, 6))
    V = {"x": {"x": torch.eye(6).expand(BATCH, 6, 6)}}
    with pytest.raises(TypeError):
        NoDeriv()(x, v=V)
