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

"""
Eager validation of the Python-native LinearIsotropicElasticity against the
gold values in tests/unit/models/solid_mechanics/elasticity/LinearIsotropicElasticity.i.

Gold: at E=100, nu=0.3, strain Ee=[0.09, 0.04, -0.02, 0, 0, 0], the C++ model
returns stress S=[13.2692, 9.4231, 4.8077, 0, 0, 0].
"""

import torch

from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity
from neml2.types import SR2

E = 100.0
NU = 0.3
GOLD_STRAIN = SR2(torch.tensor([0.09, 0.04, -0.02, 0.0, 0.0, 0.0], dtype=torch.float64))
GOLD_STRESS = torch.tensor([13.2692, 9.4231, 4.8077, 0.0, 0.0, 0.0], dtype=torch.float64)

GOLD_RTOL = 1e-4
GOLD_ATOL = 1e-4


def test_eager_matches_i_file_gold_values():
    model = LinearIsotropicElasticity(E, NU)
    stress = model(GOLD_STRAIN)
    assert isinstance(stress, SR2)
    assert torch.allclose(stress.data, GOLD_STRESS, rtol=GOLD_RTOL, atol=GOLD_ATOL)


def test_eager_K_and_G_match_lame_formulas():
    """Sanity check: σ = 3K·vol(ε) + 2G·dev(ε), K=E/(3(1-2ν)), G=E/(2(1+ν))."""
    model = LinearIsotropicElasticity(E, NU)
    K = E / (3.0 * (1.0 - 2.0 * NU))
    G = E / (2.0 * (1.0 + NU))
    eps_v = SR2(torch.tensor([0.01, 0.01, 0.01, 0.0, 0.0, 0.0], dtype=torch.float64))
    sig_v = model(eps_v)
    assert isinstance(sig_v, SR2)
    expected_v = torch.tensor([3 * K * 0.01] * 3 + [0, 0, 0], dtype=torch.float64)
    assert torch.allclose(sig_v.data, expected_v, atol=1e-12)
    eps_d = SR2(torch.tensor([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=torch.float64))
    sig_d = model(eps_d)
    expected_d = torch.tensor(
        [2 * G * 0.01, -2 * G * 0.005, -2 * G * 0.005, 0, 0, 0], dtype=torch.float64
    )
    assert torch.allclose(sig_d.data, expected_d, atol=1e-12)


def test_eager_batched_shapes():
    model = LinearIsotropicElasticity(E, NU)
    strain = SR2(torch.randn(5, 7, 6, dtype=torch.float64))
    stress = model(strain)
    assert isinstance(stress, SR2)
    assert stress.shape == (5, 7, 6)


def test_eager_forward_v_returns_jacobian():
    """forward(v=I) returns (stress, dict) with the materialized Jacobian block."""
    model = LinearIsotropicElasticity(E, NU)
    strain = SR2(torch.randn(4, 6, dtype=torch.float64))
    seed = SR2(torch.eye(6, dtype=torch.float64).reshape(6, 1, 6).expand(6, 4, 6))
    V = {"strain": {"strain": seed}}
    out = model(strain, v=V)
    assert isinstance(out, tuple) and len(out) == 2
    stress, jac = out
    assert isinstance(stress, SR2)
    assert isinstance(jac, dict) and "stress" in jac
    # chain-rule blocks are typed wrappers with K leading; move K
    # back to the trailing matrix-column position for this flat-Jacobian assert.
    J = jac["stress"]["strain"].data.movedim(0, -1)  # (4, 6, 6) -- C tangent stiffness
    assert J.shape == (4, 6, 6)
    # J @ ε should reproduce σ exactly (σ = C : ε for linear elasticity)
    assert torch.allclose((J @ strain.data.unsqueeze(-1)).squeeze(-1), stress.data, atol=1e-12)


def test_eager_E_nu_are_calibration_parameters():
    """E, nu are typed calibration parameters."""
    model = LinearIsotropicElasticity(E, NU)
    params = dict(model.named_parameters())
    assert "E" in params and "nu" in params
    assert params["E"].requires_grad and params["nu"].requires_grad
    assert "E" not in dict(model.named_buffers())
    assert "nu" not in dict(model.named_buffers())
