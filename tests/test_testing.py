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

from __future__ import annotations

import torch

from neml2.drivers.ModelUnitTest import ModelUnitTest
from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity
from neml2.types import SR2, Scalar, dev, vol


def test_model_unit_test_checks_values_and_dvalue():
    model = LinearIsotropicElasticity(E=200.0, nu=0.25)
    strain = SR2(torch.tensor([0.01, -0.02, 0.03, 0.004, -0.005, 0.006], dtype=torch.float64))
    E = Scalar(torch.tensor(200.0, dtype=torch.float64))
    nu = Scalar(torch.tensor(0.25, dtype=torch.float64))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    expected = 3.0 * K * vol(strain) + 2.0 * G * dev(strain)

    report = ModelUnitTest(
        model,
        {"strain": strain},
        expected_outputs={"stress": expected},
    ).run()

    assert report.value_checks == 1
    assert report.jvp_checks == 1


def test_model_unit_test_accepts_explicit_tangents():
    model = LinearIsotropicElasticity(E=200.0, nu=0.25)
    strain = SR2(torch.tensor([0.01, -0.02, 0.03, 0.004, -0.005, 0.006], dtype=torch.float64))
    tangent_a = SR2(torch.ones(6, dtype=torch.float64))
    tangent_b = SR2(torch.arange(1.0, 7.0, dtype=torch.float64))

    report = ModelUnitTest(
        model,
        {"strain": strain},
        tangents={"strain": [tangent_a, tangent_b]},
    ).run(check_values=False)

    assert report.value_checks == 0
    assert report.jvp_checks == 2
