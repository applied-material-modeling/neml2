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

"""Finite-difference checks for promoted parameters shared by TSL leaves."""

from __future__ import annotations

import pytest
import torch

from neml2.models.solid_mechanics.traction_separation_law.BenzeggaghKenaneFullSeparation import (
    BenzeggaghKenaneFullSeparation,
)
from neml2.models.solid_mechanics.traction_separation_law.BilinearTraction import BilinearTraction
from neml2.models.solid_mechanics.traction_separation_law.CamanhoDavilaCriticalSeparation import (
    CamanhoDavilaCriticalSeparation,
)
from neml2.models.solid_mechanics.traction_separation_law.SalehaniIraniDamage import (
    SalehaniIraniDamage,
)
from neml2.models.solid_mechanics.traction_separation_law.SalehaniIraniTraction import (
    SalehaniIraniTraction,
)


def _check_jvp_against_finite_difference(model, inputs, parameter, output):
    _, jvp = model.jvp(inputs, {parameter: 1.0})
    step = 1.0e-6
    plus = dict(inputs)
    minus = dict(inputs)
    plus[parameter] += step
    minus[parameter] -= step
    y_plus = model(*(model.input_spec[name](plus[name]) for name in model.input_spec))
    y_minus = model(*(model.input_spec[name](minus[name]) for name in model.input_spec))
    finite_difference = (y_plus.data - y_minus.data) / (2.0 * step)
    torch.testing.assert_close(jvp[output].data, finite_difference, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("parameter", ["normal_length", "shear_length"])
def test_salehani_irani_damage_length_jvps(parameter):
    model = SalehaniIraniDamage(
        normal_characteristic_length="normal_length",
        tangential_characteristic_length="shear_length",
    )
    inputs = {
        "normal_separation": 0.5,
        "tangential_separation_1": 0.3,
        "tangential_separation_2": 0.2,
        "normal_length": 1.2,
        "shear_length": 0.8,
    }
    _check_jvp_against_finite_difference(model, inputs, parameter, "trial_damage")


@pytest.mark.parametrize("parameter", ["normal_length", "shear_length"])
def test_salehani_irani_traction_length_jvps(parameter):
    model = SalehaniIraniTraction(
        normal_characteristic_length="normal_length",
        tangential_characteristic_length="shear_length",
        normal_strength=1.0,
        shear_strength=1.0,
    )
    inputs = {
        "damage": 0.3,
        "normal_separation": 0.5,
        "tangential_separation_1": 0.3,
        "tangential_separation_2": -0.2,
        "normal_length": 1.2,
        "shear_length": 0.8,
    }
    _check_jvp_against_finite_difference(model, inputs, parameter, "traction")


@pytest.mark.parametrize("compression", [False, True])
@pytest.mark.parametrize("parameter", ["stiffness", "shear_strength"])
def test_camanho_shared_parameter_jvps(compression, parameter):
    model = CamanhoDavilaCriticalSeparation(
        mode_mixity="beta",
        penalty_stiffness="stiffness",
        normal_strength=10.0,
        shear_strength="shear_strength",
    )
    inputs = {
        "normal_separation": -0.1 if compression else 0.5,
        "beta": 0.8,
        "stiffness": 1000.0,
        "shear_strength": 12.0,
    }
    _check_jvp_against_finite_difference(model, inputs, parameter, "critical_separation")


@pytest.mark.parametrize("compression", [False, True])
@pytest.mark.parametrize("parameter", ["stiffness", "shear_strength"])
def test_benzeggagh_kenane_shared_parameter_jvps(compression, parameter):
    model = BenzeggaghKenaneFullSeparation(
        mode_mixity="beta",
        critical_separation="delta_c",
        penalty_stiffness="stiffness",
        mode_I_fracture_toughness=1.0,
        mode_II_fracture_toughness=2.0,
        shear_strength="shear_strength",
        eta=2.0,
    )
    inputs = {
        "normal_separation": -0.1 if compression else 0.5,
        "beta": 0.8,
        "delta_c": 0.012,
        "stiffness": 1000.0,
        "shear_strength": 12.0,
    }
    _check_jvp_against_finite_difference(model, inputs, parameter, "full_separation")


def test_bilinear_traction_shared_stiffness_jvp():
    model = BilinearTraction(penalty_stiffness="stiffness", normal_penetration="penetration")
    inputs = {
        "damage": 0.3,
        "normal_separation": 0.02,
        "penetration": -0.001,
        "tangential_separation_1": 0.01,
        "tangential_separation_2": -0.03,
        "stiffness": 1000.0,
    }
    _check_jvp_against_finite_difference(model, inputs, "stiffness", "traction")
