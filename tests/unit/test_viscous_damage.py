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

"""Unit coverage for the registered viscous damage updates."""

from __future__ import annotations

import math
from pathlib import Path

import nmhit
import pytest
import torch

from neml2.factory import _NativeInputFile
from neml2.models.solid_mechanics.traction_separation_law import (
    BackwardEulerViscousDamage,
    ExponentialViscousDamage,
)
from neml2.types import Scalar


def _scalar(value) -> Scalar:
    return Scalar(torch.tensor(value, dtype=torch.float64))


def _evaluate(model, trial=0.9, old=0.3, time=1.0, old_time=0.0) -> Scalar:
    return model(_scalar(trial), _scalar(old), _scalar(time), _scalar(old_time))


@pytest.mark.parametrize(
    ("model_type", "alpha"),
    [
        (BackwardEulerViscousDamage, 1.0 / 3.0),
        (ExponentialViscousDamage, 1.0 - math.exp(-0.5)),
    ],
)
def test_updates_match_closed_form(model_type, alpha):
    model = model_type(viscosity="2.0")
    damage = _evaluate(model)
    torch.testing.assert_close(damage.data, _scalar(0.3 + alpha * 0.6).data)


@pytest.mark.parametrize("model_type", [BackwardEulerViscousDamage, ExponentialViscousDamage])
def test_zero_viscosity_is_exactly_rate_independent(model_type):
    model = model_type(viscosity="0.0")
    for dt in (0.0, 1.0e-12, 1.0, 1.0e6):
        damage = _evaluate(model, time=dt)
        torch.testing.assert_close(damage.data, _scalar(0.9).data)


@pytest.mark.parametrize("model_type", [BackwardEulerViscousDamage, ExponentialViscousDamage])
def test_damage_is_irreversible(model_type):
    model = model_type(viscosity="2.0")
    damage = _evaluate(model, trial=0.1, old=0.3)
    torch.testing.assert_close(damage.data, _scalar(0.3).data)


@pytest.mark.parametrize("model_type", [BackwardEulerViscousDamage, ExponentialViscousDamage])
def test_zero_dt_freezes_viscous_damage(model_type):
    model = model_type(viscosity="2.0")
    damage = _evaluate(model, time=4.0, old_time=4.0)
    torch.testing.assert_close(damage.data, _scalar(0.3).data)


@pytest.mark.parametrize("model_type", [BackwardEulerViscousDamage, ExponentialViscousDamage])
def test_negative_viscosity_is_rejected(model_type):
    with pytest.raises(ValueError, match="viscosity must be non-negative"):
        model_type(viscosity="-1.0")


def test_negative_viscosity_is_rejected_through_hit():
    text = """
[Models]
  [update]
    type = BackwardEulerViscousDamage
    viscosity = -1.0
  []
[]
"""
    factory = _NativeInputFile(nmhit.parse_text(text), Path("synthetic.i"))
    with pytest.raises(ValueError, match="BackwardEulerViscousDamage: viscosity"):
        factory.get_model("update")


def test_updates_have_the_same_public_contract():
    backward_euler = BackwardEulerViscousDamage(viscosity="1.0")
    exponential = ExponentialViscousDamage(viscosity="1.0")
    assert backward_euler.input_spec == exponential.input_spec
    assert backward_euler.output_spec == exponential.output_spec
    assert list(backward_euler.input_spec) == ["trial_damage", "damage~1", "t", "t~1"]
    assert list(backward_euler.output_spec) == ["damage"]
