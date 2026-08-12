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

"""Tests for the cold-state gate.

The gate has to switch *elementwise*: in a batched solve some members can be on
their first step while others are well into the history, so a gate that decided
per batch would apply the predictor to members that do not want it. That is the
case worth pinning; the two uniform cases are the easy ones.
"""

from __future__ import annotations

import torch

from neml2.models.common import ScalarMagnitudeGate, SR2MagnitudeGate
from neml2.types import SR2, Scalar

THRESHOLD = 1e-3


def _scalar_gate(pred, ref, threshold=THRESHOLD):
    model = ScalarMagnitudeGate(
        prediction="prediction",
        reference="reference",
        gated="gated",
        threshold=Scalar(torch.tensor(threshold, dtype=torch.float64)),
    )
    out = model.call_by_name({"prediction": Scalar(pred), "reference": Scalar(ref)})
    return out["gated"].data.detach()  # data-ok: test assertion on the numeric result


def test_cold_takes_the_prediction():
    pred = torch.tensor([7.0], dtype=torch.float64)
    ref = torch.tensor([0.0], dtype=torch.float64)
    assert torch.equal(_scalar_gate(pred, ref), pred)


def test_warm_keeps_the_reference():
    """Not merely 'ignores the prediction' -- it must pass the old value through."""
    pred = torch.tensor([7.0], dtype=torch.float64)
    ref = torch.tensor([42.0], dtype=torch.float64)
    assert torch.equal(_scalar_gate(pred, ref), ref)


def test_switches_elementwise_across_a_batch():
    """A batch mixing cold and warm members must gate each one on its own state."""
    pred = torch.tensor([7.0, 7.0, 7.0], dtype=torch.float64)
    ref = torch.tensor([0.0, 42.0, 1e-9], dtype=torch.float64)
    expected = torch.tensor([7.0, 42.0, 7.0], dtype=torch.float64)
    assert torch.equal(_scalar_gate(pred, ref), expected)


def test_scalar_coldness_uses_magnitude_not_sign():
    """A large negative reference is warm. |x| < eps, not x < eps."""
    pred = torch.tensor([7.0], dtype=torch.float64)
    ref = torch.tensor([-42.0], dtype=torch.float64)
    assert torch.equal(_scalar_gate(pred, ref), ref)


def test_sr2_gates_on_the_frobenius_norm():
    model = SR2MagnitudeGate(
        prediction="prediction",
        reference="reference",
        gated="gated",
        threshold=Scalar(torch.tensor(THRESHOLD, dtype=torch.float64)),
    )
    pred = SR2(torch.full((2, 6), 0.5, dtype=torch.float64))
    ref_raw = torch.zeros(2, 6, dtype=torch.float64)
    ref_raw[1, 0] = 1.0  # member 1 is warm, member 0 is cold
    out = model.call_by_name({"prediction": pred, "reference": SR2(ref_raw)})
    got = out["gated"].data.detach()  # data-ok
    assert torch.equal(got[0], torch.full((6,), 0.5, dtype=torch.float64))
    assert torch.equal(got[1], ref_raw[1])
