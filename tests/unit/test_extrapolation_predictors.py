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


"""Unit tests for the `cold` mapping on the extrapolation predictors.

Both predictors gained an optional `unknown:variable` mapping naming what to use
on the step where there is no history to extrapolate from. The cold test is the
extrapolator's own `|t~1 - t~2| <= eps` -- "is there a second history point",
which is a sharper question than "is this variable small": a quantity with a
non-zero initial value (an identity `Fp`, a physical dislocation density) is not
small on the step where it is nonetheless cold.

The regression companions exercise this end to end, but the coverage lane
ignores `tests/regression`, and an end-to-end run cannot isolate a branch anyway.
"""

from __future__ import annotations

import pytest
import torch

from neml2.models.common import ConstantExtrapolationPredictor, LinearExtrapolationPredictor
from neml2.types import Scalar

COLD = 7.0  # what the cold mapping supplies
U_N = 2.0  # u~1
U_NM1 = 1.0  # u~2


def _s(x: float) -> Scalar:
    return Scalar(torch.tensor(x, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# ConstantExtrapolationPredictor
# --------------------------------------------------------------------------- #
def _constant(cold: list[str] | None = None) -> ConstantExtrapolationPredictor:
    return ConstantExtrapolationPredictor(unknowns_SR2=[], unknowns_Scalar=["u", "w"], cold=cold)


def _const_call(model, *, t_n: float, t_nm1: float) -> dict[str, float]:
    args = {"u~1": _s(U_N), "w~1": _s(U_N)}
    if model._cold:
        args |= {"t": _s(2.0), "t~1": _s(t_n), "t~2": _s(t_nm1)}
        args |= {v: _s(COLD) for v in model._cold.values()}
    out = model.call_by_name(args)
    return {k: float(val.data) for k, val in out.items()}  # data-ok: test assertion


def test_constant_without_a_cold_mapping_is_a_pass_through():
    """The pre-existing path must not acquire time inputs it never had."""
    model = _constant()
    assert "t" not in model.input_spec
    assert _const_call(model, t_n=0.0, t_nm1=0.0) == {"u": U_N, "w": U_N}


def test_constant_takes_the_cold_value_when_there_is_no_history():
    """t~1 == t~2: no second history point, so nothing to have converged from."""
    got = _const_call(_constant(["u:u_cold"]), t_n=0.0, t_nm1=0.0)
    assert got["u"] == COLD


def test_constant_takes_the_history_once_there_is_some():
    got = _const_call(_constant(["u:u_cold"]), t_n=1.0, t_nm1=0.0)
    assert got["u"] == U_N


def test_constant_leaves_an_unmapped_unknown_alone_even_when_cold():
    """The mapping is partial by construction -- name only what you want seeded."""
    got = _const_call(_constant(["u:u_cold"]), t_n=0.0, t_nm1=0.0)
    assert got["u"] == COLD
    assert got["w"] == U_N


def test_constant_differentiates_both_branches():
    """A predictor is never differentiated in practice, but the chain rule is
    still expected to route the cold input rather than silently drop it."""
    model = _constant(["u:u_cold"])
    args = {
        "u~1": _s(U_N),
        "w~1": _s(U_N),
        "t": _s(2.0),
        "t~1": _s(0.0),
        "t~2": _s(0.0),
        "u_cold": _s(COLD),
    }
    _, d = model.jvp(args, {"u_cold": _s(1.0)})
    # Cold branch is live, so d(u)/d(u_cold) = 1 and w is untouched by it.
    assert float(d["u"].data) == pytest.approx(1.0)  # data-ok: test assertion
    assert float(d["w"].data) == pytest.approx(0.0)  # data-ok: test assertion


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("no_colon", "is not `unknown:variable`"),
        (":missing_name", "is not `unknown:variable`"),
        ("u:", "is not `unknown:variable`"),
        ("nope:x", "which is not one of the unknowns"),
    ],
)
def test_constant_rejects_a_malformed_cold_entry(entry, match):
    with pytest.raises(ValueError, match=match):
        _constant([entry])


def test_constant_rejects_a_duplicated_unknown():
    with pytest.raises(ValueError, match="is given a cold value twice"):
        _constant(["u:a", "u:b"])


# --------------------------------------------------------------------------- #
# LinearExtrapolationPredictor
# --------------------------------------------------------------------------- #
def _linear(cold: list[str] | None = None) -> LinearExtrapolationPredictor:
    return LinearExtrapolationPredictor(unknowns_SR2=[], unknowns_Scalar=["u"], cold=cold)


def _lin_call(model, *, t: float, t_n: float, t_nm1: float) -> float:
    args = {"u~1": _s(U_N), "u~2": _s(U_NM1), "t": _s(t), "t~1": _s(t_n), "t~2": _s(t_nm1)}
    if model._cold:
        args |= {v: _s(COLD) for v in model._cold.values()}
    return float(model.call_by_name(args)["u"].data)  # data-ok: test assertion


def test_linear_takes_the_cold_value_when_there_is_no_history():
    assert _lin_call(_linear(["u:u_cold"]), t=2.0, t_n=0.0, t_nm1=0.0) == COLD


def test_linear_falls_back_to_u_n_without_a_cold_mapping():
    """Historical behaviour: with no cold value the first step holds u~1."""
    assert _lin_call(_linear(), t=2.0, t_n=0.0, t_nm1=0.0) == U_N


def test_linear_extrapolates_once_there_is_history():
    """u = u_n + (u_n - u_nm1) * (t - t_n) / (t_n - t_nm1) = 2 + 1*2 = 4."""
    assert _lin_call(_linear(["u:u_cold"]), t=3.0, t_n=1.0, t_nm1=0.0) == pytest.approx(4.0)


def test_linear_differentiates_the_cold_branch():
    model = _linear(["u:u_cold"])
    args = {
        "u~1": _s(U_N),
        "u~2": _s(U_NM1),
        "t": _s(2.0),
        "t~1": _s(0.0),
        "t~2": _s(0.0),
        "u_cold": _s(COLD),
    }
    _, d = model.jvp(args, {"u_cold": _s(1.0), "u~1": _s(1.0)})
    # No history: the output IS the cold value, so it carries the whole tangent
    # and u~1 contributes nothing.
    assert float(d["u"].data) == pytest.approx(1.0)  # data-ok: test assertion


def test_linear_differentiates_the_warm_branch():
    model = _linear(["u:u_cold"])
    args = {
        "u~1": _s(U_N),
        "u~2": _s(U_NM1),
        "t": _s(3.0),
        "t~1": _s(1.0),
        "t~2": _s(0.0),
        "u_cold": _s(COLD),
    }
    _, d = model.jvp(args, {"u_cold": _s(1.0)})
    # History exists, so the cold value is masked out entirely.
    assert float(d["u"].data) == pytest.approx(0.0)  # data-ok: test assertion
