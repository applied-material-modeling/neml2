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


"""Unit tests for :class:`RateCondensation`.

The leaf linearizes a rate-to-driving-force model at zero rate, emitting
$b = f(0)$ and $A = -f'(0)$. These build paths whose linearization is known in
closed form, so the tests check the numbers rather than merely that it ran. The
quadratic term in :class:`_Path` is what distinguishes a *tangent at zero* from
a secant over some interval -- it must contribute nothing.
"""

from __future__ import annotations

import pytest
import torch

from neml2.models.common import RateCondensation
from neml2.models.model import Model
from neml2.types import SR2, Scalar


class _Path(Model):
    r"""$f = \text{trial} - a_0 r - q r^2$, with a real first-order chain rule."""

    input_spec = {"rate": Scalar, "trial": Scalar}
    output_spec = {"drive": Scalar}

    def __init__(self, a0: float, q: float = 0.0) -> None:
        super().__init__()
        self.a0 = a0
        self.q = q

    def forward(self, *inputs, v=None):  # type: ignore[override]
        rate, trial = inputs
        f = trial - self.a0 * rate - self.q * rate * rate
        if v is None:
            return f
        coef = -self.a0 - 2.0 * self.q * rate
        actions = {"rate": lambda V, c=coef: c * V, "trial": lambda V: V}
        return f, self.apply_chain_rule(v, "drive", actions, output=f)


class _RateOnly(Model):
    """A path with nothing but the rate -- a wiring mistake, not a real path."""

    input_spec = {"rate": Scalar}
    output_spec = {"drive": Scalar}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (rate,) = inputs
        return rate * -1.0


def _condense(path: Model, trial: float) -> tuple[float, float]:
    leaf = RateCondensation(model=path, rate="rate", driving_force="drive")
    out = leaf.call_by_name({"trial": Scalar(torch.tensor(trial, dtype=torch.float64))})
    # data-ok: test assertions on the numeric result
    return float(out["coupling"].data), float(out["trial_driving_force"].data)


def test_linear_path_is_reproduced_exactly():
    """For ``f = trial - a0*r`` the condensation is exact: ``b = trial``, ``A = a0``."""
    A, b = _condense(_Path(a0=3.5), trial=7.25)
    assert A == pytest.approx(3.5, rel=1e-12)
    assert b == pytest.approx(7.25, rel=1e-12)


def test_curvature_does_not_leak_into_the_tangent():
    """A quadratic term vanishes at zero rate, so ``A`` must be untouched by it.

    This is the difference between differentiating at the trial point and taking
    a secant over some interval -- only the former is the linearization the
    coordinate solve assumes.
    """
    A, b = _condense(_Path(a0=3.5, q=100.0), trial=7.25)
    assert A == pytest.approx(3.5, rel=1e-12)
    assert b == pytest.approx(7.25, rel=1e-12)


def test_matches_a_finite_difference_of_the_path():
    """An independent check on the jvp, using only forward evaluations."""
    path = _Path(a0=2.75, q=13.0)
    A, _ = _condense(path, trial=1.0)
    h = 1.0e-7
    t = Scalar(torch.tensor(1.0, dtype=torch.float64))

    def f(r: float) -> float:
        out = path.call_by_name({"rate": Scalar(torch.tensor(r, dtype=torch.float64)), "trial": t})
        return float(out["drive"].data)  # data-ok: test assertion

    assert A == pytest.approx(-(f(h) - f(-h)) / (2.0 * h), rel=1e-6)


def test_path_inputs_are_surfaced():
    """The path's other inputs become this leaf's, so the graph supplies them."""
    leaf = RateCondensation(model=_Path(a0=1.0), rate="rate", driving_force="drive")
    assert "trial" in leaf.input_spec
    assert "rate" not in leaf.input_spec


def test_rate_must_be_an_input_of_the_path():
    with pytest.raises(ValueError, match="is not an input of the path model"):
        RateCondensation(model=_Path(a0=1.0), rate="flow_rate", driving_force="drive")


def test_driving_force_must_be_an_output_of_the_path():
    with pytest.raises(ValueError, match="is not an output of the path model"):
        RateCondensation(model=_Path(a0=1.0), rate="rate", driving_force="yield_function")


def test_a_path_with_only_the_rate_is_rejected():
    with pytest.raises(ValueError, match="no inputs besides the rate"):
        RateCondensation(model=_RateOnly(), rate="rate", driving_force="drive")


class _SR2Path(Model):
    """A path whose rate is not a scalar -- the zero seed is undefined for it."""

    input_spec = {"rate": SR2, "trial": Scalar}
    output_spec = {"drive": Scalar}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        _, trial = inputs
        return trial


def test_a_non_scalar_rate_is_rejected():
    """`SR2(torch.zeros(()))` is malformed but silent, so catch it at construction."""
    with pytest.raises(ValueError, match="the zero-rate seed is only defined for a scalar"):
        RateCondensation(model=_SR2Path(), rate="rate", driving_force="drive")
