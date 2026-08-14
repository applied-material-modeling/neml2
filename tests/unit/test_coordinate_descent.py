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

"""Unit tests for :class:`CoordinateDescentPredictor`.

The predictor solves ``phi(g) + A g = b`` by Gauss-Seidel, where ``phi`` is the
inverse of a supplied explicit rate law. These build a problem whose root is
known by construction -- pick ``g*``, set ``b = phi(g*) + A g*`` -- so the tests
check the answer rather than merely that it ran.
"""

from __future__ import annotations

import pytest
import torch

from neml2.models.common import CoordinateDescentPredictor
from neml2.models.model import Model
from neml2.models.solid_mechanics.crystal_plasticity.PowerLawSlipRule import PowerLawSlipRule
from neml2.types import Scalar

GAMMA0 = 0.1
N = 8.0
M = 3


def _rate_law() -> PowerLawSlipRule:
    """The explicit power law: driving force in, rate out -- i.e. ``phi^-1``."""
    return PowerLawSlipRule(
        resolved_shears="resolved_shears",
        slip_strengths="slip_strengths",
        slip_rates="slip_rates",
        gamma0=Scalar(torch.tensor(GAMMA0, dtype=torch.float64)),
        n=Scalar(torch.tensor(N, dtype=torch.float64)),
    )


def _phi(g: torch.Tensor, tauhat: torch.Tensor) -> torch.Tensor:
    """Inverse of the power law: the driving force that produces rate ``g``."""
    return tauhat * torch.sign(g) * (g.abs() / GAMMA0) ** (1.0 / N)


def _problem(
    g_star: torch.Tensor, coupling: torch.Tensor, tauhat: torch.Tensor
) -> tuple[Scalar, Scalar, Scalar]:
    """``(A, b, tauhat)`` typed, for a problem whose root is exactly *g_star*."""
    b = _phi(g_star, tauhat) + coupling @ g_star
    return (
        Scalar(coupling, sub_batch_ndim=2),
        Scalar(b, sub_batch_ndim=1),
        Scalar(tauhat, sub_batch_ndim=1),
    )


def _spd(m: int, scale: float = 50.0) -> torch.Tensor:
    """A symmetric positive semi-definite coupling with non-negative diagonal."""
    torch.manual_seed(0)
    q = torch.randn(m, m, dtype=torch.float64)
    return scale * (q @ q.T) / m


def _predict(A: Scalar, b: Scalar, tauhat: Scalar, **kw) -> torch.Tensor:
    model = CoordinateDescentPredictor(
        rate_law=_rate_law(),
        driving_force_input="resolved_shears",
        **kw,
    )
    out = model.call_by_name({"coupling": A, "trial_driving_force": b, "slip_strengths": tauhat})
    return out["rate"].data.detach()  # data-ok: test assertion on the numeric result


def test_recovers_a_root_it_was_built_from():
    """The whole point: given ``b = phi(g*) + A g*``, coordinate descent finds ``g*``."""
    g_star = torch.tensor([0.05, -0.02, 0.11], dtype=torch.float64)
    tauhat = torch.full((M,), 60.0, dtype=torch.float64)
    A, b, th = _problem(g_star, _spd(M), tauhat)
    assert torch.allclose(_predict(A, b, th, sweeps=40), g_star, rtol=1e-6, atol=1e-12)


def test_handles_a_dormant_component():
    """A near-zero rate is where the inverted map is vertical -- the hard case."""
    g_star = torch.tensor([0.08, 1e-11, -0.03], dtype=torch.float64)
    tauhat = torch.full((M,), 60.0, dtype=torch.float64)
    A, b, th = _problem(g_star, _spd(M), tauhat)
    got = _predict(A, b, th, sweeps=40)
    assert torch.allclose(got, g_star, rtol=1e-5, atol=1e-12)


def test_zero_coupling_is_the_explicit_rate_law():
    """With ``A = 0`` the answer must be the plain rate law -- one sweep suffices.

    This is the identity that makes the previously best-known predictor a
    special case of this one, so it is worth pinning.
    """
    tauhat = torch.full((M,), 60.0, dtype=torch.float64)
    b_raw = torch.tensor([30.0, -55.0, 61.0], dtype=torch.float64)
    A = Scalar(torch.zeros(M, M, dtype=torch.float64), sub_batch_ndim=2)
    expected = GAMMA0 * torch.sign(b_raw / tauhat) * (b_raw / tauhat).abs() ** N
    got = _predict(A, Scalar(b_raw, sub_batch_ndim=1), Scalar(tauhat, sub_batch_ndim=1), sweeps=1)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-30)


def test_batched():
    """A batch axis in front of the sub-batched components must pass through."""
    g_star = torch.tensor([[0.05, -0.02, 0.11], [0.01, 0.07, -0.04]], dtype=torch.float64)
    tauhat = torch.full((2, M), 60.0, dtype=torch.float64)
    coupling = _spd(M).expand(2, M, M).clone()
    b = _phi(g_star, tauhat) + torch.einsum("bij,bj->bi", coupling, g_star)
    got = _predict(
        Scalar(coupling, sub_batch_ndim=2),
        Scalar(b, sub_batch_ndim=1),
        Scalar(tauhat, sub_batch_ndim=1),
        sweeps=40,
    )
    assert torch.allclose(got, g_star, rtol=1e-6, atol=1e-12)


def test_rejects_an_unknown_driving_force():
    with pytest.raises(ValueError, match="not an input of the rate law"):
        CoordinateDescentPredictor(rate_law=_rate_law(), driving_force_input="nope")


def test_rejects_a_rate_law_with_more_than_one_output():
    """The predictor reads `next(iter(out.values()))`, so two outputs is ambiguous."""

    class _TwoOut(Model):
        input_spec = {"resolved_shears": Scalar}
        output_spec = {"slip_rates": Scalar, "extra": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            (x,) = inputs
            return x, x

    with pytest.raises(ValueError, match="must have exactly one output"):
        CoordinateDescentPredictor(rate_law=_TwoOut(), driving_force_input="resolved_shears")


# --------------------------------------------------------------------------- #
# Layout validation. A mis-wired (A, b) pair must fail rather than index into
# the wrong axis and return a plausible number.
# --------------------------------------------------------------------------- #
def _tauhat(m: int = M) -> Scalar:
    return Scalar(torch.full((m,), 60.0, dtype=torch.float64), sub_batch_ndim=1)


def test_rejects_a_coupling_that_forgot_its_second_axis():
    b = Scalar(torch.ones(M, dtype=torch.float64), sub_batch_ndim=1)
    A = Scalar(torch.ones(M, dtype=torch.float64), sub_batch_ndim=1)
    with pytest.raises(ValueError, match="do not form a system"):
        _predict(A, b, _tauhat(), sweeps=1)


def test_rejects_a_non_square_coupling():
    """Rank 2 is not enough -- `(m, k)` would read entries that are not there."""
    b = Scalar(torch.ones(M, dtype=torch.float64), sub_batch_ndim=1)
    A = Scalar(torch.ones(M, M + 2, dtype=torch.float64), sub_batch_ndim=2)
    with pytest.raises(ValueError, match=r"expected \(3, 3\) to match the driving force"):
        _predict(A, b, _tauhat(), sweeps=1)


def test_a_passthrough_is_indexed_only_on_the_axis_that_is_indexed():
    """The predicate must name the same axis the action indexes.

    `sub_batch[i]` picks the FIRST sub-batch site, so a test on the *last* axis
    only agrees with it at rank 1. A passthrough at `(k, m)` used to satisfy the
    last-axis test and then be sliced along the size-`k` one -- the wrong value,
    or an IndexError when `k < m`, which is the luckier outcome.
    """
    pred = CoordinateDescentPredictor(rate_law=_rate_law(), driving_force_input="resolved_shears")
    per_component = Scalar(torch.arange(M, dtype=torch.float64), sub_batch_ndim=1)
    # (2, M): the last axis is M, but the axis sub_batch[i] would index is size 2.
    other = Scalar(torch.zeros(2, M, dtype=torch.float64), sub_batch_ndim=2)

    out = pred._slice_pt({"per": per_component, "other": other}, 2, M)

    assert float(out["per"].data) == 2.0  # data-ok: indexed, as intended
    assert out["other"] is other  # passed through, not sliced along the size-2 axis


# --------------------------------------------------------------------------- #
# Iterable-export equivalence
# --------------------------------------------------------------------------- #
def test_one_sweep_applied_n_times_equals_n_sweeps():
    """The compiled routes run `iterable_export_form()`'s single sweep in a loop.

    Pinned because the single-sweep form is an instance-level `forward`
    monkeypatch on a `copy.copy`'d module, and the AOTI suite cannot catch it
    silently ceasing to take effect: an unrolled 16-sweep graph looped 16 times
    converges to the same root, so a Newton count would still move.
    """
    n = 5
    g_star = torch.tensor([0.05, -0.02, 0.11], dtype=torch.float64)
    tauhat = torch.full((M,), 60.0, dtype=torch.float64)
    A, b, th = _problem(g_star, _spd(M), tauhat)

    whole = CoordinateDescentPredictor(
        rate_law=_rate_law(), driving_force_input="resolved_shears", sweeps=n
    )
    inputs = {"coupling": A, "trial_driving_force": b, "slip_strengths": th}
    expected = whole.call_by_name(inputs)["rate"]

    form = CoordinateDescentPredictor(
        rate_law=_rate_law(), driving_force_input="resolved_shears", sweeps=n
    ).iterable_export_form()
    assert form.iterations == n
    rate = Scalar(torch.zeros(M, dtype=torch.float64), sub_batch_ndim=1)
    for _ in range(n):
        rate = form.model.call_by_name({**inputs, form.feedback_input: rate})[form.feedback_output]

    # data-ok: test assertion on the numeric result
    assert torch.equal(rate.data, expected.data)


# --------------------------------------------------------------------------- #
# The degenerate one-coordinate layout: A and b as plain scalars, no sub-batch.
# Viscoplasticity condenses onto a single flow rate, so there is nothing to
# sub-batch over and Gauss-Seidel is one exact scalar solve.
# --------------------------------------------------------------------------- #
def test_one_coordinate_system_recovers_its_root():
    """Same construction as the vector case, with the sub-batch axes removed."""
    g_star = torch.tensor(0.05, dtype=torch.float64)
    tauhat = torch.tensor(60.0, dtype=torch.float64)
    a = torch.tensor(35.0, dtype=torch.float64)
    b = _phi(g_star, tauhat) + a * g_star

    got = _predict(Scalar(a), Scalar(b), Scalar(tauhat), sweeps=40)
    assert got.shape == ()
    assert torch.allclose(got, g_star, rtol=1e-6, atol=1e-12)


def test_one_coordinate_system_with_zero_coupling_is_the_rate_law():
    """A = 0 collapses to the plain explicit law, as in the vector case."""
    tauhat = torch.tensor(60.0, dtype=torch.float64)
    b = torch.tensor(30.0, dtype=torch.float64)
    expected = GAMMA0 * (b / tauhat) ** N
    got = _predict(Scalar(torch.zeros(())), Scalar(b), Scalar(tauhat), sweeps=1)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-30)


def test_a_scalar_driving_force_needs_a_scalar_coupling():
    """Mixing the two layouts is a wiring bug, not a broadcast."""
    b = Scalar(torch.tensor(30.0, dtype=torch.float64))
    A = Scalar(torch.ones(M, M, dtype=torch.float64), sub_batch_ndim=2)
    with pytest.raises(ValueError, match="do not form a system"):
        _predict(A, b, Scalar(torch.tensor(60.0, dtype=torch.float64)), sweeps=1)
