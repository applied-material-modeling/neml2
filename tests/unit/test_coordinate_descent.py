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
