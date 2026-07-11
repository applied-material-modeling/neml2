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

"""Tests for substepping (adaptive sub-incrementation) in ``ImplicitUpdate``.

Phase 0 here pins the recoverable-exception contract: a non-converging Newton
solve raises :class:`neml2.solvers.ConvergenceError` (a ``RuntimeError``
subclass), so an outer driver can catch it and cut the increment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2
from neml2.es._helpers import (
    SubstepRole,
    change_lag_order,
    classify_substep_roles,
    lag_order,
)
from neml2.solvers import ConvergenceError, RetCode

_REPO = Path(__file__).resolve().parents[2]
_IMPLICIT = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"


def _implicit_inputs(m, b=2):
    """One Scalar per input at ``(b,)``; ``x_rate`` nonzero so the backward-Euler
    residual at the zero initial guess is nonzero (won't converge in 0 iters)."""
    vals = {
        "x": 0.0,
        "x~1": 0.0,
        "t": 1.0,
        "t~1": 0.0,
        "x_rate": 1.0,
    }
    return tuple(
        m.input_spec[name](torch.full((b,), vals[name], dtype=torch.float64))
        for name in m.input_spec
    )


def test_solve_converges_by_default():
    """Baseline: with the shipped solver config the simple integration converges
    and ``_solve`` does not raise."""
    m = neml2.load_model(str(_IMPLICIT), "model")
    out = m(*_implicit_inputs(m))
    assert m.last_status is RetCode.SUCCESS
    # ScalarBackwardEuler: x = x~1 + (t - t~1) * x_rate = 0 + 1 * 1 = 1.
    assert torch.allclose(out[0].data, torch.ones(2, dtype=torch.float64))


def test_nonconvergence_raises_recoverable_error():
    """A capped-iteration solve that cannot converge raises the *recoverable*
    ConvergenceError (not a bare RuntimeError), so callers can subdivide/retry."""
    m = neml2.load_model(str(_IMPLICIT), "model")
    m.solver.miters = 0  # zero Newton iterations -> initial residual != 0 -> MAXITER
    with pytest.raises(ConvergenceError):
        m(*_implicit_inputs(m))
    assert m.last_status is not RetCode.SUCCESS
    # Recoverable errors subclass RuntimeError so legacy `except RuntimeError` holds.
    assert issubclass(ConvergenceError, RuntimeError)


def test_try_solve_does_not_raise():
    """``_try_solve`` reports failure via its return code instead of raising --
    the probe primitive the substep driver uses to decide whether to subdivide."""
    m = neml2.load_model(str(_IMPLICIT), "model")
    m.solver.miters = 0
    state = dict(zip(m.input_spec, _implicit_inputs(m), strict=True))
    input_sbn = {name: 0 for name in m.input_spec}
    solution, ret = m._try_solve(state, sub_batch_ndim=input_sbn)
    assert solution is None
    assert ret is not RetCode.SUCCESS


# ---------------------------------------------------------------------------
# Phase 1: lag utilities + substep role classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "base", "lag"),
    [("x", "x", 0), ("x~1", "x", 1), ("elastic_strain~2", "elastic_strain", 2)],
)
def test_lag_order(name, base, lag):
    assert lag_order(name) == (base, lag)
    assert change_lag_order(base, lag) == name
    # round-trip: re-tagging to lag 0 drops the suffix.
    assert change_lag_order(name, 0) == base


def test_lag_order_rejects_malformed():
    with pytest.raises(ValueError):
        lag_order("a~b")  # non-integer lag
    with pytest.raises(ValueError):
        lag_order("a~1~2")  # too many separators


def _roles(inputs, unknowns, incremental=()):
    return {
        n: (i.role, i.pair)
        for n, i in classify_substep_roles(inputs, unknowns, incremental).items()
    }


def test_classify_standard_rate_form():
    """A canonical rate-form set: an unknown with its old state, a paired
    force + time (interpolate), a lone static force."""
    inputs = ["x", "x~1", "t", "t~1", "x_rate"]
    r = _roles(inputs, unknowns=["x"])
    assert r["x"] == (SubstepRole.UNKNOWN, None)  # current guess slot of the unknown
    assert r["x~1"] == (SubstepRole.OLD_STATE, "x")  # chain
    assert r["t"] == (SubstepRole.CUR_FORCE, "t~1")  # interpolate (time)
    assert r["t~1"] == (SubstepRole.OLD_FORCE, "t")  # interpolate (time)
    assert r["x_rate"] == (SubstepRole.STATIC, None)  # lone force -> hold


def test_classify_paired_force():
    """A force with both current + old counterparts interpolates from both sides."""
    inputs = ["stress~1", "T", "T~1"]
    r = _roles(inputs, unknowns=["stress"])
    assert r["stress~1"] == (SubstepRole.OLD_STATE, "stress")
    assert r["T"] == (SubstepRole.CUR_FORCE, "T~1")
    assert r["T~1"] == (SubstepRole.OLD_FORCE, "T")


def test_classify_incremental_opt_in():
    """A lone increment is STATIC by default but INCREMENTAL when the user lists it."""
    inputs = ["u~1", "deformation_increment"]
    assert _roles(inputs, ["u"])["deformation_increment"] == (SubstepRole.STATIC, None)
    r = _roles(inputs, ["u"], incremental=["deformation_increment"])
    assert r["deformation_increment"] == (SubstepRole.INCREMENTAL, None)
    assert r["u~1"] == (SubstepRole.OLD_STATE, "u")  # unaffected by the opt-in


def test_implicit_update_stores_and_validates_options():
    """The options round-trip onto the model (parsed + stored so ``neml2-compile``
    can read them as attributes to build the compiled substep driver)."""
    m = neml2.load_model(str(_IMPLICIT), "model")
    assert m.max_substepping_level == 0  # default off
    assert m.incremental_variables == []


def test_incremental_variables_must_be_inputs():
    """A typo'd incremental variable name is rejected at construction."""
    m = neml2.load_model(str(_IMPLICIT), "model")
    with pytest.raises(ValueError, match="not inputs"):
        neml2.ImplicitUpdate(m.system, m.solver, incremental_variables=["not_a_real_input"])


# ---------------------------------------------------------------------------
# Substepping is an AOTI-compile-only feature. The eager runtime does not
# sub-increment: it parses + stores the options (so `neml2-compile` can read
# them) but REJECTS evaluating a model that requests substepping.
# ---------------------------------------------------------------------------

_NL = _REPO / "tests" / "aoti" / "implicit_substep_nl" / "model.i"


def _nl_inputs(dt, x1=0.2):
    return {
        "x": torch.full((1,), x1, dtype=torch.float64),
        "x~1": torch.full((1,), x1, dtype=torch.float64),
        "t": torch.full((1,), dt, dtype=torch.float64),
        "t~1": torch.zeros(1, dtype=torch.float64),
    }


def test_eager_rejects_max_substepping_level():
    """A model with ``max_substepping_level > 0`` loads eagerly (the option is
    stored for the AOTI compiler) but raises on eager forward / jacobian --
    substepping is only performed by the compiled AOTI routes."""
    m = neml2.load_model(str(_NL), "model").to(torch.float64)
    assert m.max_substepping_level == 8  # parsed + stored for neml2-compile
    args = tuple(m.input_spec[n](_nl_inputs(1.0)[n]) for n in m.input_spec)
    with pytest.raises(NotImplementedError, match="substepping"):
        m(*args)
    # The v-path (jacobian) is rejected by the same guard.
    with pytest.raises(NotImplementedError, match="substepping"):
        m.jacobian(_nl_inputs(1.0))


def test_eager_rejects_incremental_variables():
    """``incremental_variables`` (even with level 0) also rejects eager eval."""
    base = neml2.load_model(str(_NL), "model").to(torch.float64)
    m = neml2.ImplicitUpdate(base.system, base.solver, incremental_variables=["t"])
    assert m.max_substepping_level == 0
    args = tuple(m.input_spec[n](_nl_inputs(1.0)[n]) for n in m.input_spec)
    with pytest.raises(NotImplementedError, match="substepping"):
        m(*args)
