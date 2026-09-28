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

"""Unit coverage for the shared TSL damage cap + viscosity helper.

``viscous_damage`` is the single implementation of the irreversibility cap and
the backward-Euler viscosity regularization that both damage-bearing cohesive
laws (``BilinearTraction`` / ``SalehaniIraniTraction``) route their damage
through. The ``tests/models`` scenarios pin exactly one viscosity point
(``eta = dt = 1``), which is the *blended* branch and nothing else, so the
branches that actually carry risk are unpinned at the law level:

- the rate-independent branch (``eta = 0``), which is the backward-compatibility
  guarantee the regression / verification goldens are built on;
- the ``dt = 0`` branch under viscosity, i.e. a transient's first step, where
  ``alpha`` collapses to zero and damage must not jump;
- the ``eta = dt = 0`` indeterminate form that the branch on ``eta > 0`` exists
  to avoid.

The ``.i`` files also cannot catch a wrong *coefficient*: ``check_dvalue``
compares the analytic pushforward against ``torch.autograd.functional.jvp`` of
the same helper, so a branch-selection error is present on both sides of the
comparison and the two move together. Everything below is therefore pinned
against closed-form ``math`` instead.
"""

from __future__ import annotations

import pytest
import torch

from neml2.models.solid_mechanics.traction_separation_law._viscous_damage import (
    check_viscosity,
    viscous_damage,
)
from neml2.types import Scalar


def _scalar(value) -> Scalar:
    return Scalar(torch.tensor(value, dtype=torch.float64))


def _viscous(d_trial, d_old, t, t_old, eta):
    return viscous_damage(
        _scalar(d_trial), _scalar(d_old), _scalar(t), _scalar(t_old), _scalar(eta)
    )


def _autograd_partials(d_trial, d_old, t, t_old, eta):
    """Differentiate ``d`` w.r.t. ``d_old``, ``t`` and ``t_old`` separately."""
    dtrial = torch.tensor(d_trial, dtype=torch.float64, requires_grad=True)
    dold = torch.tensor(d_old, dtype=torch.float64, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.float64, requires_grad=True)
    told = torch.tensor(t_old, dtype=torch.float64, requires_grad=True)
    eta_t = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    v = viscous_damage(Scalar(dtrial), Scalar(dold), Scalar(tt), Scalar(told), Scalar(eta_t))
    # One graph, three partials, so the graph has to survive the first backward.
    (g_dold,) = torch.autograd.grad(v.d.data.sum(), dold, retain_graph=True)
    (g_t,) = torch.autograd.grad(v.d.data.sum(), tt, retain_graph=True)
    (g_told,) = torch.autograd.grad(v.d.data.sum(), told)
    return v, g_dold, g_t, g_told


# ---------- the construction-time viscosity contract ----------


def test_non_negative_viscosity_is_accepted():
    """Zero (the default) and positive values both pass."""
    for eta in (0.0, 1.0e-12, 2.0, 1.0e6):
        check_viscosity(_scalar(eta), owner="test")


def test_negative_viscosity_is_rejected_at_construction():
    """A negative dashpot has no reading, and its only silent one is inviscid."""
    for eta in (-1.0e-12, -1.0, -5.0):
        with pytest.raises(ValueError, match="viscosity must be non-negative"):
            check_viscosity(_scalar(eta), owner="MyLaw")
    # The message names the model, so a bad coefficient in a composed model is
    # traceable without a traceback.
    with pytest.raises(ValueError, match="MyLaw"):
        check_viscosity(_scalar(-1.0), owner="MyLaw")


def test_a_batched_viscosity_is_left_to_the_tensor_branch():
    """Only a scalar coefficient is decidable on the host; a batched one is not."""
    check_viscosity(_scalar([0.0, 1.0]), owner="test")
    check_viscosity(_scalar([0.0, -1.0]), owner="test")


def test_both_damage_laws_reject_a_negative_viscosity():
    """The check is wired into the leaves, not only into the helper."""
    from neml2.models.solid_mechanics.traction_separation_law.BilinearTraction import (
        BilinearTraction,
    )
    from neml2.models.solid_mechanics.traction_separation_law.SalehaniIraniTraction import (
        SalehaniIraniTraction,
    )

    with pytest.raises(ValueError, match="BilinearTraction: viscosity"):
        BilinearTraction(
            penalty_stiffness="1000.0",
            critical_separation="0.1",
            full_separation="1.0",
            viscosity="-1.0",
        )
    with pytest.raises(ValueError, match="SalehaniIraniTraction: viscosity"):
        SalehaniIraniTraction(
            normal_characteristic_length="0.1",
            tangential_characteristic_length="0.1",
            normal_strength="1.0",
            shear_strength="1.0",
            viscosity="-0.5",
        )


def test_the_default_construction_carries_no_time_input_requirement_change():
    """``eta = 0`` still builds with the same signature the laws shipped with.

    ``time`` / ``time~1`` are unconditional, not tied to the viscosity: the
    irreversibility cap already needs a previous-step damage, so a caller that
    supplies one is stepping and has a clock. This test exists to pin that
    decision -- making the pair optional is a one-line ``default=None`` away and
    would be the natural "only require time when it is used" refinement.
    """
    from neml2.models.solid_mechanics.traction_separation_law.BilinearTraction import (
        BilinearTraction,
    )

    m = BilinearTraction(
        penalty_stiffness="1000.0", critical_separation="0.1", full_separation="1.0"
    )
    for name in ("t", "t~1", "damage~1"):
        assert name in m.input_spec


def test_a_negative_viscosity_in_an_input_file_fails_at_load():
    """The check has to fire on the HIT path too, where the law is really built.

    Constructing the Python class directly is not the same surface a user
    touches; this goes through the same ``parse_text`` -> ``get_model`` route
    ``neml2-run`` takes.
    """
    from pathlib import Path

    import nmhit

    from neml2.factory import _NativeInputFile

    hit_text = """
[Models]
  [law]
    type = BilinearTraction
    critical_separation = '0.1'
    full_separation = '1.0'
    penalty_stiffness = 1000.0
    viscosity = -1.0
  []
[]
"""
    factory = _NativeInputFile(nmhit.parse_text(hit_text), Path("synthetic.i"))
    with pytest.raises(ValueError, match="BilinearTraction: viscosity"):
        factory.get_model("law")


# ---------- the irreversibility cap ----------


def test_regressing_trial_damage_does_not_heal():
    """A ``d_trial`` below ``d_old`` must be floored at ``d_old``."""
    v = _viscous(0.3, 0.6, 5.0, 0.0, 0.0)
    assert not bool(v.advance.data)
    torch.testing.assert_close(v.d.data, torch.tensor(0.6, dtype=torch.float64))
    # Frozen cap: the damage carries the old value's dependence in full.
    torch.testing.assert_close(v.dd_dd_old.data, torch.tensor(1.0, dtype=torch.float64))


def test_advancing_trial_damage_is_taken_verbatim():
    """``eta = 0`` is the rate-independent law, so ``d == d_trial``."""
    v = _viscous(0.6, 0.3, 5.0, 0.0, 0.0)
    assert bool(v.advance.data)
    torch.testing.assert_close(v.d.data, torch.tensor(0.6, dtype=torch.float64))
    torch.testing.assert_close(v.alpha.data, torch.tensor(1.0, dtype=torch.float64))
    # Advancing cap: no dependence survives on ``d_old`` at all.
    torch.testing.assert_close(v.dd_dd_old.data, torch.tensor(0.0, dtype=torch.float64))


def test_advance_mask_marks_each_point_independently():
    """The cap is elementwise, so a mixed batch splits advancing / frozen."""
    v = _viscous([0.6, 0.3, 0.5, 0.5], [0.3, 0.3, 0.7, 0.5], 0.0, 0.0, 0.0)
    assert v.advance.data.tolist() == [True, False, False, False]
    torch.testing.assert_close(v.d.data, torch.tensor([0.6, 0.3, 0.7, 0.5], dtype=torch.float64))
    # The equal pair advances not (strict ``>``), so it stays exactly put.
    torch.testing.assert_close(v.dd_dd_old.data, torch.tensor([0.0, 1.0, 1.0, 1.0]))


# ---------- the rate-independent branch ----------


def test_zero_viscosity_is_rate_independent():
    """``eta = 0`` selects the exact cap and contributes no time dependence."""
    v = _viscous(0.9, 0.3, 7.0, 0.0, 0.0)
    torch.testing.assert_close(v.d.data, torch.tensor(0.9, dtype=torch.float64))
    torch.testing.assert_close(v.alpha.data, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(v.dd_dt.data, torch.tensor(0.0, dtype=torch.float64))


def test_a_mixed_batch_blends_elementwise():
    """``eta`` is a tensor, so inviscid and viscous points coexist in one call.

    This is why the ``eta > 0`` test is a ``where`` rather than a Python
    ``if``: a viscosity carrying batch or sub-batch axes decides per point, and
    ``check_viscosity`` (which only reads a scalar) cannot pre-decide for it.
    """
    # eta = [0, 2, 0]: points 0 and 2 take d_trial, point 1 blends with
    # alpha = dt / (eta + dt) = 1 / 3.
    v = viscous_damage(
        _scalar([0.9, 0.9, 0.9]), _scalar(0.3), _scalar(1.0), _scalar(0.0), _scalar([0.0, 2.0, 0.0])
    )
    torch.testing.assert_close(
        v.alpha.data, torch.tensor([1.0, 1.0 / 3.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        v.d.data, torch.tensor([0.9, 0.3 + (1.0 / 3.0) * 0.6, 0.9], dtype=torch.float64)
    )
    torch.testing.assert_close(
        v.dd_dt.data,
        torch.tensor([0.0, 2.0 / 3.0**2 * 0.6, 0.0], dtype=torch.float64),
    )


def test_rate_independent_result_is_independent_of_dt():
    """The inviscid branch must not read the time inputs at all."""
    for dt in (0.0, 1.0e-12, 1.0, 1.0e6):
        v = _viscous(0.9, 0.3, dt, 0.0, 0.0)
        torch.testing.assert_close(v.d.data, torch.tensor(0.9, dtype=torch.float64))


def test_zero_viscosity_and_zero_dt_is_finite():
    """``eta = dt = 0`` is the 0/0 the ``eta > 0`` branch exists to avoid."""
    v = _viscous(0.4, 0.1, 0.0, 0.0, 0.0)
    assert torch.isfinite(v.d.data).all()
    assert torch.isfinite(v.alpha.data).all()
    assert torch.isfinite(v.dd_dt.data).all()
    torch.testing.assert_close(v.d.data, torch.tensor(0.4, dtype=torch.float64))
    torch.testing.assert_close(v.alpha.data, torch.tensor(1.0, dtype=torch.float64))


# ---------- the viscous blend ----------


def test_viscous_blend_matches_closed_form():
    """``d = d_old + dt / (eta + dt) * (d* - d_old)`` with ``alpha = 1 / 3``."""
    v = _viscous(0.9, 0.3, 1.0, 0.0, 2.0)
    torch.testing.assert_close(v.alpha.data, torch.tensor(1.0 / 3.0, dtype=torch.float64))
    torch.testing.assert_close(v.d.data, torch.tensor(0.5, dtype=torch.float64))


def test_zero_dt_under_viscosity_freezes_damage():
    """A transient's first step: ``alpha = 0``, so ``d`` stays at ``d_old``."""
    v = _viscous(0.9, 0.3, 3.0, 3.0, 1.0)
    torch.testing.assert_close(v.alpha.data, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(v.d.data, torch.tensor(0.3, dtype=torch.float64))


def test_dd_dt_is_maximal_at_zero_dt():
    """``eta / (eta + dt)^2`` peaks at ``dt = 0``, so the first step's tangent is
    finite and large even though its value has not moved yet. This is the
    derivative the ``eta > 0`` branch keeps well-defined; the *value* is the
    half that must be exact."""
    d_trial, d_old, eta = 0.9, 0.3, 1.0
    v = _viscous(d_trial, d_old, 4.0, 4.0, eta)
    expected = eta / (eta + 0.0) ** 2 * (d_trial - d_old)
    torch.testing.assert_close(v.dd_dt.data, torch.tensor(expected, dtype=torch.float64))
    # ...and it decays monotonically as the step grows.
    prev = float("inf")
    for dt in (0.0, 0.5, 1.0, 4.0, 1.0e3):
        got = float(_viscous(d_trial, d_old, dt, 0.0, eta).dd_dt.data)
        assert 0.0 < got <= prev + 1.0e-12
        prev = got


def test_viscosity_pulls_damage_toward_d_old():
    """A larger viscosity can only decrease the advance, never reverse it."""
    d_trial, d_old = 0.8, 0.2
    prev = float("inf")
    for eta in (0.0, 0.1, 1.0, 10.0, 1.0e6):
        d = float(_viscous(d_trial, d_old, 1.0, 0.0, eta).d.data)
        assert d_old < d <= d_trial
        assert d <= prev + 1.0e-12
        prev = d
    # eta -> 0 recovers the inviscid value; eta -> inf recovers d_old.
    assert float(_viscous(d_trial, d_old, 1.0, 0.0, 0.0).d.data) == d_trial
    assert abs(float(_viscous(d_trial, d_old, 1.0, 0.0, 1.0e6).d.data) - d_old) < 1.0e-5


def test_viscosity_damage_depends_on_dt_only():
    """Shifting both time inputs leaves the result invariant."""
    base = float(_viscous(0.9, 0.3, 1.0, 0.0, 2.0).d.data)
    for shift in (-100.0, -1.0, 1.0, 1.0e4):
        shifted = float(_viscous(0.9, 0.3, 1.0 + shift, shift, 2.0).d.data)
        torch.testing.assert_close(
            torch.tensor(shifted, dtype=torch.float64), torch.tensor(base, dtype=torch.float64)
        )


# ---------- the pushforward coefficients ----------


def test_dd_dd_old_is_one_where_frozen_and_one_minus_alpha_where_advancing():
    """``d = (1 - alpha) d_old + alpha d*`` and ``d*``'s own ``d_old`` slope is the mask."""
    v = _viscous([0.9, 0.1], [0.3, 0.3], 1.0, 0.0, 2.0)
    alpha = 1.0 / 3.0
    torch.testing.assert_close(
        v.dd_dd_old.data, torch.tensor([1.0 - alpha, 1.0], dtype=torch.float64)
    )


def test_dd_dt_matches_closed_form():
    """``d(alpha)/d(t) = eta / (eta + dt)^2``, scaled by ``(d* - d_old)``."""
    d_trial, d_old, dt, eta = 0.9, 0.3, 1.0, 2.0
    v = _viscous(d_trial, d_old, dt, 0.0, eta)
    expected = eta / (eta + dt) ** 2 * (d_trial - d_old)
    torch.testing.assert_close(v.dd_dt.data, torch.tensor(expected, dtype=torch.float64))


def test_dd_dt_is_zero_on_the_frozen_branch():
    """``d* == d_old`` leaves no gap for the time derivative to act through."""
    v = _viscous(0.3, 0.6, 4.0, 0.0, 2.0)
    torch.testing.assert_close(v.dd_dt.data, torch.tensor(0.0, dtype=torch.float64))


def test_coefficients_agree_with_autograd():
    """The analytic partials match autograd, and ``d(d)/d(t_old) = -d(d)/d(t)``."""
    for d_trial, d_old, t, t_old, eta in (
        (0.9, 0.3, 1.0, 0.0, 2.0),  # blended, advancing
        (0.2, 0.6, 1.0, 0.0, 2.0),  # blended, frozen
        (0.9, 0.3, 1.0, 0.0, 0.0),  # rate independent
        (0.9, 0.3, 0.0, 0.0, 1.0),  # dt = 0 under viscosity
    ):
        v, g_dold, g_t, g_told = _autograd_partials(d_trial, d_old, t, t_old, eta)
        torch.testing.assert_close(v.dd_dd_old.data, g_dold)
        torch.testing.assert_close(v.dd_dt.data, g_t)
        torch.testing.assert_close(g_told, -g_t)


def test_batched_scalars_broadcast_against_a_scalar_d_old():
    """A spatially uniform ``d_old`` over a batched ``d_trial`` stays elementwise."""
    v = viscous_damage(
        _scalar([0.9, 0.1, 0.5]),
        _scalar(0.3),
        _scalar(1.0),
        _scalar(0.0),
        _scalar(2.0),
    )
    alpha = 1.0 / 3.0
    # Advances on points 0 and 2 (by 0.6 and 0.2), frozen on point 1.
    assert v.advance.data.tolist() == [True, False, True]
    torch.testing.assert_close(
        v.d.data, torch.tensor([0.5, 0.3, 0.3 + alpha * 0.2], dtype=torch.float64)
    )
    torch.testing.assert_close(
        v.dd_dd_old.data, torch.tensor([1.0 - alpha, 1.0, 1.0 - alpha], dtype=torch.float64)
    )
