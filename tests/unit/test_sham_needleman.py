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

"""Composition-level tests for the simplified Sham-Needleman cavitation model.

The individual leaves are covered one at a time by the ``ModelUnitTest`` ``.i``
files under ``tests/models/solid_mechanics/traction_separation_law/``. What
those cannot see is whether the leaves *wire together* into the model the
theory describes, so that is what this module tests, on the three composed
variants under ``fixtures/sham_needleman/``:

``complete.i``
    the full model -- diffusive growth, nucleation, sliding, coalescence;
``no_nucleation.i``
    the same without nucleation, assembled by omitting leaves;
``prescribed_strain_rate.i``
    the full model with the nucleation-driving strain rate supplied as a
    force rather than computed from the local sliding.

Each is checked at three levels:

1. **Wiring** -- the resolved input/output surface of the residual model is
   exactly the expected one, so a renamed or unconnected variable fails loudly
   instead of silently becoming an extra unknown.
2. **Physics** -- the assembled growth term reproduces the closed form quoted
   in the theory, and the separated / compressive branches do what the model
   says they do.
3. **Derivatives** -- the composed Jacobian matches central differences, which
   is the only check that exercises every hand-written chain-rule action
   *through* the composition rather than in isolation.

The trajectory tests at the bottom drive the assembled ``ImplicitUpdate``
through a load history. They need the compiled ``neml2.aoti._aoti`` extension
(the shared C++ Newton loop) and skip without it.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest
import torch

from neml2 import load_input
from neml2.types import Scalar

_FIXTURES = Path(__file__).parent / "fixtures/sham_needleman"

# Mirrors the parameters in every fixture; kept here so a drift between the
# input files and the closed-form cross-check below shows up as a failure.
F_C = 0.5
F_F = 0.8
F_U = 0.99
DIFFUSIVITY = 1.0e-6
SHAPE_FACTOR = 0.61
NUCLEATION_COEFF = 3.0
REFERENCE_STRESS = 50.0
NUCLEATION_EXPONENT = 2.0
SLIDING_VISCOSITY = 4.0
LENGTH_SCALE = 0.5
K_N = 1000.0
K_T = 800.0
K_P = 2000.0
F_0 = 1.0e-2
N_0 = 2.0

_VARIANTS = ("complete", "no_nucleation", "prescribed_strain_rate")

_EXPECTED_INPUTS = {
    "complete": {
        "cavity_fraction",
        "cavity_density",
        "normal_jump",
        "tangential_jump",
        "viscous_sliding",
        "viscous_sliding~1",
        "cavity_density~1",
        "cavity_fraction~1",
        "t",
        "t~1",
    },
    "no_nucleation": {
        "cavity_fraction",
        # a prescribed force here, not an unknown
        "cavity_density",
        "normal_jump",
        "tangential_jump",
        "viscous_sliding",
        "viscous_sliding~1",
        "cavity_fraction~1",
        "t",
        "t~1",
    },
    "prescribed_strain_rate": {
        "cavity_fraction",
        "cavity_density",
        "normal_jump",
        "tangential_jump",
        "viscous_sliding",
        # the force that replaces SlidingDrivenStrainRate
        "strain_rate",
        "viscous_sliding~1",
        "cavity_density~1",
        "cavity_fraction~1",
        "t",
        "t~1",
    },
}

_EXPECTED_OUTPUTS = {
    "complete": {
        "cavity_fraction_residual",
        "cavity_density_residual",
        "viscous_sliding_residual",
    },
    "no_nucleation": {
        "cavity_fraction_residual",
        "viscous_sliding_residual",
    },
    "prescribed_strain_rate": {
        "cavity_fraction_residual",
        "cavity_density_residual",
        "viscous_sliding_residual",
    },
}


def _residual_model(variant: str):
    """The composed residual (pre-Newton) model from a fixture."""
    return load_input(_FIXTURES / f"{variant}.i").get_model("implicit_rate")


def _scalar(x: float) -> Scalar:
    """A single-element *batched* Scalar.

    The batch axis is not cosmetic: ``Model.jacobian``'s structural-zero path
    builds its block with ``torch.zeros(*batch, *sub, *out_base, *in_base)``,
    which degenerates to a zero-argument call for a fully unbatched Scalar.
    Every real caller (the drivers included) carries a batch axis.
    """
    return Scalar(torch.tensor([x], dtype=torch.float64))


def _state(variant: str, **overrides: float) -> dict[str, Scalar]:
    """A reference evaluation point, overridable per test.

    Deliberately mid-trajectory: ``f`` past ``f_c`` so the coalescence branch
    is live, ``N`` and the sliding jump away from their initial values, and a
    finite step so the backward-Euler terms do not degenerate.
    """
    base = {
        "cavity_fraction": 0.6,
        "cavity_fraction~1": 0.55,
        "cavity_density": 3.0,
        "cavity_density~1": 2.9,
        "viscous_sliding": 0.004,
        "viscous_sliding~1": 0.003,
        "normal_jump": 0.2,
        "tangential_jump": 0.01,
        "strain_rate": 1.0e-2,
        "t": 2.0,
        "t~1": 0.0,
    }
    base.update(overrides)
    spec = _residual_model(variant).input_spec
    return {k: _scalar(v) for k, v in base.items() if k in spec}


def _evaluate(variant: str, state: dict[str, Scalar]) -> dict[str, float]:
    model = _residual_model(variant)
    ordered = [state[name] for name in model.input_spec]
    outputs = model(*ordered)
    return {
        name: float(value.data.detach().reshape(()))
        for name, value in zip(model.output_spec, outputs, strict=True)
    }


def _rate_from_residual(residual: float, variable: float, old: float, dt: float) -> float:
    """Invert ``residual = x - x~1 - dt * rate`` for the rate.

    Backward Euler is the only thing between the residual the composed model
    returns and the rate the theory talks about, so recovering the rate keeps
    the physics assertions readable.
    """
    return (variable - old - residual) / dt


# ----------------------------------------------------------------- wiring


@pytest.mark.parametrize("variant", _VARIANTS)
def test_composition_resolves_to_the_expected_surface(variant: str):
    """Each variant wires up to exactly the unknowns and residuals it declares.

    A leaf whose output name does not match a downstream input does not error
    in NEML2 -- the dependency resolver just leaves it dangling and the unmet
    input becomes a model input. Pinning the surface is what turns that into a
    failure.
    """
    model = _residual_model(variant)
    assert set(model.input_spec) == _EXPECTED_INPUTS[variant]
    assert set(model.output_spec) == _EXPECTED_OUTPUTS[variant]


def test_no_nucleation_drops_the_density_equation():
    """Omitting the nucleation leaves removes an unknown, it does not zero one."""
    complete = _residual_model("complete")
    reduced = _residual_model("no_nucleation")
    assert "cavity_density_residual" in complete.output_spec
    assert "cavity_density_residual" not in reduced.output_spec
    assert "cavity_density~1" not in reduced.input_spec


# ---------------------------------------------------------------- physics


def test_growth_term_matches_the_closed_form():
    """The assembled growth rate equals the Sham-Needleman closed form.

    The model splits the growth law into a volume rate and a geometric
    conversion so a second growth mechanism can be summed onto $\\dot V$ later.
    That split is only legitimate if the composition still reproduces

    $$
    \\dot{f}_\\text{growth}
      = \\frac{4\\pi^{3/2}\\mathcal{D} N^{3/2} \\langle T_n \\rangle}
             {h \\sqrt{f}\\, q(f^*)}.
    $$

    Checked on ``no_nucleation``, where the growth term *is* the whole
    ``cavity_fraction`` rate and no summation can mask an error in it.
    """
    f, N = 0.6, 3.0
    un, ut = 0.2, 0.01
    dt = 2.0
    state = _state(
        "no_nucleation",
        cavity_fraction=f,
        cavity_density=N,
        normal_jump=un,
        tangential_jump=ut,
    )
    out = _evaluate("no_nucleation", state)
    rate = _rate_from_residual(
        out["cavity_fraction_residual"], f, float(state["cavity_fraction~1"].data), dt
    )

    # Rebuild the right-hand side independently of the model graph.
    kappa = (F_U - F_C) / (F_F - F_C)
    f_star = F_C + kappa * (f - F_C) if f > F_C else f
    q = -2.0 * math.log(f_star) - (3.0 - f_star) * (1.0 - f_star)
    # The normal traction is the series spring acting on the elastic part of
    # the jump; the viscous part is the smeared cavity volume.
    c = 4.0 * SHAPE_FACTOR / (3.0 * math.sqrt(math.pi))
    unv = c * f**1.5 / math.sqrt(N) - c * F_0**1.5 / math.sqrt(N_0)
    Tn = K_N * (un - unv)
    expected = (
        4.0 * math.pi**1.5 * DIFFUSIVITY * N**1.5 * max(Tn, 0.0) / (SHAPE_FACTOR * math.sqrt(f) * q)
    )

    assert rate == pytest.approx(expected, rel=1e-12)
    # Guard the guard: the closed form must not be trivially zero here, or the
    # comparison would pass for a model that does nothing.
    assert expected > 1e-6


def test_nucleation_adds_to_growth_rather_than_replacing_it():
    """``cavity_fraction_rate`` is the sum of the two contributions.

    The complete model sums the growth and nucleation damage rates through a
    ``ScalarLinearCombination``; the reduced model has only the first. Both
    see the same state here, so the difference isolates the nucleation term.
    """
    kwargs = dict(cavity_fraction=0.6, cavity_density=3.0, normal_jump=0.2)
    dt = 2.0
    f_old = 0.55

    full = _evaluate("complete", _state("complete", **kwargs))
    reduced = _evaluate("no_nucleation", _state("no_nucleation", **kwargs))

    full_rate = _rate_from_residual(full["cavity_fraction_residual"], 0.6, f_old, dt)
    growth_rate = _rate_from_residual(reduced["cavity_fraction_residual"], 0.6, f_old, dt)
    nucleation_rate = _rate_from_residual(full["cavity_density_residual"], 3.0, 2.9, dt)

    # f_dot_nucleation = (f / N) * N_dot, by construction of the damage split.
    assert full_rate - growth_rate == pytest.approx(0.6 / 3.0 * nucleation_rate, rel=1e-12)
    assert nucleation_rate > 0.0


def test_separated_interface_freezes_every_internal_variable():
    """Past ``f_F`` under a positive jump, all three rates vanish.

    This is the invariant the model relies on instead of clamping: both
    tractions are zero on the separated branch, every rate is driven by a
    traction, so the state holds itself. If a rate ever stopped being
    traction-driven this is the test that would catch it.
    """
    f = 0.95  # > f_F
    dt = 2.0
    state = _state("complete", cavity_fraction=f, normal_jump=0.05, tangential_jump=0.02)
    state["cavity_fraction~1"] = _scalar(f)
    out = _evaluate("complete", state)
    assert _rate_from_residual(out["cavity_fraction_residual"], f, f, dt) == pytest.approx(0.0)
    assert _rate_from_residual(out["cavity_density_residual"], 3.0, 2.9, dt) == pytest.approx(0.0)
    assert _rate_from_residual(out["viscous_sliding_residual"], 0.004, 0.003, dt) == pytest.approx(
        0.0
    )


def test_compression_before_separation_suppresses_growth_but_not_shear():
    """A compressive normal jump switches the normal path to the penalty.

    Diffusive growth and nucleation are both bracketed on $\\langle T_n
    \\rangle$, so they stop; sliding is driven by $T_t$, which is untouched,
    so it does not.
    """
    dt = 2.0
    out = _evaluate(
        "complete",
        _state("complete", normal_jump=-0.02, tangential_jump=0.01),
    )
    assert _rate_from_residual(out["cavity_fraction_residual"], 0.6, 0.55, dt) == pytest.approx(0.0)
    assert _rate_from_residual(out["cavity_density_residual"], 3.0, 2.9, dt) == pytest.approx(0.0)
    sliding_rate = _rate_from_residual(out["viscous_sliding_residual"], 0.004, 0.003, dt)
    assert sliding_rate > 0.0


def test_tension_compression_switch_is_continuous_at_zero_jump():
    """Crossing the contact kink does not jump the state rates.

    ``K_n`` and ``K_p`` differ, so the normal *stiffness* is discontinuous by
    design -- but the traction itself is continuous at the switch, and the
    rates that read it must be too.
    """
    eps = 1e-9
    dt = 2.0
    # The kink the damage rates see is the sign change of T_n, not of the raw
    # jump: the spring is in series with the viscous opening, so T_n crosses
    # zero at un == unv(f, N).
    f, N = 0.6, 3.0
    c = 4.0 * SHAPE_FACTOR / (3.0 * math.sqrt(math.pi))
    unv = c * f**1.5 / math.sqrt(N) - c * F_0**1.5 / math.sqrt(N_0)

    def damage_rate(un: float) -> float:
        out = _evaluate(
            "complete",
            _state("complete", cavity_fraction=f, cavity_density=N, normal_jump=un),
        )
        return _rate_from_residual(out["cavity_fraction_residual"], f, 0.55, dt)

    # Continuity is a statement about scaling, not about an absolute
    # tolerance: a genuine jump would leave the same gap however small the
    # probe gets, so assert the gap shrinks with the probe instead.
    gaps = [damage_rate(unv + e) - damage_rate(unv - e) for e in (eps, eps / 10.0)]
    assert damage_rate(unv - eps) == pytest.approx(0.0, abs=1e-14)
    assert gaps[0] == pytest.approx(10.0 * gaps[1], rel=1e-3)
    assert abs(gaps[0]) < 1e-6
    # The two probes must straddle the kink, not sit on one side of it.
    assert unv > eps


def test_combined_shear_and_normal_loading_drives_both_mechanisms():
    """Under mixed loading, damage and sliding both accumulate.

    Separately they are covered above; the point here is that neither
    mechanism switches the other off through the shared ``f*`` softening.
    """
    dt = 2.0
    out = _evaluate("complete", _state("complete", normal_jump=0.2, tangential_jump=0.02))
    assert _rate_from_residual(out["cavity_fraction_residual"], 0.6, 0.55, dt) > 0.0
    assert _rate_from_residual(out["cavity_density_residual"], 3.0, 2.9, dt) > 0.0
    assert _rate_from_residual(out["viscous_sliding_residual"], 0.004, 0.003, dt) > 0.0


def test_prescribed_strain_rate_variant_tracks_its_force():
    """With the strain rate prescribed, ``N_dot`` is linear in it.

    $\\dot N = F_N \\langle T_n / \\Sigma_0 \\rangle^\\beta \\dot\\varepsilon$,
    so doubling the force doubles the density rate exactly -- which is the
    whole reason to expose it as an input.
    """
    dt = 2.0
    rates = []
    for edot in (1.0e-2, 2.0e-2):
        out = _evaluate(
            "prescribed_strain_rate",
            _state("prescribed_strain_rate", strain_rate=edot),
        )
        rates.append(_rate_from_residual(out["cavity_density_residual"], 3.0, 2.9, dt))
    assert rates[1] == pytest.approx(2.0 * rates[0], rel=1e-12)
    assert rates[0] > 0.0


# ------------------------------------------------------------ derivatives


@pytest.mark.parametrize("variant", _VARIANTS)
def test_composed_jacobian_matches_central_differences(variant: str):
    """Every hand-written chain-rule action, exercised through the composition.

    The leaf-level ``.i`` tests already finite-difference each action in
    isolation. They cannot catch an action keyed on the wrong input name,
    because in isolation the name is whatever the leaf declared -- it is only
    once a renamed variable flows between leaves that a mis-keyed action
    silently drops a term. Newton would then converge to the right answer
    slowly, or not at all, with no test failing.
    """
    model = _residual_model(variant)
    state = _state(variant)
    inputs = {name: state[name] for name in model.input_spec}

    _, jac = model.jacobian(inputs)

    h = 1.0e-7
    for out_name in model.output_spec:
        for in_name in model.input_spec:
            perturbed = dict(inputs)
            base = float(inputs[in_name].data)
            step = h * max(abs(base), 1.0)

            perturbed[in_name] = _scalar(base + step)
            plus = _evaluate(variant, perturbed)[out_name]
            perturbed[in_name] = _scalar(base - step)
            minus = _evaluate(variant, perturbed)[out_name]

            numerical = (plus - minus) / (2.0 * step)
            analytic = float(jac[out_name][in_name].data.detach().reshape(()))
            assert analytic == pytest.approx(numerical, rel=1e-5, abs=1e-7), (
                f"{variant}: d{out_name}/d{in_name} analytic {analytic} vs numerical {numerical}"
            )


# ------------------------------------------------------------ trajectories


# The assembled ImplicitUpdate runs the shared C++ Newton loop, which lives in
# the compiled extension. Everything above this line is pure Python and runs
# without it.
def _newton_loop_available() -> bool:
    """``neml2.aoti`` imports the extension eagerly, so probe by importing."""
    try:
        importlib.import_module("neml2.aoti._aoti")
    except ImportError:
        return False
    return True


_needs_newton = pytest.mark.skipif(
    not _newton_loop_available(),
    reason="assembled ImplicitUpdate needs the compiled Newton loop in neml2.aoti._aoti",
)


def _trajectory(
    result: dict[str, torch.Tensor], name: str, kind: str = "output"
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Stack the per-step records of ``name`` into one time series.

    ``TransientDriver.result`` is a flat ``<kind>.<step>.<var>`` dict with one
    entry per step, not a stacked trajectory, so the series has to be
    reassembled here. Step 0 records only the initial conditions, which is why
    the steps are returned alongside the values: ``viscous_sliding`` has no IC
    and so starts at step 1, one entry shorter than ``cavity_fraction``.

    Solved unknowns land under ``output``; prescribed forces land under
    ``input`` and never appear as an output, so a variable that is an unknown
    in one variant and a force in another has to be read from the matching
    side. ``cavity_density`` is exactly that variable.
    """
    records = []
    for key, value in result.items():
        record_kind, step, var = key.split(".", 2)
        if record_kind != kind or var != name:
            continue
        flat = value.reshape(-1)
        assert flat.numel() == 1, f"{key} holds {flat.numel()} values, expected a scalar step"
        records.append((int(step), flat[0]))
    assert records, f"driver recorded no {kind} for {name!r}"
    records.sort()
    return tuple(step for step, _ in records), torch.stack([v for _, v in records])


@_needs_newton
@pytest.mark.parametrize("variant", _VARIANTS)
def test_trajectory_runs_and_damage_is_monotone(variant: str):
    """Driving the full model over a tensile history accumulates damage.

    The fixtures ramp the normal jump far enough to grow ``f`` well above its
    initial 1% by diffusive cavity growth, but not far enough to reach the
    failure fraction: this exercises the pre-coalescence branch of the
    softening map end to end. Freezing at and past separation is covered at
    the model level by ``test_separated_interface_freezes_every_internal_variable``,
    which prescribes the separated state directly instead of trying to reach
    it through a history.
    """
    driver = load_input(_FIXTURES / f"{variant}.i").get_driver("driver")
    driver.run()
    result = driver.result()

    _, f = _trajectory(result, "cavity_fraction")
    assert float(f[0]) == pytest.approx(F_0, rel=1e-10)
    assert torch.all(torch.diff(f) >= -1e-14), "cavity fraction must never heal"
    assert float(f[-1]) > F_0, "no damage accumulated over the history"


@_needs_newton
def test_trajectory_stays_below_failure_so_the_history_never_separates():
    """The complete fixture damages but stops short of the failure fraction.

    This pins the reach of the shipped history. Earlier revisions of this file
    asserted that the trajectory froze after separation, which was never
    exercised: ``f`` plateaus well under ``F_F`` for this ramp, so that
    assertion sat behind a ``pytest.skip`` on every platform and tested
    nothing. If a future fixture change does carry the history into
    separation, this test fails and the freeze assertion can be written for
    real instead of being silently skipped.
    """
    driver = load_input(_FIXTURES / "complete.i").get_driver("driver")
    driver.run()
    result = driver.result()

    _, f = _trajectory(result, "cavity_fraction")
    assert float(f.max()) > 10 * F_0, "history barely damages; the ramp is too weak to be useful"
    assert float(f.max()) < F_F, (
        "history now reaches separation: replace this test with the freeze assertion"
    )


@_needs_newton
def test_nucleation_free_trajectory_holds_the_density_fixed():
    """The reduced variant keeps ``N`` at ``N_0`` and still damages.

    ``cavity_density`` is a prescribed force here, not an unknown, so the
    driver records it under ``input.<step>.cavity_density`` and emits no
    output for it at all. Both halves are asserted: the absence of an output
    record is what "the implicit system dropped the density equation" looks
    like from the driver, and the input series is what pins the value.
    """
    driver = load_input(_FIXTURES / "no_nucleation.i").get_driver("driver")
    driver.run()
    result = driver.result()

    assert not [k for k in result if k.endswith(".cavity_density") and k.startswith("output.")], (
        "cavity_density is solved for in the nucleation-free variant; "
        "it should be a prescribed force"
    )
    _, density = _trajectory(result, "cavity_density", kind="input")
    assert torch.allclose(density, torch.full_like(density, N_0))
    _, f = _trajectory(result, "cavity_fraction")
    assert float(f[-1]) > F_0
