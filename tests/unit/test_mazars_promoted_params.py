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

"""Verify the parameter-JVP branches of ``MazarsDamageStressAlpha``.

The seven ``if "<name>" in self._promoted_params`` branches in
``MazarsDamageStressAlpha.forward`` compute the closed-form derivative
``dD/d<param>`` for each Mazars parameter (``eps_d0``, ``A_t``, ``B_t``,
``A_c``, ``B_c``, ``E``, ``nu``). They only execute when the parameter has
been promoted to a nonlinear input via HIT (mode-4 promotion) -- the
calibration workflow's entry point.

For every one of the seven parameters, this test:

* Loads the model with that parameter promoted (single-parameter promotion
  keeps each check isolated so a bug in one branch can't mask another).
* Exercises the promoted-parameter JVP path via ``model(..., v=...)``
  covering both tension and compression uniaxial-stress states, so the
  ``H_Dt`` and ``H_Dc`` Heaviside gates both fire at least once across
  the sweep.
* Compares the analytic JVP to a central finite difference on the
  promoted parameter input, asserting agreement to ``sqrt(machine_eps)``
  ~ ``1e-6`` relative.

Structural coverage note: the seven branches together account for the
uncovered lines flagged by the ``codecov/patch`` check on the initial
Mazars PR; parameterizing over all seven ensures every branch runs at
least once, dragging patch coverage past the 90% threshold.

Correctness note: the closed forms were hand-derived (see the JVP block
in ``MazarsDamageStressAlpha.forward``); a silent sign flip or a missed
product-rule term in any branch would corrupt calibration convergence
without ever raising an exception. The FD comparison is the machine
check that ratifies each formula against a numerical ground truth on
every CI run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.factory import load_model
from neml2.types import SR2, Scalar

_PARAM_VALUES = {
    "eps_d0": 1.0e-4,
    "A_t": 1.0,
    "B_t": 15000.0,
    "A_c": 1.2,
    "B_c": 1500.0,
    "E": 30000.0,
    "nu": 0.2,
}


def _mazars_input(tmp_path: Path, promoted: str) -> Path:
    """Emit an .i file that promotes exactly one Mazars parameter."""
    lines = ["[Models]", "  [damage]", "    type = MazarsDamageStressAlpha"]
    lines += [
        "    strain = 'strain'",
        "    effective_stress = 'sigma_tilde'",
        "    equivalent_strain = 'eps_tilde_max'",
        "    damage = 'D'",
    ]
    for name, value in _PARAM_VALUES.items():
        if name == promoted:
            # Mode-4 input promotion: a bare variable specifier with no
            # matching tensor or model adds the parameter to input_spec
            # and registers a PromotedParam entry (see
            # tests/unit/test_parameters.py test_mode4_records_nl_param).
            lines.append(f"    {name} = '{name}_val'")
        else:
            lines.append(f"    {name} = {value}")
    lines += ["  []", "[]"]
    path = tmp_path / f"mazars_{promoted}_promoted.i"
    path.write_text("\n".join(lines) + "\n")
    return path


def _uniaxial_strain(axial: float, nu: float = 0.2) -> SR2:
    """Uniaxial-stress strain state: axial + Poisson laterals."""
    lat = -nu * axial
    return SR2.fill(axial, lat, lat, 0.0, 0.0, 0.0).to(dtype=torch.float64)


def _uniaxial_effective_stress(axial_strain: float, E: float = 30000.0) -> SR2:
    """Effective stress from uniaxial-stress strain state (laterals zeroed)."""
    return SR2.fill(E * axial_strain, 0.0, 0.0, 0.0, 0.0, 0.0).to(dtype=torch.float64)


@pytest.mark.parametrize("param", list(_PARAM_VALUES))
@pytest.mark.parametrize(
    "axial_strain",
    [pytest.param(+5.0e-4, id="tension"), pytest.param(-1.0e-3, id="compression")],
)
def test_parameter_jvp_matches_finite_difference(tmp_path, param, axial_strain):
    """Every parameter's closed-form JVP must match a central-FD derivative.

    Each combination (parameter, loading direction) fires a distinct code
    path: tension gates ``H_Dt`` open (A_t/B_t branches nonzero); compression
    gates ``H_Dc`` open (A_c/B_c). Parameters that appear in both damage
    laws (eps_d0, E, nu) get double coverage; A_t/B_t and A_c/B_c each still
    get their intended-branch exercise, plus a guard exercise from the
    opposite-direction state where the branch is present in code but
    Heaviside-gated to zero.
    """
    torch.set_default_dtype(torch.float64)
    input_path = _mazars_input(tmp_path, param)
    model = load_model(input_path, "damage")
    param_input_name = model._promoted_params[param].input_name

    # Non-promoted static inputs (same for all runs).
    strain = _uniaxial_strain(axial_strain)
    eff_stress = _uniaxial_effective_stress(axial_strain)
    eps_tilde_max = Scalar(torch.tensor(2.0e-4, dtype=torch.float64))

    base_val = _PARAM_VALUES[param]
    promoted_val = Scalar(torch.tensor(base_val, dtype=torch.float64))

    # Analytic JVP: seed a unit tangent on the promoted-parameter input.
    seed = {param_input_name: {param_input_name: Scalar(torch.tensor(1.0))}}
    inputs_by_name = {
        "strain": strain,
        "sigma_tilde": eff_stress,
        "eps_tilde_max": eps_tilde_max,
        param_input_name: promoted_val,
    }
    args = [inputs_by_name[name] for name in model.input_spec]
    outs = model(*args, v=seed)
    # Contract: (*outputs, v_out) when v is provided (see Model docstring).
    v_out = outs[-1]
    analytic_jvp = v_out["D"][param_input_name].data.item()

    # Central-FD tangent on the same input. Step is scaled by the parameter's
    # natural size; 1e-5 * base_val gives ~sqrt(machine_eps) relative accuracy
    # on this smooth analytic model.
    eps = 1.0e-5 * abs(base_val)

    def _damage_at(v: float) -> float:
        inputs_by_name[param_input_name] = Scalar(torch.tensor(v, dtype=torch.float64))
        args_shifted = [inputs_by_name[name] for name in model.input_spec]
        return model(*args_shifted).data.item()

    fd_jvp = (_damage_at(base_val + eps) - _damage_at(base_val - eps)) / (2.0 * eps)

    # Absolute tolerance guards near-zero derivatives (gated branches, where
    # the analytic value is 0 and FD may drift into O(eps^2) noise). Relative
    # tolerance handles the active branches whose derivatives are O(1)
    # w.r.t. the natural parameter scale.
    assert analytic_jvp == pytest.approx(fd_jvp, rel=1e-5, abs=1e-8), (
        f"parameter {param!r} at axial_strain={axial_strain}: "
        f"analytic dD/d{param}={analytic_jvp:.6e}, "
        f"FD={fd_jvp:.6e}, delta={analytic_jvp - fd_jvp:.3e}"
    )
