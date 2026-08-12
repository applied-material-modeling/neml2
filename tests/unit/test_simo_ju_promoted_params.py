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

"""Verify Simo-Ju damage parameter derivatives vs central finite difference.

Analog of ``tests/unit/test_mazars_promoted_params.py``. Uses NEML2's
dedicated calibration-derivative API (``Model.param_jacobian``, backed by
``neml2.models.param_ad.param_jacobian``) to compute analytic
``d(output)/d(param)`` blocks, then compares to central FD on the same
parameter at machine-precision-adjacent tolerance.

Four parameters across the two damage leaves:

* ``WeibullDamage``:            Y_in, p1, p2  (attr names)
* ``ViscousDamageRelaxation``:  mu_visc

Silent sign flips or missed cross-terms in any parameter branch corrupt
Adam calibration without raising an exception -- this test catches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.factory import load_model
from neml2.types import Scalar

_WEIBULL_PARAMS = {"Y_in": 300.0, "p1": 5.0, "p2": 2.0}
_VISCOUS_PARAMS = {"mu_visc": 20.0}


def _weibull_input(tmp_path: Path) -> Path:
    """Emit a WeibullDamage-only .i with all 3 parameters as static values."""
    body = f"""
[Models]
  [model]
    type = WeibullDamage
    r    = 'r'
    D    = 'D'
    Y_in = {_WEIBULL_PARAMS['Y_in']}
    p1   = {_WEIBULL_PARAMS['p1']}
    p2   = {_WEIBULL_PARAMS['p2']}
  []
[]
"""
    path = tmp_path / "weibull.i"
    path.write_text(body)
    return path


def _viscous_input(tmp_path: Path) -> Path:
    """Emit a ViscousDamageRelaxation-only .i with mu_visc as static value."""
    body = f"""
[Models]
  [model]
    type    = ViscousDamageRelaxation
    target  = 'target'
    omega   = 'omega'
    time    = 't'
    mu_visc = {_VISCOUS_PARAMS['mu_visc']}
  []
[]
"""
    path = tmp_path / "viscous.i"
    path.write_text(body)
    return path


def _scalar_1(x: float) -> Scalar:
    """Batched Scalar with shape (1,) -- sidesteps a 0-D reshape edge case
    in neml2/models/input_ad.py:188."""
    return Scalar(torch.tensor([x], dtype=torch.float64))


@pytest.mark.parametrize("param", list(_WEIBULL_PARAMS))
def test_weibull_parameter_jacobian_matches_fd(tmp_path, param):
    """Each of the 3 WeibullDamage parameters (Y_in, p1, p2) individually."""
    torch.set_default_dtype(torch.float64)
    input_path = _weibull_input(tmp_path)
    model = load_model(input_path, "model")

    # Fixed non-zero r > Y_in so damage is active
    inputs = {"r": _scalar_1(1500.0)}
    _outs, pjac = model.param_jacobian(inputs, params=[param])
    # pjac["D"][param] is shape (*batch,) since output and param are both Scalars
    analytic = pjac["D"][param][0].item()

    # Central FD by mutating the model's static parameter attribute in place.
    base_val = _WEIBULL_PARAMS[param]
    eps = 1.0e-5 * abs(base_val)
    orig_data = getattr(model, param).data.clone()

    def _D_at(param_val: float) -> float:
        with torch.no_grad():
            getattr(model, param).data.fill_(param_val)
        args = [inputs[n] for n in model.input_spec]
        result = model(*args)
        return result.data[0].item()

    try:
        fd = (_D_at(base_val + eps) - _D_at(base_val - eps)) / (2.0 * eps)
    finally:
        with torch.no_grad():
            getattr(model, param).data.copy_(orig_data)

    assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-10), (
        f"WeibullDamage parameter {param!r}: analytic dD/d{param} = {analytic:.6e}, "
        f"FD = {fd:.6e}, delta = {analytic - fd:.3e}"
    )


@pytest.mark.parametrize("param", list(_VISCOUS_PARAMS))
def test_viscous_parameter_jacobian_matches_fd(tmp_path, param):
    """The single ViscousDamageRelaxation parameter mu_visc."""
    torch.set_default_dtype(torch.float64)
    input_path = _viscous_input(tmp_path)
    model = load_model(input_path, "model")

    # Fixed state: loading branch (target > omega_prev), non-trivial dt
    inputs = {
        "target": _scalar_1(0.7),
        "t": _scalar_1(0.05),
        "omega~1": _scalar_1(0.3),
        "t~1": _scalar_1(0.0),
    }
    _outs, pjac = model.param_jacobian(inputs, params=[param])
    analytic = pjac["omega"][param][0].item()

    base_val = _VISCOUS_PARAMS[param]
    eps = 1.0e-5 * abs(base_val)
    orig_data = getattr(model, param).data.clone()

    def _omega_at(param_val: float) -> float:
        with torch.no_grad():
            getattr(model, param).data.fill_(param_val)
        args = [inputs[n] for n in model.input_spec]
        result = model(*args)
        return result.data[0].item()

    try:
        fd = (_omega_at(base_val + eps) - _omega_at(base_val - eps)) / (2.0 * eps)
    finally:
        with torch.no_grad():
            getattr(model, param).data.copy_(orig_data)

    assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-10), (
        f"ViscousDamageRelaxation parameter {param!r}: "
        f"analytic d(omega)/d{param} = {analytic:.6e}, "
        f"FD = {fd:.6e}, delta = {analytic - fd:.3e}"
    )
