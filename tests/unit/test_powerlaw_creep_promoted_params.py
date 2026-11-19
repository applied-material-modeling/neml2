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

"""Verify PowerLawCreepFlowRate's hand-written parameter JVPs vs finite difference.

The leaf writes its chain rule out by hand rather than deriving it with
autograd, so these closures are checked nowhere else:

    gamma_dot = A <f>^n exp(-Q / (R T))

    d/df = (n / <f>) * gamma_dot
    d/dT = (Q / (R T^2)) * gamma_dot
    d/dA = gamma_dot / A
    d/dn = gamma_dot * log(<f>)
    d/dQ = -gamma_dot / (R T)
    d/dR =  gamma_dot * Q / (R^2 T)

A sign flip or a dropped factor in any of them is silent: the forward value
stays correct and only calibration gradients go wrong.

Two details make this test hit the code it means to test, both of which cost a
false pass on the way here:

* **The parameters must be promoted.** The ``d/dA``/``d/dn``/``d/dQ``/``d/dR``
  branches are guarded by ``if "<p>" in self._promoted_params``, so a model
  built with literal parameter values never enters them. Each parameter here is
  therefore wired to a variable name in the input file, which turns it into a
  runtime input.
* **The seed must be forward-mode.** These closures live in the ``v=`` chain
  rule. ``Model.param_jacobian`` is reverse-mode autograd over the primal and
  bypasses them entirely -- a test written against it passes even with a sign
  deliberately inverted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.factory import load_model
from neml2.types import Scalar

# Representative of the metallic-fuel regime: small coefficient, stress
# exponent above 1 so d/dn is non-trivial, activation energy large enough that
# exp(-Q/(R T)) is far from 1.
_PARAMS = {"A": 1.0e-14, "n": 5.0, "Q": 2.5e5, "R": 8.3143}

# Parameter attribute -> the variable name it is promoted to in the input file.
_VARS = {"A": "A_in", "n": "n_in", "Q": "Q_in", "R": "R_in"}

_F_VAL = 120.0e6  # driving force (von Mises), Pa
_T_VAL = 750.0  # temperature, K


def _write_input(tmp_path: Path, *, with_temperature: bool) -> Path:
    """Emit a PowerLawCreepFlowRate-only .i with every parameter promoted.

    A quoted, non-numeric value that matches no ``[Tensors]`` entry is parsed
    as a bare variable name, which promotes the parameter to a runtime input.
    """
    temp_line = "    temperature       = 'T'\n" if with_temperature else ""
    body = f"""
[Models]
  [model]
    type              = PowerLawCreepFlowRate
    yield_function    = 'f'
    flow_rate         = 'gamma'
{temp_line}    coefficient       = '{_VARS["A"]}'
    exponent          = '{_VARS["n"]}'
    activation_energy = '{_VARS["Q"]}'
    gas_constant      = '{_VARS["R"]}'
  []
[]
"""
    path = tmp_path / ("creep_T.i" if with_temperature else "creep_noT.i")
    path.write_text(body)
    return path


def _scalar_1(x: float) -> Scalar:
    """Batched Scalar with shape (1,) rather than 0-D.

    The 0-D reverse-blocks path is a separate concern, covered by
    ``tests/unit/test_request_ad.py``.
    """
    return Scalar(torch.tensor([x], dtype=torch.float64))


def _base_inputs(with_temperature: bool) -> dict[str, Scalar]:
    inputs = {"f": _scalar_1(_F_VAL)}
    if with_temperature:
        inputs["T"] = _scalar_1(_T_VAL)
    for attr, var in _VARS.items():
        inputs[var] = _scalar_1(_PARAMS[attr])
    return inputs


def _check(model, inputs: dict[str, Scalar], wrt: str) -> None:
    """Compare the analytic JVP w.r.t. input ``wrt`` against central FD."""
    seed = {wrt: {wrt: _scalar_1(1.0)}}
    args = [inputs[n] for n in model.input_spec]
    _out, v_out = model(*args, v=seed)
    analytic = v_out["gamma"][wrt].data[0].item()

    base = inputs[wrt].data[0].item()
    eps = 1.0e-6 * abs(base)

    def _value_at(val: float) -> float:
        perturbed = dict(inputs)
        perturbed[wrt] = _scalar_1(val)
        return model(*[perturbed[n] for n in model.input_spec]).data[0].item()

    fd = (_value_at(base + eps) - _value_at(base - eps)) / (2.0 * eps)

    # Relative only: gamma_dot spans many orders of magnitude here, so a fixed
    # absolute floor would either be meaningless or mask a real error.
    assert analytic == pytest.approx(fd, rel=1e-5), (
        f"PowerLawCreepFlowRate d(gamma)/d({wrt}): "
        f"analytic = {analytic:.6e}, FD = {fd:.6e}, "
        f"rel delta = {(analytic - fd) / fd:.3e}"
    )


@pytest.mark.parametrize("wrt", ["f", "T", *_VARS.values()])
def test_jvp_matches_fd_with_temperature(tmp_path, wrt):
    """Every differentiable input on the temperature-coupled path.

    Covers the two structural inputs (f, T) and all four promoted parameters.
    """
    torch.set_default_dtype(torch.float64)
    model = load_model(_write_input(tmp_path, with_temperature=True), "model")
    _check(model, _base_inputs(with_temperature=True), wrt)


@pytest.mark.parametrize("wrt", ["f", "A_in", "n_in"])
def test_jvp_matches_fd_without_temperature(tmp_path, wrt):
    """The temperature-free path, where the Arrhenius factor is absent.

    Q and R are omitted: with ``temperature`` unwired they do not enter the
    expression, so the leaf emits no derivative branch for them.
    """
    torch.set_default_dtype(torch.float64)
    model = load_model(_write_input(tmp_path, with_temperature=False), "model")
    _check(model, _base_inputs(with_temperature=False), wrt)
