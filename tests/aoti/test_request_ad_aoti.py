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

"""AOTI + cross-route parity for ``request_AD`` (the forward-segment path).

Exports the ``request_ad_forward`` scenario (the ``SurrogateFlowRate`` ML surrogate,
whose first-order chain rule is auto-derived by ``request_AD``), then asserts the
compiled ``jacobian`` matches both the cpp-eager surface (``_EagerModel``) and
finite differences. The request_AD leaf's reverse-mode local Jacobian is traced
inline into the forward-mode chain-rule graph and lowered through AOTInductor
(``SurrogateFlowRate`` is registered for this suite by ``tests/aoti/conftest.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.cli.aoti_export import _reverse_ad_aoti_unsupported_reason

# request_AD AOTI graphs go through the reverse-mode-AD compile path, so they share
# the exporter's support predicate: skipped on torch < 2.11 (no trace_autograd_ops)
# and torch 2.11.x (requires_grad_() unsupported under strict export).
_AUTOGRAD_LOWERING_UNSUPPORTED = _reverse_ad_aoti_unsupported_reason()
_REQUIRES_AUTOGRAD_LOWERING = pytest.mark.skipif(
    _AUTOGRAD_LOWERING_UNSUPPORTED is not None,
    reason=f"request_AD AOTI compilation {_AUTOGRAD_LOWERING_UNSUPPORTED}",
)

_SCENARIO = Path(__file__).parent / "request_ad_forward"
_OUT = "equivalent_plastic_strain_rate"
_INS = ("vonmises_stress", "temperature")


@_REQUIRES_AUTOGRAD_LOWERING
def test_request_ad_aoti_jacobian_matches_eager_and_fd(tmp_path: Path):
    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    out_dir = tmp_path / "request_ad_forward"
    # `-d :` = all (out, structural-in) pairs; request_AD makes them reverse-mode.
    meta = export_model_for_aoti(_SCENARIO / "model.i", "model", out_dir, derivatives=(":",))

    # The forward segment carries a jvp package + both auto-derived pairs.
    seg = meta["segments"][0]
    assert "jvp_package" in seg
    pairs = {(p["out_var"], p["in_var"]) for p in seg["jacobian_pairs"]}
    assert pairs == {(_OUT, "vonmises_stress"), (_OUT, "temperature")}

    aoti = AOTIModel(str(out_dir))
    eager = _EagerModel(str(_SCENARIO / "model.i"), "model")

    # Inputs straddling the surrogate's yield (sy = 1000) so the Jacobian is
    # non-trivial (sub-yield -> zero flow rate and zero derivative).
    torch.manual_seed(0)
    vm = torch.rand(6, dtype=torch.float64) * 800.0 + 900.0
    T = torch.rand(6, dtype=torch.float64) * 400.0 + 800.0
    raw = {"vonmises_stress": vm, "temperature": T}

    out_aoti, jac_aoti = aoti.jacobian(raw)
    out_eager, jac_eager = eager.jacobian(raw)

    # Value + Jacobian parity: py-aoti (== cpp-aoti / cpp-dispatch artifact) vs
    # the cpp-eager surface.
    assert torch.allclose(out_aoti[_OUT], out_eager[_OUT], atol=1e-12)
    for i in _INS:
        assert torch.allclose(jac_aoti[_OUT][i], jac_eager[_OUT][i], atol=1e-10), (
            f"jacobian d({_OUT})/d({i}) aoti vs eager mismatch"
        )

    # Finite-difference cross-check on d(ep)/d(vonmises_stress). The shim's
    # public ``forward`` is typed-positional (it plays the native Model role);
    # the raw-dict forward used here for a quick numeric probe is the binding
    # handle's interface, matching the raw-dict ``jacobian`` above.
    eps = 1.0
    fp = aoti._inner.forward({"vonmises_stress": vm + eps, "temperature": T})[_OUT]
    fm = aoti._inner.forward({"vonmises_stress": vm - eps, "temperature": T})[_OUT]
    fd = (fp - fm) / (2 * eps)
    assert torch.allclose(jac_aoti[_OUT]["vonmises_stress"], fd, rtol=1e-4, atol=1e-6)


# Implicit case: a request_AD leaf (SurrogateFlowRate) INSIDE an ImplicitUpdate
# residual. The Newton-step / IFT graphs differentiate the residual (which contains
# the AD leaf) by reverse-mode; this lowers because the equation-system assembly was
# made strict-export-friendly. The model is the scalar radial-return regression
# fixture (registered via tests/aoti/conftest.py importing _fixtures).
_IMPLICIT_I = (
    Path(__file__).parent.parent
    / "regression/solid_mechanics/viscoplasticity/misc/ml_surrogate/model_scalar.i"
)


def _implicit_inputs(b: int = 4):
    torch.manual_seed(0)
    e = torch.zeros(b, 6, dtype=torch.float64)
    e[:, 0] = 0.02 + 0.005 * torch.rand(b, dtype=torch.float64)  # deviatoric, yields (sy=1000)
    e[:, 1] = -0.01
    e[:, 2] = -0.01
    z = torch.zeros(b, dtype=torch.float64)
    return {
        "E": e,
        "plastic_strain~1": torch.zeros(b, 6, dtype=torch.float64),
        "equivalent_plastic_strain~1": z.clone(),
        "temperature": torch.full((b,), 1200.0, dtype=torch.float64),
        "t": torch.full((b,), 1.0, dtype=torch.float64),
        "t~1": torch.full((b,), 0.5, dtype=torch.float64),
        "t~2": z.clone(),
        "equivalent_plastic_strain~2": z.clone(),
    }


@_REQUIRES_AUTOGRAD_LOWERING
def test_request_ad_in_implicit_residual_matches_eager(tmp_path: Path):
    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    out_dir = tmp_path / "implicit_request_ad"
    meta = export_model_for_aoti(_IMPLICIT_I, "model", out_dir, derivatives=(":",))
    # The implicit segment carries the IFT operator + solve graphs with the
    # residual's du/dg pairs.
    seg = next(s for s in meta["segments"] if s.get("jacobian_package"))
    assert "solve_ift_package" in seg
    # d(equivalent_plastic_strain)/d(temperature) flows through the request_AD
    # surrogate inside the residual.
    pairs = {(p["out_var"], p["in_var"]) for p in seg["jacobian_pairs"]}
    assert ("equivalent_plastic_strain", "temperature") in pairs

    aoti = AOTIModel(str(out_dir))
    eager = _EagerModel(str(_IMPLICIT_I), "model")
    raw = _implicit_inputs()

    out_a, jac_a = aoti.jacobian(raw)
    out_e, jac_e = eager.jacobian(raw)

    # Converged-state parity (the Newton solve runs identically on both routes).
    for name in out_e:
        assert torch.allclose(out_a[name], out_e[name], atol=1e-9), f"value {name} mismatch"
    # Implicit-function-theorem sensitivity parity, including the pair routed
    # through the request_AD leaf.
    worst = max((jac_a[o][i] - jac_e[o][i]).abs().max().item() for o in jac_e for i in jac_e[o])
    assert worst < 1e-8, f"implicit jacobian aoti-vs-eager worst err {worst:.2e}"
