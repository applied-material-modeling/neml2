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

"""End-to-end tests for shallow boundary renames (``neml2-compile --rename-*``).

Compiles real artifacts and drives the public :class:`~neml2.aoti.AOTIModel`
shim (whose ops delegate to the C++ ``neml2::aoti::Model`` facade) to confirm the
rename is a pure relabel: with an active ``boundary_aliases`` map the reported
``input_spec`` / ``output_spec`` / ``named_parameters`` and every
``forward`` / ``jvp`` / ``jacobian`` / ``param_jacobian`` / ``param_vjp`` dict
key use the RENAMED boundary names, while the values are byte-for-byte identical
to an unrenamed compile of the same model. The internal wiring keeps the original
authored names -- verified indirectly by the numerical parity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2  # noqa: F401 — registers models

_TESTS_ROOT = Path(__file__).resolve().parents[1]
# Forward model: structural input `x` (Scalar) -> output `out` (Scalar), with a
# promotable Scalar parameter `ys.C`. Small + fast to compile.
_MODEL_I = _TESTS_ROOT / "aoti/forward_exp_param/model.i"

# Structural + parameter derivatives, in ORIGINAL names (rename is applied at the
# boundary only; derivative specs resolve against the authored names).
_DERIV = [":", "out:ys.C"]
_RENAMES = {
    "inputs": {"x": "strain"},
    "outputs": {"out": "stress"},
    "parameters": {"ys.C": "Cbar"},
}


@pytest.fixture(scope="module", autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _compile(tmp_path: Path, sub: str, *, renames=None):
    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / sub
    meta = export_model_for_aoti(
        _MODEL_I, "model", out, promoted={"ys.C"}, derivatives=_DERIV, renames=renames
    )
    return AOTIModel(str(out / "model_meta.json")), meta


def test_rename_round_trip_matches_plain_compile(tmp_path):
    """A renamed artifact reports boundary names on every surface and is
    numerically identical to an unrenamed compile of the same model."""
    plain, plain_meta = _compile(tmp_path, "plain")
    ren, ren_meta = _compile(tmp_path, "ren", renames=_RENAMES)

    # Reported names are the boundary names.
    assert list(ren.input_spec) == ["strain"]
    assert list(ren.output_spec) == ["stress"]
    assert list(ren.named_parameters()) == ["Cbar"]
    # ... and the plain compile keeps the authored names (control).
    assert list(plain.input_spec) == ["x"]
    assert list(plain.named_parameters()) == ["ys.C"]

    # Metadata: renamed carries boundary_aliases; plain omits it.
    assert ren_meta["boundary_aliases"] == _RENAMES
    assert "boundary_aliases" not in plain_meta

    x = torch.linspace(0.1, 0.5, 5)

    # forward (dict keys are boundary names in / out).
    op = plain.forward(neml2.Scalar(x))[0].data
    orr = ren.forward(neml2.Scalar(x))[0].data
    assert torch.allclose(op, orr)

    # jacobian: J[stress][strain] == J[out][x].
    _, Jp = plain.jacobian({"x": x})
    _, Jr = ren.jacobian({"strain": x})
    assert list(Jr) == ["stress"] and list(Jr["stress"]) == ["strain"]
    assert torch.allclose(Jp["out"]["x"], Jr["stress"]["strain"])

    # jvp: seeded by boundary input name, keyed by boundary output name.
    _, jp = plain.jvp({"x": x}, {"x": torch.ones(5)})
    _, jr = ren.jvp({"strain": x}, {"strain": torch.ones(5)})
    assert list(jr) == ["stress"]
    assert torch.allclose(jp["out"], jr["stress"])

    # param_jacobian: P[stress][Cbar] == P[out][ys.C].
    _, Pp = plain.param_jacobian({"x": x})
    _, Pr = ren.param_jacobian({"strain": x})
    assert list(Pr["stress"]) == ["Cbar"]
    assert torch.allclose(Pp["out"]["ys.C"], Pr["stress"]["Cbar"])

    # param_vjp: cotangent keyed by boundary output name; result by boundary param.
    cot = torch.ones(5)
    gp = plain.param_vjp({"x": x}, {"out": cot})
    gr = ren.param_vjp({"strain": x}, {"stress": cot})
    assert list(gr) == ["Cbar"]
    assert torch.allclose(gp["ys.C"], gr["Cbar"])


def test_rename_set_parameter_by_boundary_name(tmp_path):
    """A promoted parameter is set / read through its boundary name, and the change
    flows into the value graph identically to the authored-name path."""
    plain, _ = _compile(tmp_path, "plain")
    ren, _ = _compile(tmp_path, "ren", renames=_RENAMES)

    x = torch.linspace(0.1, 0.5, 5)
    plain.set_parameter("ys.C", torch.tensor(0.25))
    ren.set_parameter("Cbar", torch.tensor(0.25))
    op = plain.forward(neml2.Scalar(x))[0].data
    orr = ren.forward(neml2.Scalar(x))[0].data
    assert torch.allclose(op, orr)
