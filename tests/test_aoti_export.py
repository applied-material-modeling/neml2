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

"""Tests for neml2.cli.aoti_export.export_model_for_aoti."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import neml2  # noqa: F401 — ensures all models are registered

_TESTS_NATIVE = Path(__file__).parent
_ELASTICITY_I = _TESTS_NATIVE / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
_J2_NATIVE_I = _TESTS_NATIVE / "fixtures/j2_linear_isoharden/model.i"

pytestmark = pytest.mark.aoti_compile


@pytest.fixture(scope="session")
def forward_export(tmp_path_factory):
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_forward")
    meta = export_model_for_aoti(_ELASTICITY_I, "model", out_dir)
    return meta, out_dir


@pytest.fixture(scope="session")
def implicit_export(tmp_path_factory):
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_implicit")
    meta = export_model_for_aoti(_J2_NATIVE_I, "return_map", out_dir)
    return meta, out_dir


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def test_export_forward_model_produces_artifacts(forward_export):
    meta, out_dir = forward_export

    # Phase-7 unified schema: every export is "composed". A pure forward model
    # is a one-segment composition with kind=forward.
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "forward"
    assert (out_dir / "model.pt2").exists()
    assert (out_dir / "model_meta.json").exists()


def test_export_forward_model_metadata(forward_export):
    _, out_dir = forward_export
    meta = json.loads((out_dir / "model_meta.json").read_text())
    assert meta["type"] == "composed"
    assert meta["inputs"] == [{"name": "strain", "var_size": 6, "var_type": "SR2"}]
    assert meta["outputs"] == [{"name": "stress", "var_size": 6, "var_type": "SR2"}]
    seg = meta["segments"][0]
    assert seg["kind"] == "forward"
    assert seg["inputs"] == [{"name": "strain", "var_size": 6, "var_type": "SR2"}]
    assert seg["outputs"] == [{"name": "stress", "var_size": 6, "var_type": "SR2"}]
    # Forward segments carry a `jvp_package` next to the value package.
    assert seg["jvp_package"] == "model_jvp.pt2"
    assert (out_dir / "model_jvp.pt2").exists()


def test_export_forward_jvp_matches_finite_difference(forward_export):
    """The JVP package's flat Jacobian must agree with central differences on
    the value package."""
    from neml2.export import load_package

    _, out_dir = forward_export

    val_pkg = load_package(out_dir / "model.pt2")
    jvp_pkg = load_package(out_dir / "model_jvp.pt2")

    gen = torch.Generator().manual_seed(0)
    strain = torch.randn(4, 6, generator=gen, dtype=torch.float64) * 1e-3

    *_, J = jvp_pkg(strain)
    # Elasticity is linear: stress = C : strain, so J = C (6x6) independent of
    # the operating point. Cross-check column-by-column via FD.
    eps = 1e-6
    for k in range(6):
        bumped = strain.clone()
        bumped[:, k] += eps
        s0 = val_pkg(strain)
        s1 = val_pkg(bumped)
        if isinstance(s0, tuple):
            s0 = s0[0]
            s1 = s1[0]
        fd = (s1 - s0) / eps
        assert torch.allclose(J[..., k], fd, rtol=5e-4, atol=5e-4)


# test_export_forward_model_output_matches_eager was retired: the
# eager-vs-compiled comparison is now covered end-to-end by the
# `python/tests/native/aoti/` suite, which goes through the pybind binding
# (raw-Tensor I/O) instead of the legacy `load_package` path that returned
# typed wrappers and tripped `torch.allclose(Tensor, SR2)`.


# ---------------------------------------------------------------------------
# Implicit model
# ---------------------------------------------------------------------------


def test_export_implicit_model_produces_artifacts(implicit_export):
    meta, out_dir = implicit_export

    # Phase-7 unified schema: a single ImplicitUpdate is a one-segment
    # composition with kind=implicit. Artifact filenames keep their original
    # rhs/step/predictor suffixes (no `_seg0_` prefix for single-segment).
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "implicit"
    assert (out_dir / "return_map_rhs.pt2").exists()
    assert (out_dir / "return_map_step.pt2").exists()
    assert (out_dir / "return_map_ift.pt2").exists()
    assert (out_dir / "return_map_meta.json").exists()
    assert meta["segments"][0]["ift_package"] == "return_map_ift.pt2"


def test_export_implicit_ift_satisfies_IFT_identity(implicit_export):
    """The compiled IFT graph returns du/dg satisfying ``A @ (du/dg) + B = 0``
    where A, B come from the same Newton system. Validates the math at a
    random (u, g) without requiring a converged state — the IFT identity is
    purely local."""
    from neml2.equation_systems import DenseImplicitSensitivity, ModelNonlinearSystem
    from neml2.export import load_package
    from neml2.factory import load_input

    f = load_input(_J2_NATIVE_I)
    return_map = f.get_model("return_map")
    _, out_dir = implicit_export
    ift_pkg = load_package(out_dir / "return_map_ift.pt2")

    system: ModelNonlinearSystem = return_map.system  # type: ignore[attr-defined]
    sensitivity = DenseImplicitSensitivity(system)

    gen = torch.Generator().manual_seed(0)
    u = torch.randn(2, system.ulayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-3
    g = torch.randn(2, system.glayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-2

    J = ift_pkg(u, g)
    if isinstance(J, tuple):
        J = J[0]
    A, B = sensitivity(u, g)
    # IFT identity: A J + B = 0.
    residual = A @ J + B
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)


def test_export_implicit_model_metadata(implicit_export):
    _, out_dir = implicit_export
    meta = json.loads((out_dir / "return_map_meta.json").read_text())
    assert meta["type"] == "composed"
    seg = meta["segments"][0]
    assert seg["kind"] == "implicit"

    unknown_names = [u["name"] for u in seg["unknowns"]]
    assert unknown_names == ["plastic_strain", "equivalent_plastic_strain", "flow_rate"]
    unknown_sizes = [u["var_size"] for u in seg["unknowns"]]
    assert unknown_sizes == [6, 1, 1]

    # u_size = sum of unknown var_sizes
    assert seg["u_size"] == 8
    # g_size = system.glayout.storage_size() — the AOTI residual surface's
    # given variables only. The predictor's extra inputs (e.g. `flow_rate~1`
    # added by ConstantExtrapolationPredictor) are NOT part of g_flat; they
    # flow into the predictor `.pt2` separately. g_flat packs only the 5
    # residual-surface givens: total_strain(6), plastic_strain~1(6),
    # equivalent_plastic_strain~1(1), t(1), t~1(1) → 15.
    assert seg["g_size"] == 15

    assert seg["atol"] == pytest.approx(1e-12)
    assert seg["rtol"] == pytest.approx(1e-12)
    assert seg["miters"] == 25


def test_export_implicit_rhs_output_matches_eager(implicit_export):
    """Compiled DenseRHS must match Python DenseRHS at a random state."""
    from neml2.equation_systems import DenseRHS, ModelNonlinearSystem
    from neml2.export import load_package
    from neml2.factory import load_input

    # Register all native types and load the implicit model
    f = load_input(_J2_NATIVE_I)
    return_map = f.get_model("return_map")

    _, out_dir = implicit_export

    rhs_pkg = load_package(out_dir / "return_map_rhs.pt2")

    system: ModelNonlinearSystem = return_map.system  # type: ignore[attr-defined]
    rhs = DenseRHS(system)

    gen = torch.Generator().manual_seed(42)
    u = torch.randn(4, system.ulayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-3
    g = torch.randn(4, system.glayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-2

    eager = rhs(u, g)
    compiled = rhs_pkg(u, g)
    if isinstance(compiled, tuple):
        compiled = compiled[0]

    assert torch.allclose(eager, compiled, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Unregistered type error
# ---------------------------------------------------------------------------


def test_export_implicit_with_predictor_emits_third_artifact(implicit_export):
    """j2_linear_isoharden.i has a ConstantExtrapolationPredictor — verify the
    AOTI export now emits a third .pt2 plus predictor metadata in the JSON.
    Predictor info lives on the (single) implicit segment under the Phase-7
    unified schema."""

    meta, out_dir = implicit_export
    seg = meta["segments"][0]

    assert "predictor_package" in seg
    assert (out_dir / seg["predictor_package"]).exists()

    # Predictor inputs are the history values for each unknown.
    pred_in_names = [v["name"] for v in seg["predictor_inputs"]]
    assert pred_in_names == [
        "plastic_strain~1",
        "equivalent_plastic_strain~1",
        "flow_rate~1",
    ]
    pred_out_names = [v["name"] for v in seg["predictor_outputs"]]
    assert pred_out_names == ["plastic_strain", "equivalent_plastic_strain", "flow_rate"]


def test_export_unregistered_type_raises(tmp_path):
    from neml2.cli.aoti_export import export_model_for_aoti

    hit_text = """
[Models]
  [m]
    type = VoceIsotropicHardening
    saturated_hardening = 1
    saturation_rate = 1
  []
[]
"""
    hit_file = tmp_path / "unregistered.i"
    hit_file.write_text(hit_text)

    with pytest.raises(KeyError, match="VoceIsotropicHardening"):
        export_model_for_aoti(hit_file, "m", tmp_path / "out")


# ---------------------------------------------------------------------------
# nonlinear parameter (mode 3) flows through AOTI export
# ---------------------------------------------------------------------------


def test_export_with_nl_parameter_matches_eager(tmp_path):
    """A ComposedModel whose child uses a [Models]-wired parameter exports cleanly
    and matches eager output. The auto-included provider becomes an extra
    segment in the chain; the host's parameter slot resolves to a runtime
    input from the provider's output."""
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.export import load_package
    from neml2.factory import load_model

    hit_text = """
[Tensors]
  [K_x]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 1000.0]))'
  []
  [K_y]
    type = Python
    expr = 'Scalar(torch.tensor([500.0, 1500.0]))'
  []
[]

[Models]
  [K_model]
    type = ScalarLinearInterpolation
    argument = 'temperature'
    abscissa = 'K_x'
    ordinate = 'K_y'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 'K_model'
  []
  [chain]
    type = ComposedModel
    models = 'isoharden'
  []
[]
"""
    hit_file = tmp_path / "nl_param.i"
    hit_file.write_text(hit_text)

    meta = export_model_for_aoti(hit_file, "chain", tmp_path)
    # Pure forward composition (no implicit segments) — one segment.
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "forward"
    assert {iv["name"] for iv in meta["inputs"]} == {"temperature", "equivalent_plastic_strain"}
    assert (tmp_path / "chain.pt2").exists()

    # Numerical parity with the eager Python-native model.
    eager = load_model(hit_file, "chain")
    pkg = load_package(tmp_path / "chain.pt2")
    in_order = list(eager.input_spec)
    test_vals = {
        "temperature": torch.tensor([200.0, 500.0, 800.0], dtype=torch.float64),
        "equivalent_plastic_strain": torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
    }
    args = [test_vals[k] for k in in_order]
    eager_out = eager(*args)
    if isinstance(eager_out, tuple):
        eager_out = eager_out[0]
    aoti_out = pkg(*args)
    if isinstance(aoti_out, tuple):
        aoti_out = aoti_out[0]
    assert torch.allclose(eager_out, aoti_out, rtol=1e-10, atol=1e-10)
