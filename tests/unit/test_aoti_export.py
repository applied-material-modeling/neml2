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

_TESTS_UNIT = Path(__file__).parent
_TESTS_ROOT = _TESTS_UNIT.parent
_ELASTICITY_I = _TESTS_ROOT / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
_J2_NATIVE_I = _TESTS_UNIT / "fixtures/j2_linear_isoharden/model.i"


def test_parse_example_batch_shape_no_sub_batch():
    from neml2.cli.aoti_export import _parse_example_batch_shape

    # No sub-batch axes: dyn-only specs return an empty sub tuple.
    assert _parse_example_batch_shape("(2,)") == ((2,), ())
    assert _parse_example_batch_shape("(2, 3)") == ((2, 3), ())


def test_parse_example_batch_shape_with_sub_batch():
    from neml2.cli.aoti_export import _parse_example_batch_shape

    # Semicolon splits dyn from sub.
    assert _parse_example_batch_shape("(2; 3)") == ((2,), (3,))
    assert _parse_example_batch_shape("(2; 3, 12)") == ((2,), (3, 12))
    # Empty dyn region is allowed (static-batch + sub).
    assert _parse_example_batch_shape("(; 100)") == ((), (100,))


def test_parse_example_batch_shape_rejects_label_suffix():
    """The ``:label`` suffix on sub-batch extents was removed in V2P-9
    (the chain rule no longer dispatches on labels). A leftover ``:foo``
    must be flagged with a clear error rather than silently parsed."""
    from neml2.cli.aoti_export import _parse_example_batch_shape

    with pytest.raises(ValueError, match=":label.*removed"):
        _parse_example_batch_shape("(2; 3:grain)")


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
    assert meta["inputs"] == [
        {"name": "strain", "var_size": 6, "var_type": "SR2", "base_shape": [6]}
    ]
    assert meta["outputs"] == [
        {"name": "stress", "var_size": 6, "var_type": "SR2", "base_shape": [6]}
    ]
    seg = meta["segments"][0]
    assert seg["kind"] == "forward"
    # Forward-segment per-variable metadata is name-only (v4 schema). The
    # master-level dicts above still carry var_size/var_type for the
    # Python shim's input_spec/output_spec reconstruction.
    assert seg["inputs"] == [{"name": "strain"}]
    assert seg["outputs"] == [{"name": "stress"}]
    # Forward segments carry a `jvp_package` next to the value package.
    assert seg["jvp_package"] == "model_jvp.pt2"
    assert (out_dir / "model_jvp.pt2").exists()


def test_export_forward_jvp_matches_finite_difference(forward_export):
    """The JVP package's flat Jacobian must agree with central differences on
    the value package."""
    from neml2.models.export import load_package
    from neml2.types import TensorWrapper

    def _as_tensor(x):
        # load_package re-wraps outputs as typed wrappers (SR2 / Scalar / ...);
        # torch.allclose only accepts raw tensors, so unwrap here.
        return x.data if isinstance(x, TensorWrapper) else x

    _, out_dir = forward_export

    val_pkg = load_package(out_dir / "model.pt2")
    jvp_pkg = load_package(out_dir / "model_jvp.pt2")

    gen = torch.Generator().manual_seed(0)
    strain = torch.randn(4, 6, generator=gen, dtype=torch.float64) * 1e-3

    *_, J = jvp_pkg(strain)
    J = _as_tensor(J)
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
        fd = (_as_tensor(s1) - _as_tensor(s0)) / eps
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
    random (u, g) without requiring a converged state -- the IFT identity is
    purely local."""
    from neml2.es import ModelNonlinearSystem
    from neml2.factory import load_input
    from neml2.models.export import load_package

    f = load_input(_J2_NATIVE_I)
    return_map = f.get_model("return_map")
    _, out_dir = implicit_export
    ift_pkg = load_package(out_dir / "return_map_ift.pt2")

    system: ModelNonlinearSystem = return_map.system  # type: ignore[attr-defined]

    gen = torch.Generator().manual_seed(0)
    u = torch.randn(2, system.ulayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-3
    g = torch.randn(2, system.glayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-2

    # Initialize the system at (u, g) so the eager A_and_B() reflects the
    # same state the compiled IFT graph evaluates at. Wrap each raw slice
    # at the construction boundary -- SparseVector is strictly typed
    # (rule 1), so the boundary is here, not deep in the system helper.
    def _wrap(raw, type_cls):
        slab = raw if type_cls.BASE_NDIM > 0 else raw.squeeze(-1)
        return type_cls(slab)

    u_dict = {
        name: _wrap(
            u[..., off : off + system.ulayout.var_size(name)],
            system.ulayout.type_of(name),
        )
        for name, off in zip(
            system.unknown_names,
            [
                sum(system.ulayout.var_size(n) for n in system.unknown_names[:i])
                for i in range(len(system.unknown_names))
            ],
            strict=True,
        )
    }
    g_dict = {
        name: _wrap(
            g[..., off : off + system.glayout.var_size(name)],
            system.glayout.type_of(name),
        )
        for name, off in zip(
            system.given_names,
            [
                sum(system.glayout.var_size(n) for n in system.given_names[:i])
                for i in range(len(system.given_names))
            ],
            strict=True,
        )
    }
    from neml2.es import SparseVector  # noqa: PLC0415

    system.initialize(
        u=SparseVector(system.ulayout, u_dict),
        g=SparseVector(system.glayout, g_dict),
        dyn_shape=u.shape[:1],
    )
    A_assembled, B_assembled = system.A_and_B()
    A = A_assembled.tensors[0][0].data
    B = B_assembled.tensors[0][0].data

    J = ift_pkg(u, g)
    if isinstance(J, tuple):
        J = J[0]
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
    # The aggregated unknown surface (sum of unknown var_sizes) is 6+1+1=8;
    # ``u_size`` was the v6-and-earlier scalar; v7 keeps the per-variable
    # list and the per-group infos rather than the redundant sum.
    assert sum(unknown_sizes) == 8

    # The AOTI residual-surface givens (NOT the predictor's inputs) pack
    # only ``total_strain``(6), ``plastic_strain~1``(6),
    # ``equivalent_plastic_strain~1``(1), ``t``(1), ``t~1``(1) -> 15.
    given_sizes = [g["var_size"] for g in seg["givens"]]
    assert sum(given_sizes) == 15

    # Schema v4+: solver convergence / line-search configuration is no longer
    # baked into the metadata -- it is carried by the stub's [Solvers] block and
    # forwarded to the C++ runtime at load via Model.set_solver_config.
    assert "atol" not in seg
    assert "rtol" not in seg
    assert "miters" not in seg
    assert "linesearch" not in seg


def test_export_implicit_rhs_output_matches_eager(implicit_export):
    """Compiled rhs.pt2 must match Python ``RHS`` at a random state."""
    from neml2.es import RHS, ModelNonlinearSystem
    from neml2.factory import load_input
    from neml2.models.export import load_package

    # Register all native types and load the implicit model
    f = load_input(_J2_NATIVE_I)
    return_map = f.get_model("return_map")

    _, out_dir = implicit_export

    rhs_pkg = load_package(out_dir / "return_map_rhs.pt2")

    system: ModelNonlinearSystem = return_map.system  # type: ignore[attr-defined]

    gen = torch.Generator().manual_seed(42)
    u = torch.randn(4, system.ulayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-3
    g = torch.randn(4, system.glayout.storage_size(), generator=gen, dtype=torch.float64) * 1e-2

    # ``RHS.__init__`` snapshots ``system._dynamic_batch_ndim`` -- empty
    # before ``initialize()`` populates it -- so build the per-variable
    # u / g dicts and initialize the system first. Wrap raw slices at
    # construction (SparseVector is strictly typed per rule 1).
    def _wrap(raw, type_cls):
        slab = raw if type_cls.BASE_NDIM > 0 else raw.squeeze(-1)
        return type_cls(slab)

    u_dict = {
        name: _wrap(
            u[..., off : off + system.ulayout.var_size(name)],
            system.ulayout.type_of(name),
        )
        for name, off in zip(
            system.unknown_names,
            [
                sum(system.ulayout.var_size(n) for n in system.unknown_names[:i])
                for i in range(len(system.unknown_names))
            ],
            strict=True,
        )
    }
    g_dict = {
        name: _wrap(
            g[..., off : off + system.glayout.var_size(name)],
            system.glayout.type_of(name),
        )
        for name, off in zip(
            system.given_names,
            [
                sum(system.glayout.var_size(n) for n in system.given_names[:i])
                for i in range(len(system.given_names))
            ],
            strict=True,
        )
    }
    from neml2.es import SparseVector  # noqa: PLC0415

    system.initialize(
        u=SparseVector(system.ulayout, u_dict),
        g=SparseVector(system.glayout, g_dict),
        dyn_shape=u.shape[:1],
    )
    rhs = RHS(system)

    # v7 contract: RHS takes per-group raws and returns per-group raws.
    # j2_linear_isoharden has a single (DENSE) unknown group + a single
    # given group, so the per-group tensors collapse to the flat shapes.
    eager = rhs(u, g)
    compiled = rhs_pkg(u, g)
    if not isinstance(eager, tuple):
        eager = (eager,)
    if not isinstance(compiled, tuple):
        compiled = (compiled,)

    assert len(eager) == len(compiled)
    for e, c in zip(eager, compiled, strict=True):
        assert torch.allclose(e, c, rtol=1e-10, atol=1e-10)


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
    """Asking AOTI to export an unregistered type must fail loudly at the
    factory boundary, before any compile work happens. Uses an obviously
    fictional type name so the test stays meaningful as the native catalog
    expands (we picked a real-but-then-ported v2 type before — it became a
    pass-through the moment that type was ported).
    """
    from neml2.cli.aoti_export import export_model_for_aoti

    hit_text = """
[Models]
  [m]
    type = NotARealNeml2Type
  []
[]
"""
    hit_file = tmp_path / "unregistered.i"
    hit_file.write_text(hit_text)

    with pytest.raises(KeyError, match="NotARealNeml2Type"):
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
    from neml2.factory import load_model
    from neml2.models.export import load_package

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
    from neml2.types import TensorWrapper

    def _as_tensor(x):
        return x.data if isinstance(x, TensorWrapper) else x

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
    assert torch.allclose(_as_tensor(eager_out), _as_tensor(aoti_out), rtol=1e-10, atol=1e-10)
