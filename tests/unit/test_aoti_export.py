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
from neml2.cli.aoti_export import AOTI_META_SCHEMA_VERSION

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


def test_resolve_derivative_specs_grammar_and_errors():
    """``-d/--derivative OUT:IN`` resolution: explicit pairs, the omission
    grammar (either/both sides empty = 'all'), the empty-specs default, the
    structural/parameter partition, and the error branches (no colon, unknown
    output, unknown input)."""
    from neml2.cli.aoti_export import _resolve_derivative_specs

    outs = ["stress", "energy"]
    ins = ["strain", "temperature"]
    params = ["E", "nu"]

    def struct(specs):  # structural pairs only (no param names declared)
        return _resolve_derivative_specs(specs, outs, ins)[0]

    # The function returns (structural_pairs, param_pairs).
    # No specs -> no pairs (derivative graphs are opt-in).
    assert _resolve_derivative_specs([], outs, ins) == (set(), set())

    # One explicit pair.
    assert struct(["stress:strain"]) == {("stress", "strain")}

    # Omit the output side -> every output w.r.t. that input.
    assert struct([":strain"]) == {("stress", "strain"), ("energy", "strain")}

    # Omit the input side -> that output w.r.t. every STRUCTURAL input.
    assert struct(["stress:"]) == {("stress", "strain"), ("stress", "temperature")}

    # Both sides omitted -> all structural pairs; whitespace tolerated.
    assert struct([":"]) == {(o, i) for o in outs for i in ins}
    assert struct([" stress : strain "]) == {("stress", "strain")}

    # Multiple specs union (and dedupe).
    assert struct(["stress:strain", "stress:strain", "energy:"]) == {
        ("stress", "strain"),
        ("energy", "strain"),
        ("energy", "temperature"),
    }

    # Parameter derivatives: an IN naming a promoted parameter resolves to a
    # param pair, kept separate from structural pairs.
    assert _resolve_derivative_specs(["stress:E"], outs, ins, param_names=params) == (
        set(),
        {("stress", "E")},
    )
    # An empty IN selects structural inputs only -- params are opt-in by name.
    assert _resolve_derivative_specs(["stress:"], outs, ins, param_names=params) == (
        {("stress", "strain"), ("stress", "temperature")},
        set(),
    )
    # Mixed specs partition into the two sets.
    assert _resolve_derivative_specs(
        ["stress:strain", "energy:nu"], outs, ins, param_names=params
    ) == ({("stress", "strain")}, {("energy", "nu")})

    # Missing colon.
    with pytest.raises(ValueError, match="expected OUT:IN"):
        _resolve_derivative_specs(["stress"], outs, ins)

    # Unknown output.
    with pytest.raises(ValueError, match="unknown output 'nope'"):
        _resolve_derivative_specs(["nope:strain"], outs, ins)

    # Unknown input: a name that is neither a structural input nor a declared
    # promoted parameter is rejected (here no param_names are declared, so 'E'
    # is unknown -- promote it with -p to request a parameter derivative).
    with pytest.raises(ValueError, match="unknown input 'E'"):
        _resolve_derivative_specs(["stress:E"], outs, ins)


# ---------------------------------------------------------------------------
# Artifact planning (plan_export_artifacts + the emit-order predictors).
# Cheap: no compile -- pure structural analysis. The drift-guard test in
# tests/aoti/test_compile_cli.py checks these predictions against what an actual
# compile produces.
# ---------------------------------------------------------------------------

_COMPOSED_I = _TESTS_ROOT / "aoti/composed_param/model.i"


def test_predict_forward_artifacts_emit_order():
    """The forward predictor mirrors, in emission order, the compile_model calls
    in _compile_forward_segment (+ the param-Jacobian / VJP helpers)."""
    from neml2.cli.aoti_export import _predict_forward_artifacts

    # Value only (no derivative pairs selected).
    assert _predict_forward_artifacts("m", set(), set(), single_forward_pvjp=True) == ["m.pt2"]
    # Value + jvp when input pairs are selected.
    assert _predict_forward_artifacts("m", {("y", "x")}, set(), single_forward_pvjp=True) == [
        "m.pt2",
        "m_jvp.pt2",
    ]
    # ``None`` selection means "all pairs" -> jvp emitted.
    assert _predict_forward_artifacts("m", None, set(), single_forward_pvjp=False) == [
        "m.pt2",
        "m_jvp.pt2",
    ]
    # The pvjp graph is emitted ONLY for the lone forward segment of a
    # forward-only model (single_forward_pvjp=True).
    assert _predict_forward_artifacts(
        "m", {("y", "x")}, {("y", "p")}, single_forward_pvjp=True
    ) == ["m.pt2", "m_jvp.pt2", "m_pjac.pt2", "m_pvjp.pt2"]
    # A composed forward segment with the same selection emits pjac but NOT pvjp.
    assert _predict_forward_artifacts(
        "m_seg0", {("y", "x")}, {("y", "p")}, single_forward_pvjp=False
    ) == ["m_seg0.pt2", "m_seg0_jvp.pt2", "m_seg0_pjac.pt2"]


def test_predict_implicit_artifacts_emit_order():
    """The implicit predictor mirrors the compile_model calls in
    _compile_implicit_segment: residual, jacobian, solve, then optional
    ift/pift/predictor."""
    from neml2.cli.aoti_export import _predict_implicit_artifacts

    assert _predict_implicit_artifacts(
        "m", iterative=False, precond_on=False, emit_ift=False, emit_pift=False, has_predictor=False
    ) == ["m_residual.pt2", "m_jacobian.pt2", "m_solve.pt2"]
    assert _predict_implicit_artifacts(
        "m", iterative=False, precond_on=False, emit_ift=True, emit_pift=True, has_predictor=True
    ) == [
        "m_residual.pt2",
        "m_jacobian.pt2",
        "m_solve.pt2",
        "m_jacobian_given.pt2",
        "m_solve_ift.pt2",
        "m_dr_dparam.pt2",
        "m_solve_param.pt2",
        "m_predictor.pt2",
    ]


def test_predict_implicit_artifacts_krylov():
    """A matrix-free Krylov solve emits matvec instead of solve; a preconditioner
    adds its authored precond_setup/precond_apply graphs (not the standalone
    jacobian A -- the setup assembles what it needs itself); an input derivative
    still needs the assembled A (jacobian) for the direct IFT solve."""
    from neml2.cli.aoti_export import _predict_implicit_artifacts

    # Pure matrix-free (no preconditioner, no derivatives): residual + matvec only.
    assert _predict_implicit_artifacts(
        "m", iterative=True, precond_on=False, emit_ift=False, emit_pift=False, has_predictor=False
    ) == ["m_residual.pt2", "m_matvec.pt2"]
    # A preconditioner authors its own setup/apply graphs (no standalone jacobian).
    assert _predict_implicit_artifacts(
        "m", iterative=True, precond_on=True, emit_ift=False, emit_pift=False, has_predictor=False
    ) == ["m_residual.pt2", "m_matvec.pt2", "m_precond_setup.pt2", "m_precond_apply.pt2"]
    # An input derivative reuses A (jacobian); a DIRECT input-sensitivity solver
    # keeps the baked IFT solve.
    assert _predict_implicit_artifacts(
        "m", iterative=True, precond_on=False, emit_ift=True, emit_pift=False, has_predictor=False
    ) == [
        "m_residual.pt2",
        "m_jacobian.pt2",
        "m_matvec.pt2",
        "m_jacobian_given.pt2",
        "m_solve_ift.pt2",
    ]
    # An ITERATIVE input/param sensitivity solver omits its solve_ift / solve_param
    # graph (the C++ Krylov-solves over the assembled A); the operators stay.
    assert _predict_implicit_artifacts(
        "m",
        iterative=False,
        precond_on=False,
        emit_ift=True,
        emit_pift=True,
        has_predictor=False,
        input_sensitivity_iterative=True,
        param_sensitivity_iterative=True,
    ) == [
        "m_residual.pt2",
        "m_jacobian.pt2",
        "m_solve.pt2",
        "m_jacobian_given.pt2",
        "m_dr_dparam.pt2",
    ]


def test_plan_export_artifacts_forward():
    """A forward-only model plans one forward segment named after the model,
    with the meta.json last."""
    from neml2.cli.aoti_export import plan_export_artifacts

    plan = plan_export_artifacts(_ELASTICITY_I, "model")
    assert [s.kind for s in plan.segments] == ["forward"]
    assert plan.artifacts == ["model.pt2", "metadata.json"]

    plan_d = plan_export_artifacts(_ELASTICITY_I, "model", derivatives=[":"])
    assert plan_d.artifacts == ["model.pt2", "model_jvp.pt2", "metadata.json"]
    assert plan_d.total == 3


def test_plan_export_artifacts_single_implicit():
    """A single ImplicitUpdate plans one implicit segment (rhs + step, + ift when
    derivatives are requested), meta.json last."""
    from neml2.cli.aoti_export import plan_export_artifacts

    # This J2 return map carries a Predictor, so the plan includes its graph too
    # (rhs, step, ift, predictor) -- exercising the has_predictor branch.
    plan = plan_export_artifacts(_J2_NATIVE_I, "return_map", derivatives=[":"])
    assert [s.kind for s in plan.segments] == ["implicit"]
    assert plan.artifacts == [
        "return_map_residual.pt2",
        "return_map_jacobian.pt2",
        "return_map_solve.pt2",
        "return_map_jacobian_given.pt2",
        "return_map_solve_ift.pt2",
        "return_map_predictor.pt2",
        "metadata.json",
    ]


def test_plan_export_artifacts_composed_multi_segment():
    """A ComposedModel with an ImplicitUpdate plans one segment per partition,
    named ``<model>_seg{i}``, with the meta.json as the final entry."""
    from neml2.cli.aoti_export import plan_export_artifacts

    plan = plan_export_artifacts(_COMPOSED_I, "model", derivatives=[":"])
    assert [s.kind for s in plan.segments] == ["forward", "implicit", "forward"]
    assert plan.artifacts == [
        "model_seg0.pt2",
        "model_seg0_jvp.pt2",
        "model_seg1_residual.pt2",
        "model_seg1_jacobian.pt2",
        "model_seg1_solve.pt2",
        "model_seg1_jacobian_given.pt2",
        "model_seg1_solve_ift.pt2",
        "model_seg2.pt2",
        "model_seg2_jvp.pt2",
        "metadata.json",
    ]
    # The metadata.json is always the last generated file.
    assert plan.artifacts[-1] == "metadata.json"


@pytest.fixture(scope="session")
def forward_export(tmp_path_factory):
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_forward")
    # Derivative graphs are opt-in; request all pairs so the
    # jvp/jacobian assertions below have a graph to exercise.
    meta = export_model_for_aoti(_ELASTICITY_I, "model", out_dir, derivatives=[":"])
    # Schema v10: out_dir is the artifact ROOT. Binaries live under
    # <device>/<dtype>/; the shared metadata.json sits at the root.
    return meta, out_dir, out_dir / "cpu" / "float64"


@pytest.fixture(scope="session")
def forward_export_no_deriv(tmp_path_factory):
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_forward_no_deriv")
    meta = export_model_for_aoti(_ELASTICITY_I, "model", out_dir)
    return meta, out_dir, out_dir / "cpu" / "float64"


@pytest.fixture(scope="session")
def param_jac_export(tmp_path_factory):
    from neml2.cli.aoti_export import _reverse_ad_aoti_unsupported_reason  # noqa: PLC0415

    reason = _reverse_ad_aoti_unsupported_reason()
    if reason is not None:
        pytest.skip(f"reverse-mode AD AOTI compilation {reason}")
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_param_jac")
    # Promote E to a runtime input and request the parameter derivative
    # d(stress)/d(E) -- the schema-v7 parameter-Jacobian path. A bare leaf is
    # wrapped in ComposedModel([leaf]) at export, so its qualified name is "model.E".
    meta = export_model_for_aoti(
        _ELASTICITY_I, "model", out_dir, promoted={"model.E"}, derivatives=["stress:model.E"]
    )
    return meta, out_dir, out_dir / "cpu" / "float64"


@pytest.fixture(scope="session")
def implicit_export(tmp_path_factory):
    from neml2.cli.aoti_export import export_model_for_aoti

    out_dir = tmp_path_factory.mktemp("aoti_implicit")
    meta = export_model_for_aoti(_J2_NATIVE_I, "return_map", out_dir, derivatives=[":"])
    return meta, out_dir, out_dir / "cpu" / "float64"


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def test_export_forward_model_produces_artifacts(forward_export):
    meta, out_dir, leaf = forward_export

    # Phase-7 unified schema: every export is "composed". A pure forward model
    # is a one-segment composition with kind=forward.
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "forward"
    # Schema v10: the .pt2 binary lives under <device>/<dtype>/; the single
    # shared metadata.json sits at the artifact root. No device/dtype is
    # recorded in the meta -- both are derived from the folder path at load.
    assert (leaf / "model.pt2").exists()
    assert (out_dir / "metadata.json").exists()
    assert "device" not in meta
    assert "dtype" not in meta
    # A forward-only model has no implicit solve, so no solver config is recorded
    # (the implicit-model test asserts the symmetric presence).
    assert "solver_config" not in meta


def test_export_forward_model_metadata(forward_export):
    _, out_dir, leaf = forward_export
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["type"] == "composed"
    assert meta["inputs"] == [
        {"name": "strain", "var_size": 6, "var_type": "SR2", "base_shape": [6]}
    ]
    assert meta["outputs"] == [
        {"name": "stress", "var_size": 6, "var_type": "SR2", "base_shape": [6]}
    ]
    # Schema v7: top-level `derivatives` lists the requested master pairs;
    # `parameter_derivatives` is empty (no -d over a promoted parameter here).
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION
    assert meta["derivatives"] == [["stress", "strain"]]
    assert meta["parameter_derivatives"] == []
    seg = meta["segments"][0]
    assert seg["kind"] == "forward"
    # Forward-segment per-variable metadata is name-only (v4 schema). The
    # master-level dicts above still carry var_size/var_type for the
    # Python shim's input_spec/output_spec reconstruction.
    assert seg["inputs"] == [{"name": "strain"}]
    assert seg["outputs"] == [{"name": "stress"}]
    # Forward segments carry a `jvp_package` next to the value package.
    assert seg["jvp_package"] == "model_jvp.pt2"
    assert (leaf / "model_jvp.pt2").exists()
    # Each pair carries a batch_independent flag; elasticity's Jacobian is the
    # constant stiffness tensor -> batch-independent.
    pair = seg["jacobian_pairs"][0]
    assert (pair["out_var"], pair["in_var"]) == ("stress", "strain")
    assert pair["batch_independent"] is True


def test_export_forward_default_off_no_derivatives(forward_export_no_deriv):
    """With no -d flags, NO derivative graph is compiled: empty top-level
    `derivatives`, no segment `jvp_package`, no `*_jvp.pt2` on disk."""
    meta, out_dir, leaf = forward_export_no_deriv
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION
    assert meta["derivatives"] == []
    assert meta["parameter_derivatives"] == []
    seg = meta["segments"][0]
    assert "jvp_package" not in seg
    assert "jacobian_pairs" not in seg
    assert "param_jacobian_package" not in seg
    assert not (leaf / "model_jvp.pt2").exists()


def test_export_forward_param_jacobian_metadata(param_jac_export):
    """Schema v7: a requested d(out)/d(param) is recorded at top level
    (`parameter_derivatives`) and on the segment (`param_jacobian_package` +
    `param_jacobian_pairs`), with the promoted parameter listed under
    `parameters`."""
    meta, out_dir, leaf = param_jac_export
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION
    assert meta["parameter_derivatives"] == [["stress", "model.E"]]
    # No structural derivative was requested, so the input-jvp graph is absent.
    assert meta["derivatives"] == []
    e_param = next(p for p in meta["parameters"] if p["name"] == "model.E")
    # Schema v10: a parameter entry is device- AND dtype-independent so a single
    # shared metadata.json backs every <device>/<dtype>/ artifact. No `device`
    # is recorded; a floating-point parameter (the usual case) also omits
    # `dtype` -- it inherits the leaf dtype at load. The entry still carries the
    # dtype-neutral (float64) `values` and its shape metadata.
    assert "device" not in e_param
    assert "dtype" not in e_param
    assert set(e_param) >= {"name", "shape", "param_base_shape", "values", "origin"}
    seg = meta["segments"][0]
    assert "jvp_package" not in seg
    assert seg["param_jacobian_package"] == "model_pjac.pt2"
    assert (leaf / "model_pjac.pt2").exists()
    pair = seg["param_jacobian_pairs"][0]
    assert (pair["out_var"], pair["param"]) == ("stress", "model.E")
    assert pair["out_base_shape"] == [6]
    assert pair["param_base_shape"] == []  # scalar parameter -> no trailing axis


def test_export_forward_param_jacobian_matches_finite_difference(param_jac_export):
    """The compiled parameter-Jacobian graph's d(stress)/d(E) block agrees with
    central differences taken on the value package. The parameter enters the
    derivative graph as a PER-BATCH input (the C++ runtime broadcasts the stored
    scalar parameter to the runtime batch before calling it)."""
    from neml2.models.export import load_package
    from neml2.types import SR2

    meta, out_dir, leaf = param_jac_export
    seg = meta["segments"][0]
    pjac = load_package(str(leaf / seg["param_jacobian_package"]))
    value = load_package(str(leaf / seg["package"]))

    torch.manual_seed(0)
    b = 5
    strain = torch.randn(b, 6, dtype=torch.float64)

    def stress(e_val):
        # The value graph now takes the promoted parameter as a PER-BATCH input
        # so feed it at the batch shape (the runtime broadcasts a
        # stored scalar; here we pass the value at (b,) directly).
        res = value(SR2(strain), torch.full((b,), float(e_val), dtype=torch.float64))[0]
        return res.data if hasattr(res, "data") else res

    for e_val in (100.0, 250.0):
        out = pjac(SR2(strain), torch.full((b,), float(e_val), dtype=torch.float64))
        block = out[0] if isinstance(out, tuple) else out  # (b, 6)
        assert tuple(block.shape) == (b, 6)
        h = 1e-4 * e_val
        fd = (stress(e_val + h) - stress(e_val - h)) / (2 * h)
        rel = (block - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
        assert rel < 1e-5, f"E={e_val}: d(stress)/dE disagrees with FD (rel={rel:.2e})"


def test_export_forward_param_vjp_metadata(param_jac_export):
    """Requesting a parameter derivative also emits the adjoint (VJP) graph:
    `param_vjp_package` + the parameter (grad-output) and output (cotangent-input)
    orderings."""
    meta, out_dir, leaf = param_jac_export
    seg = meta["segments"][0]
    assert seg["param_vjp_package"] == "model_pvjp.pt2"
    assert (leaf / "model_pvjp.pt2").exists()
    assert seg["param_vjp_params"] == ["model.E"]  # grad-output order
    assert seg["param_vjp_outputs"] == ["stress"]  # cotangent-input order


def test_export_forward_param_vjp_matches_finite_difference(param_jac_export):
    """The adjoint graph's dL/dE (L = <w, stress>) agrees with central differences
    of the scalar loss on the value package. The promoted parameter enters the
    adjoint graph PER-BATCH (like the param-Jacobian graph), so the graph
    returns one gradient per batch element; for a global (uniform) E those
    per-element gradients sum to the scalar dL/dE -- which is what the C++ runtime
    returns for an unbatched parameter."""
    from neml2.models.export import load_package
    from neml2.types import SR2

    meta, out_dir, leaf = param_jac_export
    seg = meta["segments"][0]
    pvjp = load_package(str(leaf / seg["param_vjp_package"]))
    value = load_package(str(leaf / seg["package"]))

    torch.manual_seed(1)
    b = 4
    strain = torch.randn(b, 6, dtype=torch.float64)
    w = torch.randn(b, 6, dtype=torch.float64)  # output cotangent

    def loss(e_val):
        # Value graph takes the promoted parameter per-batch.
        res = value(SR2(strain), torch.full((b,), float(e_val), dtype=torch.float64))[0]
        s = res.data if hasattr(res, "data") else res
        return (s * w).sum().item()

    for e_val in (100.0, 250.0):
        # The parameter is per-batch and the cotangent is a typed input (matching
        # the output's wrapper class); the graph returns a per-element (b,) adjoint.
        out = pvjp(
            SR2(strain),
            torch.full((b,), float(e_val), dtype=torch.float64),
            SR2(w),
        )
        g = out[0] if isinstance(out, tuple) else out  # per-element (b,) dL/dE_i
        assert tuple(g.shape) == (b,)
        h = 1e-4 * e_val
        fd = (loss(e_val + h) - loss(e_val - h)) / (2 * h)
        # The per-element adjoints sum to the global dL/dE (the scalar-loss FD).
        rel = abs(float(g.sum()) - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-5, f"E={e_val}: dL/dE adjoint disagrees with FD (rel={rel:.2e})"


def test_export_forward_jvp_matches_finite_difference(forward_export):
    """The JVP package's flat Jacobian must agree with central differences on
    the value package."""
    from neml2.models.export import load_package
    from neml2.types import TensorWrapper

    def _as_tensor(x):
        # load_package re-wraps outputs as typed wrappers (SR2 / Scalar / ...);
        # torch.allclose only accepts raw tensors, so unwrap here.
        return x.data if isinstance(x, TensorWrapper) else x

    _, out_dir, leaf = forward_export

    val_pkg = load_package(leaf / "model.pt2")
    jvp_pkg = load_package(leaf / "model_jvp.pt2")

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
    meta, out_dir, leaf = implicit_export

    # Phase-7 unified schema: a single ImplicitUpdate is a one-segment
    # composition with kind=implicit. Artifact filenames keep their original
    # residual/jacobian/solve/predictor suffixes (no `_seg0_` prefix for
    # single-segment).
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "implicit"
    # Schema v10: operator + solve graphs, binaries under <device>/<dtype>/,
    # shared metadata.json at root.
    assert (leaf / "return_map_residual.pt2").exists()
    assert (leaf / "return_map_jacobian.pt2").exists()
    assert (leaf / "return_map_solve.pt2").exists()
    assert (leaf / "return_map_jacobian_given.pt2").exists()
    assert (leaf / "return_map_solve_ift.pt2").exists()
    assert (out_dir / "metadata.json").exists()
    assert meta["segments"][0]["solve_ift_package"] == "return_map_solve_ift.pt2"
    # An implicit model records the runtime solver config in the shared meta so
    # the Python-free C++ runtime is configured straight from the artifact.
    assert "solver_config" in meta
    assert set(meta["solver_config"]) == {
        "atol",
        "rtol",
        "miters",
        "ls_type",
        "ls_max_iters",
        "ls_cutback",
        "ls_c",
        # schema v11: the linear-solver kind (this fixture uses the default direct
        # DenseLU, so no nested `krylov` block).
        "solver_kind",
    }
    assert meta["solver_config"]["solver_kind"] == "direct"
    assert "krylov" not in meta["solver_config"]
    assert "verbose" not in meta["solver_config"]


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
    _, out_dir, leaf = implicit_export
    # Schema v10: the IFT is un-baked into operator + solve graphs. Chain
    # `jacobian` (A = ∂r/∂u + b) + `jacobian_given` (B = ∂r/∂g) -> `solve_ift`
    # (A^{-1}B -> per-pair -du/dg blocks), exactly as the C++ runtime does.
    jacobian_pkg = load_package(leaf / "return_map_jacobian.pt2")
    jacobian_given_pkg = load_package(leaf / "return_map_jacobian_given.pt2")
    solve_ift_pkg = load_package(leaf / "return_map_solve_ift.pt2")

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

    # The solve_ift graph emits one block per (unknown, given) pair (the
    # disassemble of -du/dg), in unknown x given order. Reassemble them into the
    # full -du/dg matrix and check the IFT identity A J + B = 0.
    jac_out = jacobian_pkg(u, g)
    if not isinstance(jac_out, tuple):
        jac_out = (jac_out,)
    n_a = system.blayout.ngroup * system.ulayout.ngroup
    a_blocks = jac_out[:n_a]  # drop the trailing b groups
    b_blocks = jacobian_given_pkg(u, g)
    if not isinstance(b_blocks, tuple):
        b_blocks = (b_blocks,)
    blocks = solve_ift_pkg(*a_blocks, *b_blocks)
    if not isinstance(blocks, tuple):
        blocks = (blocks,)
    u_off = {
        n: sum(system.ulayout.var_size(m) for m in system.unknown_names[:i])
        for i, n in enumerate(system.unknown_names)
    }
    g_off = {
        n: sum(system.glayout.var_size(m) for m in system.given_names[:i])
        for i, n in enumerate(system.given_names)
    }
    J = torch.zeros(
        u.shape[0],
        system.ulayout.storage_size(),
        system.glayout.storage_size(),
        dtype=torch.float64,
    )
    k = 0
    for un in system.unknown_names:
        rs = system.ulayout.var_size(un)
        for gn in system.given_names:
            cs = system.glayout.var_size(gn)
            J[:, u_off[un] : u_off[un] + rs, g_off[gn] : g_off[gn] + cs] = blocks[k]
            k += 1
    residual = A @ J + B
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)


def test_export_implicit_model_metadata(implicit_export):
    _, out_dir, _ = implicit_export
    meta = json.loads((out_dir / "metadata.json").read_text())
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

    # Solver convergence / line-search config is not stored PER SEGMENT; it lives
    # at the top level (meta["solver_config"]), read by the C++ runtime at load.
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

    _, out_dir, leaf = implicit_export

    rhs_pkg = load_package(leaf / "return_map_residual.pt2")

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

    meta, out_dir, leaf = implicit_export
    seg = meta["segments"][0]

    assert "predictor_package" in seg
    assert (leaf / seg["predictor_package"]).exists()

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
# promoted parameter (mode 3) flows through AOTI export
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
    hit_file = tmp_path / "promoted_param.i"
    hit_file.write_text(hit_text)

    meta = export_model_for_aoti(hit_file, "chain", tmp_path)
    # Pure forward composition (no implicit segments) — one segment.
    assert meta["type"] == "composed"
    assert len(meta["segments"]) == 1
    assert meta["segments"][0]["kind"] == "forward"
    assert {iv["name"] for iv in meta["inputs"]} == {"temperature", "equivalent_plastic_strain"}
    # Schema v10 layout: binary under <device>/<dtype>/, shared meta at the root.
    leaf = tmp_path / "cpu" / "float64"
    assert (leaf / "chain.pt2").exists()
    assert (tmp_path / "metadata.json").exists()

    # Numerical parity with the eager Python-native model.
    from neml2.types import TensorWrapper

    def _as_tensor(x):
        return x.data if isinstance(x, TensorWrapper) else x

    eager = load_model(hit_file, "chain")
    pkg = load_package(leaf / "chain.pt2")
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


# ---------------------------------------------------------------------------
# metadata.json atomic write (_write_meta)
# ---------------------------------------------------------------------------


def test_write_meta_roundtrip_no_temp_left(tmp_path):
    """Happy path: metadata.json holds the exact dict and no *.tmp file lingers."""
    from neml2.cli.aoti_export import _write_meta

    p = tmp_path / "metadata.json"
    meta = {"schema_version": "9", "inputs": [{"name": "x"}], "nested": {"a": 1}}
    _write_meta(p, meta)
    assert json.loads(p.read_text()) == meta
    # The temp file is renamed onto the target, never left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_meta_atomic_overwrite(tmp_path):
    """A second write replaces the file wholesale (atomic overwrite)."""
    from neml2.cli.aoti_export import _write_meta

    p = tmp_path / "metadata.json"
    _write_meta(p, {"v": 1})
    _write_meta(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_meta_cleans_temp_on_failure(tmp_path, monkeypatch):
    """If the atomic replace fails, the exception propagates AND the temp file is
    removed (no partial/stray metadata left in the artifact dir)."""
    import os

    from neml2.cli.aoti_export import _write_meta

    def _boom(*_a, **_k):
        raise OSError("simulated replace failure")

    # _write_meta calls os.replace via the shared os module; patch it there.
    monkeypatch.setattr(os, "replace", _boom)
    p = tmp_path / "metadata.json"
    with pytest.raises(OSError, match="simulated replace failure"):
        _write_meta(p, {"v": 1})
    # Target never appeared, and the temp was cleaned up on the failure path.
    assert not p.exists()
    assert list(tmp_path.glob("*")) == []


# ---------------------------------------------------------------------------
# spawn-worker extension re-import (_prepare_opts_in_worker)
# ---------------------------------------------------------------------------


def test_prepare_opts_in_worker_imports_load(tmp_path):
    """A spawn worker re-imports the ``--load`` extensions and strips ``load``
    from the kwargs it forwards to ``_prepare_export``."""
    from neml2.cli.aoti_export import _prepare_opts_in_worker

    # An extension whose top-level code has an observable side effect on import.
    sentinel = tmp_path / "loaded.flag"
    ext = tmp_path / "ext_mod.py"
    ext.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('loaded')\n")
    opts = _prepare_opts_in_worker({"load": (str(ext),), "promoted": ["a"]})
    assert sentinel.exists()  # the extension module was executed
    assert opts == {"promoted": ["a"]}  # 'load' consumed, rest forwarded verbatim


def test_prepare_opts_in_worker_no_load_is_identity_copy():
    """With no ``load`` key the kwargs pass through unchanged, as a fresh copy."""
    from neml2.cli.aoti_export import _prepare_opts_in_worker

    src = {"promoted": ["a"], "renames": None, "derivatives": ()}
    opts = _prepare_opts_in_worker(src)
    assert opts == src
    assert opts is not src  # a copy -- never mutate the caller's dict
