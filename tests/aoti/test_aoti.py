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

"""Parametrized AOTI export + reload tests.

Walks every ``<scenario>/model.i`` under this directory, compiles the model
named ``model`` (fully baked -- no promotions) via
:func:`~neml2.cli.aoti_export.export_model_for_aoti`, reloads the artifact
through :class:`~neml2.aoti.Model`, and asserts the compiled forward output
matches the eager native run within ``rtol/atol``. Promoted-parameter behavior
is exercised separately by the dedicated parameter-derivative tests below
(``test_aoti_param_jacobian_and_vjp_match_fd`` and friends), which pass
``promoted=`` explicitly.

Runs by default. Each scenario triggers an Inductor compile, so the
full sweep is slow -- the CI ``test`` job allocates several minutes
to this directory alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from neml2.cli.aoti_export import _reverse_ad_aoti_unsupported_reason

_SCENARIO_DIR = Path(__file__).parent

# Best-effort AOTI on Windows: torch's ``AOTIModelPackageLoader`` re-extracts the
# ``.pt2`` to a temp dir keyed by the artifact's content hash on every load, and
# a ``.pyd`` already loaded elsewhere in the process holds a file lock. Loading
# the *same* artifact twice in one process therefore fails to overwrite the
# locked ``.pyd`` (``WinError 32``). Tests that intentionally load one artifact
# twice (e.g. pybind-vs-shim parity) are skipped on Windows for this reason.
_skip_win_reextract = pytest.mark.skipif(
    sys.platform == "win32",
    reason="AOTI re-extraction locks a loaded .pyd on Windows (WinError 32); "
    "same-artifact double-load unsupported (best-effort).",
)

# Reverse-mode-AD AOTI graphs (parameter derivatives) can't be lowered on every
# torch: < 2.11 lacks trace_autograd_ops, and 2.11.x rejects requires_grad_()
# under strict export. Skip those cases on the exact predicate the exporter guards
# on, so the test gate and the runtime guard can't drift. Forward / jvp / jacobian
# are unaffected and still run on every torch.
_PARAM_DERIV_UNSUPPORTED = _reverse_ad_aoti_unsupported_reason()
_REQUIRES_PARAM_DERIV_TORCH = pytest.mark.skipif(
    _PARAM_DERIV_UNSUPPORTED is not None,
    reason=f"reverse-mode AD AOTI compilation {_PARAM_DERIV_UNSUPPORTED}",
)
# Scenarios with a dedicated test that need domain-specific inputs the generic
# randn-driven value/stub tests here cannot supply. ``request_ad_forward`` is a
# physically-sensitive ML surrogate (Arrhenius exp(-Q/RT) overflows for the
# negative temperatures randn produces); it is covered by test_request_ad_aoti.py.
_DEDICATED = {"request_ad_forward", "implicit_substep", "implicit_substep_nl"}
_SCENARIOS = sorted(
    d
    for d in _SCENARIO_DIR.iterdir()
    if d.is_dir() and (d / "model.i").exists() and d.name not in _DEDICATED
)


def _make_inputs(
    input_spec: dict,
    seed: int = 0,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> dict[str, tuple[torch.Tensor, int]]:
    """Build a dict of test inputs keyed by spec name.

    Each value is ``(raw_tensor, sub_batch_ndim)`` where the raw tensor
    has shape ``(*dyn, *sub, *BASE_SHAPE)``. Defaults to ``dyn=(2,)`` and
    no sub-batch when *shapes* doesn't provide an entry for a name. The
    fixed seed makes runs reproducible; batch=2 forces the dynamic-batch
    dim to a true ``Dim`` rather than a specialized constant.
    """
    gen = torch.Generator().manual_seed(seed)
    shapes = shapes or {}
    inputs: dict[str, tuple[torch.Tensor, int]] = {}
    for name, type_cls in input_spec.items():
        dyn, sub = shapes.get(name, ((2,), ()))
        base = tuple(type_cls.BASE_SHAPE)
        t = torch.randn(*dyn, *sub, *base, generator=gen, dtype=torch.float64)
        inputs[name] = (t, len(sub))
    return inputs


def _read_per_input_shapes(
    hit_path: Path, input_spec: dict
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Resolve per-input ``(dyn, sub)`` shapes from the scenario's HIT
    ``[Settings]/example_batch_shape`` block, mirroring the export-time
    resolution in :func:`neml2.cli.aoti_export._resolve_example_shapes`.
    Scenarios without a ``[Settings]`` block get the library default
    ``dyn=(2,), sub=()`` for every input, matching the legacy behavior.
    """
    from neml2 import load_input
    from neml2.cli.aoti_export import _read_settings, _resolve_example_shapes

    factory = load_input(hit_path)
    declared, _ = _read_settings(factory)
    return _resolve_example_shapes(input_spec, declared)


def _structural_input_spec(model, promoted: set[str]) -> dict:
    """The leaf's input_spec MINUS promoted-parameter names."""
    return {k: v for k, v in model.input_spec.items() if k not in promoted}


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
def test_aoti_export_reload_matches_eager(scenario: Path, tmp_path: Path):
    """Compile the scenario via AOTI, reload via the pybind binding, compare
    forward output against eager."""
    from neml2 import load_input
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import AOTI_META_SCHEMA_VERSION, export_model_for_aoti

    hit_path = scenario / "model.i"

    # Compile (fully baked -- promotion is covered by the dedicated tests).
    out_dir = tmp_path / scenario.name
    meta = export_model_for_aoti(hit_path, "model", out_dir)
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION
    assert meta["parameters"] == []

    # Reload through the pybind binding: pass the artifact ROOT (holding the
    # shared metadata.json + <device>/<dtype>/ binaries).
    aoti = AOTIModel(str(out_dir))

    # Build a reproducible structural-input batch shared by eager and AOTI.
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = dict(eager_model.input_spec)
    # Match the export-time shape declaration so per-input sub-batch
    # axes line up between eager and AOTI -- the .pt2 was compiled with
    # these shapes, so calling it with mismatched ones is an error
    # rather than a silent broadcast.
    shapes = _read_per_input_shapes(hit_path, structural_spec)
    inputs = _make_inputs(structural_spec, shapes=shapes)
    raw_inputs = {name: t for name, (t, _) in inputs.items()}

    # AOTI forward: structural inputs only -- promoted values come from the
    # binding's named_parameters() (populated from the metadata snapshots).
    aoti_outs = aoti.forward(raw_inputs)

    # Eager forward: wrap each structural input in its TensorWrapper type
    # before calling. Promoted entries (if any) stay as the leaf's static
    # parameters -- eager reads them via _get_param's static branch, no need
    # to pass them in *promoted_params order. Pass sub_batch_ndim so the leaf
    # sees the right axis split.
    eager_args = tuple(
        structural_spec[name](t, sub_batch_ndim=sbn) for name, (t, sbn) in inputs.items()
    )
    eager_raw = eager_model(*eager_args)
    eager_tuple = eager_raw if isinstance(eager_raw, tuple) else (eager_raw,)

    out_names = list(eager_model.output_spec)
    assert list(aoti_outs.keys()) == out_names, (
        f"AOTI output keys {list(aoti_outs.keys())} do not match eager output_spec {out_names}"
    )
    for i, name in enumerate(out_names):
        eager_val = eager_tuple[i]
        eager_data = eager_val.data if hasattr(eager_val, "data") else eager_val
        aoti_data = aoti_outs[name]
        max_diff = (aoti_data - eager_data).abs().max().item()
        assert torch.allclose(eager_data, aoti_data, rtol=1e-10, atol=1e-10), (
            f"{scenario.name}: output {name!r} mismatch (max abs diff = {max_diff:.3e})"
        )


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
def test_aoti_stub_loads_through_native_factory(scenario: Path, tmp_path: Path):
    """End-to-end check that the `.i` stub produced by `neml2-compile`
    loads through ``neml2.load_input`` and the resulting AOTIModel
    shim drives like a native Model (typed-wrapper positional args, tuple
    output) -- the path TransientDriver / TransientRegression will use.

    This covers the HIT shim (``register_neml2_object("AOTIModel")``) end-to-end:
    factory dispatch, ``from_hit`` meta-path resolution, typed-wrapper
    marshalling at the call boundary, output matches eager.
    """
    from neml2 import load_input
    from neml2.cli.aoti_compile import emit_aoti_stub
    from neml2.cli.aoti_export import export_model_for_aoti

    hit_path = scenario / "model.i"

    # Schema v10 layout: <out>/model/ is the artifact root (shared metadata.json +
    # <device>/<dtype>/ binaries, created by the exporter) + a standalone
    # <out>/model_aoti.i stub pointing at that folder. The shim picks the
    # <device>/<dtype>/ leaf for the current defaults, so compile for that device.
    out_dir = tmp_path / scenario.name
    dev = torch.get_default_device().type
    artifact_dir = out_dir / "model"
    export_model_for_aoti(hit_path, "model", artifact_dir, device=dev)
    # `export_model_for_aoti` only writes the .pt2 segments + metadata; the
    # `.i` stub is `neml2-compile`'s additional step. Drive it directly so
    # this test stays a single in-process call.
    stub_path = out_dir / "model_aoti.i"
    emit_aoti_stub(hit_path, "model", artifact_dir, stub_path)
    assert stub_path.exists()

    # Load the stub through the same path neml2-run / TransientDriver use.
    shim = load_input(stub_path).get_model("model")
    assert type(shim).__name__ == "AOTIModel"
    assert hasattr(shim, "input_spec") and hasattr(shim, "output_spec")

    # Build positional typed-wrapper args matching the shim's input_spec
    # (fully baked, so all inputs are structural).
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = dict(eager_model.input_spec)
    shapes = _read_per_input_shapes(hit_path, structural_spec)
    inputs = _make_inputs(structural_spec, shapes=shapes)
    typed_args = tuple(
        structural_spec[name](t, sub_batch_ndim=sbn) for name, (t, sbn) in inputs.items()
    )

    # Shim's __call__ returns a tuple of typed wrappers in output_spec order.
    shim_outs = shim(*typed_args)
    assert isinstance(shim_outs, tuple)
    assert len(shim_outs) == len(eager_model.output_spec)

    eager_raw = eager_model(*typed_args)
    eager_tuple = eager_raw if isinstance(eager_raw, tuple) else (eager_raw,)

    for shim_val, eager_val in zip(shim_outs, eager_tuple, strict=True):
        shim_data = shim_val.data if hasattr(shim_val, "data") else shim_val
        eager_data = eager_val.data if hasattr(eager_val, "data") else eager_val
        max_diff = (shim_data - eager_data).abs().max().item()
        assert torch.allclose(eager_data, shim_data, rtol=1e-10, atol=1e-10), (
            f"{scenario.name}: shim output mismatch (max abs diff = {max_diff:.3e})"
        )


@pytest.mark.skip(
    reason="V7 schema bump: sub_batch_labels metadata is not currently "
    "written by _var_infos (static-base wrappers have no labels attribute). "
    "Re-enable when label persistence is plumbed end-to-end."
)
def test_aoti_export_persists_sub_batch_labels(tmp_path: Path):
    """The implicit_cross_group_sub_batch scenario declares
    ``example_batch_shape = '(2; 3:grain)'`` for its per-sub-batch inputs.
    After export, the master meta must carry ``sub_batch_labels: ['grain']``
    on those inputs (and on the matching unknown outputs); after load the
    shim must re-attach ``sub_batch_labels=('grain',)`` to the wrapped
    outputs.

    Pinned in Phase 4 so a later regression that drops the labels at the
    export or load boundary surfaces here instead of silently as a
    label-dispatch miss downstream.
    """
    import json

    from neml2 import load_input
    from neml2.aoti._shim import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    scenario = _SCENARIO_DIR / "implicit_cross_group_sub_batch"
    hit_path = scenario / "model.i"
    out_dir = tmp_path / scenario.name

    export_model_for_aoti(hit_path, "model", out_dir, promoted=set())

    # Read the persisted master meta and assert the grain label is on the
    # per-site input AND the corresponding unknown output.
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    inputs_by_name = {info["name"]: info for info in meta["inputs"]}
    outputs_by_name = {info["name"]: info for info in meta["outputs"]}
    # u_per is the per-site unknown; the master meta records the grain
    # label on both the input (it surfaces as an input to ImplicitUpdate
    # since this scenario has no time-history) and the output.
    assert "u_per" in inputs_by_name
    assert inputs_by_name["u_per"].get("sub_batch_labels") == ["grain"], (
        f"u_per input missing grain label in meta: {inputs_by_name['u_per']!r}"
    )
    assert "u_per" in outputs_by_name
    assert outputs_by_name["u_per"].get("sub_batch_labels") == ["grain"], (
        f"u_per output missing grain label in meta: {outputs_by_name['u_per']!r}"
    )

    # The Python shim re-attaches labels to typed outputs at the load boundary.
    # Schema v10: pass the artifact ROOT, not the metadata.json path.
    shim = AOTIModel(out_dir)
    assert shim.output_labels.get("u_per") == ("grain",)

    # Driving the shim should produce typed wrappers that carry the
    # persisted label on the per-site output.
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = _structural_input_spec(eager_model, set())
    shapes = _read_per_input_shapes(hit_path, structural_spec)
    inputs = _make_inputs(structural_spec, shapes=shapes)
    typed_args = tuple(
        structural_spec[name](t, sub_batch_ndim=sbn) for name, (t, sbn) in inputs.items()
    )
    shim_outs = shim(*typed_args)
    out_names = list(eager_model.output_spec)
    per_out = shim_outs[out_names.index("u_per")]
    assert per_out.sub_batch_labels == ("grain",), (
        f"shim u_per output dropped grain label: labels={per_out.sub_batch_labels!r}"
    )


def test_scenarios_discovered():
    """Sanity: at least one scenario is on disk."""
    assert _SCENARIOS, (
        f"No scenarios found under {_SCENARIO_DIR}. Each subdirectory with a "
        "model.i is treated as a scenario."
    )


def test_parameter_base_shapes_map_matches_eager(tmp_path: Path):
    """The unified parameter-introspection surface: both ``neml2.aoti.Model`` and
    the eager ``_EagerModel`` expose ``parameter_base_shapes`` as a
    ``{qualified_name: base_shape}`` map whose keys are exactly the runtime
    parameter set (``named_parameters()``). For the same model with every typed
    parameter promoted, the two maps are byte-identical -- both use the qualified
    name (a bare leaf named ``model`` reports ``model.E`` on either route, since
    the AOTI export now wraps the leaf in ``ComposedModel([leaf])`` before
    promotion, matching the eager runtime). Promotion alone (no derivative graphs)
    suffices, so this needs no param-deriv torch support."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / "forward_single" / "model.i"

    # The eager parameter surface = every typed parameter; keys == named_parameters().
    em = _EagerModel(str(hit_path), "model")
    assert em.parameter_base_shapes  # forward_single exposes scalar params (E, nu)
    assert set(em.parameter_base_shapes) == set(em.named_parameters())
    assert set(em.parameter_base_shapes) == {"model.E", "model.nu"}  # qualified names

    # Promote everything eager exposes (same qualified namespace) so the AOTI
    # (promoted-subset) surface covers the same parameters.
    out_dir = tmp_path / "pbs"
    export_model_for_aoti(hit_path, "model", out_dir, promoted=set(em.parameter_base_shapes))
    m = PybindModel(str(out_dir))

    # aoti: the map keys are exactly the runtime-settable promoted parameters.
    assert set(m.parameter_base_shapes) == set(m.named_parameters())

    # Byte-identical qualified-name -> base-shape map across the cpp-aoti and
    # cpp-eager routes.
    assert m.parameter_base_shapes == em.parameter_base_shapes


@_REQUIRES_PARAM_DERIV_TORCH
@_skip_win_reextract
def test_aoti_param_jacobian_and_vjp_match_fd(tmp_path: Path):
    """The compiled cpp-aoti parameter-derivative path through the
    pybind Model: ``param_jacobian`` (dense d(out)/d(param)) and ``param_vjp``
    (adjoint dL/d(param)) both agree with finite differences taken by mutating
    the runtime (promoted) parameter. Exercises the full
    export -> lower -> C++ load -> run path; the parameter is runtime-settable."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti

    hit_path = _SCENARIO_DIR / "forward_single" / "model.i"
    out_dir = tmp_path / "param"
    # A bare leaf is wrapped in ComposedModel([leaf]) at export, so the promoted
    # parameter's qualified name is "model.E" (matching the eager runtime).
    export_model_for_aoti(
        hit_path, "model", out_dir, promoted={"model.E"}, derivatives=["stress:model.E"]
    )
    m = PybindModel(str(out_dir))
    assert "model.E" in m.named_parameters()

    torch.manual_seed(0)
    b = 5
    strain = torch.randn(b, 6, dtype=torch.float64)
    raw = {"strain": strain}  # E is a promoted parameter, not a structural input
    e0 = float(m.named_parameters()["model.E"])
    h = 1e-4 * e0

    def set_e(v):
        m.named_parameters()["model.E"].fill_(float(v))

    def stress_at(v):
        set_e(v)
        return m.forward(raw)["stress"].clone()

    # param_jacobian vs FD.
    set_e(e0)
    outs, pjac = m.param_jacobian(raw)
    block = pjac["stress"]["model.E"]
    assert tuple(block.shape) == (b, 6)
    assert torch.allclose(outs["stress"], m.forward(raw)["stress"])
    fd = (stress_at(e0 + h) - stress_at(e0 - h)) / (2 * h)
    set_e(e0)
    rel_j = (block - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
    assert rel_j < 1e-5, f"param_jacobian disagrees with FD (rel={rel_j:.2e})"

    # param_vjp (adjoint) vs FD on a scalar loss L = <w, stress>.
    w = torch.randn(b, 6, dtype=torch.float64)
    set_e(e0)
    grads = m.param_vjp(raw, {"stress": w})
    g_e = float(grads["model.E"])

    def loss_at(v):
        set_e(v)
        return (m.forward(raw)["stress"] * w).sum().item()

    fd_l = (loss_at(e0 + h) - loss_at(e0 - h)) / (2 * h)
    set_e(e0)
    rel_v = abs(g_e - fd_l) / (abs(fd_l) + 1e-30)
    assert rel_v < 1e-5, f"param_vjp disagrees with FD (rel={rel_v:.2e})"

    # Parity: the py-aoti shim forwards the same parameter-derivative surface
    # (named_parameters / param_jacobian / param_vjp / set_parameter) to the
    # binding, so it must agree block-for-block on the same artifact.
    from neml2.aoti import AOTIModel

    set_e(e0)
    shim = AOTIModel(str(out_dir))  # fresh load: E at the e0 snapshot
    assert "model.E" in shim.named_parameters()
    _, s_pjac = shim.param_jacobian(raw)
    assert torch.allclose(s_pjac["stress"]["model.E"], block)
    s_grads = shim.param_vjp(raw, {"stress": w})
    assert torch.allclose(s_grads["model.E"], grads["model.E"])
    shim.set_parameter("model.E", torch.tensor(2.0 * e0, dtype=torch.float64))
    assert float(shim.named_parameters()["model.E"]) == 2.0 * e0


@_REQUIRES_PARAM_DERIV_TORCH
def test_aoti_batched_param_matches_fd_and_eager(tmp_path: Path):
    """A per-batch-element promoted parameter (Scalar ``E`` set to ``(B,)`` via
    ``set_parameter``) flows through the compiled VALUE + param-Jacobian + param-VJP
    graphs (the parameter input carries a dynamic batch dim). ``forward``,
    ``param_jacobian``, and ``param_vjp`` match the eager result with the same
    batched ``E`` and per-element finite differences -- the discriminating test for
    batched params (a batch-collapsing bug passes a uniform-E FD but fails
    per-element). ``param_vjp`` in particular returns a PER-ELEMENT ``(B,)`` adjoint
    for the batched parameter, matching eager rather than the global batch-sum."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / "forward_single" / "model.i"
    out_dir = tmp_path / "batched"
    # Bare leaf wrapped at export => qualified parameter name "model.E" (== eager).
    export_model_for_aoti(
        hit_path, "model", out_dir, promoted={"model.E"}, derivatives=["stress:model.E"]
    )
    m = PybindModel(str(out_dir))

    torch.manual_seed(0)
    b = 5
    strain = torch.randn(b, 6, dtype=torch.float64)
    raw = {"strain": strain}
    e_batched = torch.linspace(90.0, 110.0, b, dtype=torch.float64)  # per-element Youngs modulus
    m.set_parameter("model.E", e_batched.clone())

    # Eager reference with the same batched parameter (same qualified name).
    em = _EagerModel(str(hit_path), "model")
    em.set_parameter("model.E", e_batched.clone())

    # forward: the batched parameter flows through the compiled value graph.
    out = m.forward(raw)["stress"]
    assert tuple(out.shape) == (b, 6)
    assert torch.allclose(out, em.forward(raw)["stress"], atol=1e-10), "forward != eager"

    # param_jacobian block (b, 6) = per-element d(stress_i)/d(E_i); off-diagonal zero.
    outs, pjac = m.param_jacobian(raw)
    block = pjac["stress"]["model.E"]
    assert tuple(block.shape) == (b, 6)
    assert torch.allclose(outs["stress"], out, atol=1e-10)
    _, p_eager = em.param_jacobian(raw)
    assert torch.allclose(block, p_eager["stress"]["model.E"], atol=1e-8), "param_jacobian != eager"

    # Per-element central differences: E_i only affects stress_i.
    h = 1e-6
    fd = torch.zeros(b, 6, dtype=torch.float64)
    for i in range(b):
        up = e_batched.clone()
        up[i] += h
        m.set_parameter("model.E", up)
        sp = m.forward(raw)["stress"]
        dn = e_batched.clone()
        dn[i] -= h
        m.set_parameter("model.E", dn)
        sm = m.forward(raw)["stress"]
        fd[i] = (sp[i] - sm[i]) / (2 * h)
    rel = (block - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
    assert rel < 1e-5, f"batched param_jacobian disagrees with FD (rel={rel:.2e})"

    # param_vjp with the batched parameter: a PER-ELEMENT (b,) adjoint (E_i only
    # affects stress_i), matching eager -- NOT the global batch-summed scalar.
    m.set_parameter("model.E", e_batched.clone())
    w = torch.randn(b, 6, dtype=torch.float64)
    g = m.param_vjp(raw, {"stress": w})["model.E"]
    assert tuple(g.shape) == (b,), f"batched param_vjp should be per-element, got {tuple(g.shape)}"
    g_eager = em.param_vjp(raw, {"stress": w})["model.E"]
    assert torch.allclose(g, g_eager, atol=1e-8), "batched param_vjp != eager"
    # Ground truth: the per-element adjoint is the param-Jacobian block contracted
    # with the cotangent over the output base axis only (batch kept).
    assert torch.allclose(g, (block * w).sum(-1), atol=1e-8), "param_vjp != block-contraction"


@_REQUIRES_PARAM_DERIV_TORCH
def test_aoti_provider_param_jacobian_and_vjp_match_fd(tmp_path: Path):
    """Promote a ``ScalarConstantParameter`` PROVIDER and take its derivatives.

    The provider's value feeds a consumer as a promoted parameter, and the
    dependency resolver places the promoted provider input AHEAD of the
    structural input -- so the graph's input order differs from the C++ runtime's
    structural-then-param feed. Without the spec-container rebuild + params-last
    reorder this silently swapped arguments (param_jacobian returned the
    parameter value, vjp returned 0). Regression: param_jacobian / param_vjp
    must agree with finite differences and the eager route."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / "forward_provider_param" / "model.i"
    out_dir = tmp_path / "provider"
    export_model_for_aoti(
        hit_path, "model", out_dir, promoted={"k.value"}, derivatives=["y:k.value"]
    )
    m = PybindModel(str(out_dir))

    x = torch.tensor([2.0, 5.0, -1.0, 3.5], dtype=torch.float64)
    k0 = float(m.named_parameters()["k.value"])

    # param_jacobian: y = k * x  =>  d(y)/d(k) = x.
    outs, pjac = m.param_jacobian({"x": x})
    block = pjac["y"]["k.value"]
    assert torch.allclose(outs["y"], k0 * x)
    assert torch.allclose(block.flatten(), x), f"d(y)/d(k) wrong: {block.flatten().tolist()}"

    # FD on the value to be doubly sure (mutate the runtime parameter).
    h = 1e-5 * max(abs(k0), 1.0)
    m.named_parameters()["k.value"].fill_(k0 + h)
    yp = m.forward({"x": x})["y"].clone()
    m.named_parameters()["k.value"].fill_(k0 - h)
    ym = m.forward({"x": x})["y"].clone()
    m.named_parameters()["k.value"].fill_(k0)
    fd = (yp - ym) / (2 * h)
    assert (block.flatten() - fd).abs().max() < 1e-5 * (fd.abs().max() + 1e-30)

    # param_vjp: dL/dk for L = <w, y>  =>  sum(w * x).
    torch.manual_seed(0)
    w = torch.randn(4, dtype=torch.float64)
    g = float(m.param_vjp({"x": x}, {"y": w})["k.value"])
    assert abs(g - float((w * x).sum())) < 1e-9

    # Eager parity.
    em = _EagerModel(str(hit_path), "model")
    _, PE = em.param_jacobian({"x": x})
    assert torch.allclose(block.flatten(), PE["y"]["k.value"].flatten())


@_REQUIRES_PARAM_DERIV_TORCH
def test_aoti_implicit_param_jacobian_matches_fd(tmp_path: Path):
    """The compiled cpp-aoti parameter-derivative path through a single
    ``ImplicitUpdate``: ``param_jacobian`` returns ``du/dθ`` for a
    parameter promoted INSIDE the residual, computed by the ParamIFT graph
    (``-A⁻¹ ∂r/∂θ`` -- analytic-equivalent reverse-mode A + reverse-mode ∂r/∂θ +
    dense solve). Validated against finite differences taken by mutating the
    runtime (promoted) parameter, and cross-checked against the eager
    ``param_jacobian`` (the six-route parity reference). The parameter is
    runtime-settable, which is what MOOSE inverse optimization needs."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / "implicit_param" / "model.i"
    qname = "residual.rate.weight_0"
    out_dir = tmp_path / "implicit_param_deriv"
    export_model_for_aoti(hit_path, "model", out_dir, promoted={qname}, derivatives=[f"x:{qname}"])
    m = PybindModel(str(out_dir))
    assert qname in m.named_parameters()

    # Uniaxial-style implicit step: x~1 ramps, dt = t - t~1 = 0.5, x IC = 0.
    b = 5
    xn = torch.linspace(1.0, 2.0, b, dtype=torch.float64)
    ins = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": xn,
        "t": torch.full((b,), 0.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
    }
    w0 = float(m.named_parameters()[qname])
    h = 1e-5 * max(abs(w0), 1.0)

    def set_w(v):
        m.named_parameters()[qname].fill_(float(v))

    def x_at(v):
        set_w(v)
        return m.forward(ins)["x"].clone()

    set_w(w0)
    outs, pjac = m.param_jacobian(ins)
    block = pjac["x"][qname]
    assert tuple(block.shape) == (b,)  # scalar unknown, scalar param -> (*B,)
    assert torch.allclose(outs["x"], m.forward(ins)["x"])

    fd = (x_at(w0 + h) - x_at(w0 - h)) / (2 * h)
    set_w(w0)
    rel = (block - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
    assert rel < 1e-5, f"implicit param_jacobian disagrees with FD (rel={rel:.2e})"

    # Cross-check against the eager route (parity invariant). Eager keys the
    # parameter under the `_EagerModel`-wrapped module prefix `model.`.
    em = _EagerModel(str(hit_path), "model")
    _, P = em.param_jacobian(ins)
    eager_block = P["x"][f"model.{qname}"]
    rel_e = (block - eager_block).abs().max().item() / (eager_block.abs().max().item() + 1e-30)
    assert rel_e < 1e-9, f"cpp-aoti vs eager implicit param_jacobian disagree (rel={rel_e:.2e})"

    # param_vjp (adjoint dL/dθ for L = <w, x>) through the implicit solve: the
    # carrier contracts du/dθ with the cotangent. Validate vs FD on the scalar
    # loss and vs the Jacobian contraction <w, dx/dθ>.
    set_w(w0)
    torch.manual_seed(0)
    w = torch.randn(b, dtype=torch.float64)
    g_vjp = float(m.param_vjp(ins, {"x": w})[qname])

    def loss_at(v):
        set_w(v)
        return float((m.forward(ins)["x"] * w).sum())

    fd_l = (loss_at(w0 + h) - loss_at(w0 - h)) / (2 * h)
    set_w(w0)
    assert abs(g_vjp - fd_l) < 1e-5 * max(abs(fd_l), 1.0), (
        f"implicit param_vjp disagrees with FD (vjp={g_vjp}, fd={fd_l})"
    )
    contracted = float((w * block).sum())
    assert abs(g_vjp - contracted) < 1e-9, "implicit param_vjp != <w, dx/dθ>"


@_REQUIRES_PARAM_DERIV_TORCH
def test_aoti_composed_param_jacobian_matches_fd(tmp_path: Path):
    """The compiled cpp-aoti MULTI-SEGMENT parameter-Jacobian carrier:
    d(out)/d(param) for parameters promoted in EACH segment of a composed
    forward -> implicit -> forward model. The carrier composes the direct
    contribution at the parameter's own segment (forward param-Jacobian /
    ParamIFT) with the indirect propagation through every downstream segment
    (jvp / IFT), so this exercises:

      * c in the leading forward segment  -> propagated through the implicit IFT
        and the trailing forward jvp,
      * m inside the implicit residual    -> ParamIFT direct + trailing jvp,
      * b in the trailing forward segment -> direct only.

    Validated vs finite differences (mutating each runtime-settable parameter)
    and cross-checked against the eager param_jacobian (six-route parity)."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / "composed_param" / "model.i"
    params = [
        "scale.weight_0",
        "return_map.impl_residual.modrate.weight_0",
        "out.weight_0",
    ]
    out_dir = tmp_path / "composed_param_deriv"
    export_model_for_aoti(
        hit_path, "model", out_dir, promoted=set(params), derivatives=[f"y:{p}" for p in params]
    )
    m = PybindModel(str(out_dir))
    for p in params:
        assert p in m.named_parameters()

    b = 5
    f = torch.linspace(0.5, 1.5, b, dtype=torch.float64)
    xn = torch.linspace(1.0, 2.0, b, dtype=torch.float64)
    ins = {
        "f": f,
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": xn,
        "t": torch.full((b,), 0.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
    }

    p0 = {p: float(m.named_parameters()[p]) for p in params}

    def set_p(p, v):
        m.named_parameters()[p].fill_(float(v))

    def restore():
        for p, v in p0.items():
            set_p(p, v)

    restore()
    outs, pjac = m.param_jacobian(ins)
    assert torch.allclose(outs["y"], m.forward(ins)["y"])

    # FD per parameter (central difference on the forward output y).
    for p in params:
        block = pjac["y"][p]
        assert tuple(block.shape) == (b,)
        v0 = p0[p]
        h = 1e-5 * max(abs(v0), 1.0)
        set_p(p, v0 + h)
        yp = m.forward(ins)["y"].clone()
        set_p(p, v0 - h)
        ym = m.forward(ins)["y"].clone()
        restore()
        fd = (yp - ym) / (2 * h)
        rel = (block - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
        assert rel < 1e-5, f"composed param_jacobian d(y)/d({p}) disagrees with FD (rel={rel:.2e})"

    # Cross-check against the eager route (parity invariant).
    em = _EagerModel(str(hit_path), "model")
    _, PE = em.param_jacobian(ins)
    for p in params:
        eager_block = PE["y"][p]
        rel_e = (pjac["y"][p] - eager_block).abs().max().item() / (
            eager_block.abs().max().item() + 1e-30
        )
        assert rel_e < 1e-9, (
            f"cpp-aoti vs eager composed param_jacobian d(y)/d({p}) (rel={rel_e:.2e})"
        )

    # param_vjp (adjoint dL/dθ for L = <w, y>) across the composed graph: the
    # carrier contracts d(y)/dθ with the cotangent. Validate vs FD on the scalar
    # loss and vs the Jacobian contraction <w, d(y)/dθ> for every parameter.
    restore()
    torch.manual_seed(0)
    w = torch.randn(b, dtype=torch.float64)
    grads = m.param_vjp(ins, {"y": w})
    for p in params:
        v0 = p0[p]
        h = 1e-6 * max(abs(v0), 1.0)
        set_p(p, v0 + h)
        lp = float((m.forward(ins)["y"] * w).sum())
        set_p(p, v0 - h)
        lm = float((m.forward(ins)["y"] * w).sum())
        restore()
        fd_l = (lp - lm) / (2 * h)
        g_vjp = float(grads[p])
        assert abs(g_vjp - fd_l) < 1e-4 * max(abs(fd_l), 1.0), (
            f"composed param_vjp d(L)/d({p}) disagrees with FD (vjp={g_vjp}, fd={fd_l})"
        )
        contracted = float((w * pjac["y"][p]).sum())
        assert abs(g_vjp - contracted) < 1e-9, f"composed param_vjp d(L)/d({p}) != <w, d(y)/dθ>"


# Saved-output element-wise ops (sqrt / exp / reciprocal+division) on the
# parameter-derivative path: each would hit the upstream AOTI lowering bug
# (pytorch/pytorch#187907; pinned by test_upstream_pytorch_187907) without the
# input-recompute workaround in ``neml2/types/functions.py``. These compile +
# match FD + eager, proving the workaround restores AOTI parity. ``inp`` builds
# the structural input (raw tensor) for each scenario.
_SAVED_OUTPUT_PARAM_CASES = [
    # (scenario, output var, promoted param, structural-input builder)
    (
        "forward_exp_param",
        "out",
        "ys.C",
        lambda: {"x": torch.linspace(1.0, 5.0, 5, dtype=torch.float64)},
    ),
    (
        "forward_reciprocal_param",
        "eta",
        "visc.mu",
        lambda: {"T": torch.linspace(1.0, 3.0, 5, dtype=torch.float64)},
    ),
    (
        "forward_vonmises_param",
        "effective_stress",
        "scale_stress.weight_0",
        lambda: {
            "stress_in": torch.randn(
                5, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64
            )
            * 50.0
        },
    ),
]


@pytest.mark.parametrize(
    ("scenario", "out_var", "param", "make_inputs"),
    _SAVED_OUTPUT_PARAM_CASES,
    ids=[c[0] for c in _SAVED_OUTPUT_PARAM_CASES],
)
@_REQUIRES_PARAM_DERIV_TORCH
def test_aoti_saved_output_param_derivative_matches_fd_and_eager(
    scenario, out_var, param, make_inputs, tmp_path: Path
):
    """A parameter derivative through a saved-output op (sqrt / exp / reciprocal)
    compiles and matches FD + eager. Without the input-recompute workaround the
    export would fail to lower (pytorch/pytorch#187907)."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit_path = _SCENARIO_DIR / scenario / "model.i"
    out_dir = tmp_path / scenario
    export_model_for_aoti(
        hit_path, "model", out_dir, promoted={param}, derivatives=[f"{out_var}:{param}"]
    )
    m = PybindModel(str(out_dir))
    ins = make_inputs()

    # cpp-aoti param_jacobian vs finite differences on the runtime parameter.
    _, pjac = m.param_jacobian(ins)
    block = pjac[out_var][param].flatten()
    p0 = float(m.named_parameters()[param])
    h = 1e-6 * max(abs(p0), 1.0)
    m.named_parameters()[param].fill_(p0 + h)
    yp = m.forward(ins)[out_var].clone()
    m.named_parameters()[param].fill_(p0 - h)
    ym = m.forward(ins)[out_var].clone()
    m.named_parameters()[param].fill_(p0)
    fd = ((yp - ym) / (2 * h)).flatten()
    assert block.abs().max() > 1e-6, "degenerate scenario: parameter derivative is ~0"
    rel = (block - fd).abs().max() / (fd.abs().max() + 1e-30)
    assert rel < 1e-5, f"{scenario} param_jacobian disagrees with FD (rel={rel:.2e})"

    # Eager parity (the six-route reference).
    em = _EagerModel(str(hit_path), "model")
    _, PE = em.param_jacobian(ins)
    eblock = PE[out_var][param].flatten()
    rel_e = (block - eblock).abs().max() / (eblock.abs().max() + 1e-30)
    assert rel_e < 1e-9, f"{scenario} cpp-aoti vs eager param_jacobian (rel={rel_e:.2e})"

    # param_vjp adjoint == <w, d(out)/dparam>.
    torch.manual_seed(0)
    w = torch.randn_like(fd)
    g = float(m.param_vjp(ins, {out_var: w})[param])
    contracted = float((w * eblock).sum())
    assert abs(g - contracted) < 1e-8 * max(abs(contracted), 1.0), (
        f"{scenario} param_vjp != <w, d(out)/dparam> (vjp={g}, contracted={contracted})"
    )
