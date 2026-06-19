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
named ``model`` via :func:`~neml2.cli.aoti_export.export_model_for_aoti`,
reloads the artifact through :class:`~neml2.aoti.Model`, and asserts
the compiled forward output matches the eager native run within
``rtol/atol``.

Per-scenario knobs (all optional, sidecar files):

* ``promote.txt`` -- one parameter qname per line; passed as the ``promoted``
  argument to ``export_model_for_aoti``. Absent = fully baked.

Runs by default. Each scenario triggers an Inductor compile, so the
full sweep is slow -- the CI ``test`` job allocates several minutes
to this directory alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_SCENARIO_DIR = Path(__file__).parent
_SCENARIOS = sorted(d for d in _SCENARIO_DIR.iterdir() if d.is_dir() and (d / "model.i").exists())


def _load_promoted(scenario: Path) -> set[str]:
    promote_file = scenario / "promote.txt"
    if not promote_file.exists():
        return set()
    return {
        line.strip()
        for line in promote_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


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

    promoted = _load_promoted(scenario)
    hit_path = scenario / "model.i"

    # Compile.
    out_dir = tmp_path / scenario.name
    meta = export_model_for_aoti(hit_path, "model", out_dir, promoted=promoted)
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION
    if promoted:
        assert {p["name"] for p in meta["parameters"]} == promoted
    else:
        assert meta["parameters"] == []

    # Reload through the pybind binding.
    aoti = AOTIModel(str(out_dir / "model_meta.json"))

    # Build a reproducible structural-input batch shared by eager and AOTI.
    # ``export_model_for_aoti`` mutates its own model copy via _promote_to_nl_params,
    # not the one we load here, so eager_model retains its original input_spec.
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = _structural_input_spec(eager_model, promoted)
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
    # to pass them in *nl_params order. Pass sub_batch_ndim so the leaf
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

    promoted = _load_promoted(scenario)
    hit_path = scenario / "model.i"

    # New layout: <out>/model/<device>/ artifacts + a standalone <out>/model_aoti.i
    # stub that points at the artifact folder. The shim picks the subfolder for
    # the current default device, so compile for exactly that device.
    out_dir = tmp_path / scenario.name
    dev = torch.get_default_device().type
    artifact_dir = out_dir / "model"
    export_model_for_aoti(hit_path, "model", artifact_dir / dev, device=dev, promoted=promoted)
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

    # Build positional typed-wrapper args matching the shim's input_spec.
    # The shim's input_spec excludes promoted entries (they live on the
    # underlying binding's named_parameters); structural inputs match the
    # eager model's structural spec.
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = _structural_input_spec(eager_model, promoted)
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
    meta_path = out_dir / "model_meta.json"
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
    shim = AOTIModel(meta_path)
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
