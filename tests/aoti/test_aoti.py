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

Gated by the same ``--run-aoti-compile`` pytest flag as
``test_aoti_export.py`` because AOTI compilation is slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.aoti_compile

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


def _make_inputs(input_spec: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build a dict of ``randn(2, *BASE_SHAPE)`` tensors keyed by spec name.

    The fixed seed makes the test reproducible across runs. Batch=2 forces
    the dynamic-batch dim to be a true Dim rather than a specialized constant.
    """
    gen = torch.Generator().manual_seed(seed)
    inputs: dict[str, torch.Tensor] = {}
    for name, type_cls in input_spec.items():
        base = tuple(type_cls.BASE_SHAPE)
        inputs[name] = torch.randn(2, *base, generator=gen, dtype=torch.float64)
    return inputs


def _structural_input_spec(model, promoted: set[str]) -> dict:
    """The leaf's input_spec MINUS promoted-parameter names."""
    return {k: v for k, v in model.input_spec.items() if k not in promoted}


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
def test_aoti_export_reload_matches_eager(scenario: Path, tmp_path: Path):
    """Compile the scenario via AOTI, reload via the pybind binding, compare
    forward output against eager."""
    from neml2 import load_input
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    promoted = _load_promoted(scenario)
    hit_path = scenario / "model.i"

    # Compile.
    out_dir = tmp_path / scenario.name
    meta = export_model_for_aoti(hit_path, "model", out_dir, promoted=promoted)
    assert meta["schema_version"] == 2
    if promoted:
        assert {p["name"] for p in meta["parameters"]} == promoted
    else:
        assert meta["parameters"] == []

    # Reload through the pybind binding.
    aoti = AOTIModel(out_dir / "model_meta.json")

    # Build a reproducible structural-input batch shared by eager and AOTI.
    # ``export_model_for_aoti`` mutates its own model copy via _promote_to_nl_params,
    # not the one we load here, so eager_model retains its original input_spec.
    eager_model = load_input(hit_path).get_model("model")
    structural_spec = _structural_input_spec(eager_model, promoted)
    inputs = _make_inputs(structural_spec)

    # AOTI forward: structural inputs only -- promoted values come from the
    # binding's named_parameters() (populated from the metadata snapshots).
    aoti_outs = aoti.forward(inputs)

    # Eager forward: wrap each structural input in its TensorWrapper type
    # before calling. Promoted entries (if any) stay as the leaf's static
    # parameters -- eager reads them via _get_param's static branch, no need
    # to pass them in *nl_params order.
    eager_args = tuple(structural_spec[name](inputs[name]) for name in structural_spec)
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

    out_dir = tmp_path / scenario.name
    export_model_for_aoti(hit_path, "model", out_dir, promoted=promoted)
    # `export_model_for_aoti` only writes the .pt2 segments + metadata; the
    # `.i` stub is `neml2-compile`'s additional step. Drive it directly so
    # this test stays a single in-process call.
    stub_path = out_dir / "model_aoti.i"
    emit_aoti_stub(hit_path, "model", out_dir / "model_meta.json", stub_path)
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
    inputs = _make_inputs(structural_spec)
    typed_args = tuple(structural_spec[name](inputs[name]) for name in structural_spec)

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


def test_scenarios_discovered():
    """Sanity: at least one scenario is on disk."""
    assert _SCENARIOS, (
        f"No scenarios found under {_SCENARIO_DIR}. Each subdirectory with a "
        "model.i is treated as a scenario."
    )
