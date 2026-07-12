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

"""Drive the ``neml2-compile`` CLI surface end-to-end.

Covers the multi-device output layout: ``main()`` and the
``compile_and_emit_stub`` helper both emit one artifact folder per model
(``<out>/<model>/`` holding one shared ``metadata.json`` + per-``<device>/<dtype>/``
binaries) plus a single standalone ``<out>/<model>_aoti.i`` stub that points at
the folder via a stub-relative ``artifact_path``. Also exercises the two cheap
error exits (missing input, unknown model).

The forward_single leaf is the smallest scenario; the second compile of it hits
the warm Inductor cache, so the per-device-layout assertions cost roughly one
compile.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neml2.cli.aoti_compile import compile_and_emit_stub, main

_INPUT = Path(__file__).parent / "forward_single" / "model.i"
# A ComposedModel containing an ImplicitUpdate -> partitions into 3 segments
# (forward / implicit / forward), the shape that exercises parallel compilation.
_COMPOSED_INPUT = Path(__file__).parent / "composed_param" / "model.i"


def test_main_emits_standalone_stub_and_per_device_folder(tmp_path):
    rc = main([str(_INPUT), "--model", "model", "--output-dir", str(tmp_path)])
    assert rc == 0

    stub = tmp_path / "model_aoti.i"
    assert stub.exists()
    # Schema v10: one shared metadata.json at the artifact root, binaries under
    # <device>/<dtype>/.
    assert (tmp_path / "model" / "metadata.json").exists()
    assert (tmp_path / "model" / "cpu" / "float64").is_dir()

    # The shim points at the artifact folder via artifact_path; the old
    # per-device `meta =` field is gone.
    stub_text = stub.read_text()
    assert "artifact_path" in stub_text
    assert "\n    meta =" not in stub_text
    # Schema v10: the stub carries no [Solvers] block / solver field (the config
    # rides along in metadata.json).
    assert "[Solvers]" not in stub_text
    assert "\n    solver =" not in stub_text


def test_compile_and_emit_stub_layout(tmp_path):
    stub = compile_and_emit_stub(_INPUT, tmp_path, model="model")
    assert stub == tmp_path / "model_aoti.i"
    assert stub.exists()
    assert (tmp_path / "model" / "metadata.json").exists()
    assert (tmp_path / "model" / "cpu" / "float64").is_dir()


def test_compile_and_emit_stub_no_stub(tmp_path):
    """``emit_stub=False`` skips the standalone ``_aoti.i`` while still writing the
    self-describing artifact folder (metadata.json + <device>/<dtype>/ binaries)."""
    stub = compile_and_emit_stub(_INPUT, tmp_path, model="model", emit_stub=False)
    assert not stub.exists()
    assert (tmp_path / "model" / "metadata.json").exists()
    assert (tmp_path / "model" / "cpu" / "float64").is_dir()


def test_main_no_stub(tmp_path):
    """``--no-stub`` on the CLI suppresses the ``_aoti.i`` stub."""
    rc = main([str(_INPUT), "--model", "model", "--output-dir", str(tmp_path), "--no-stub"])
    assert rc == 0
    assert not (tmp_path / "model_aoti.i").exists()
    assert (tmp_path / "model" / "metadata.json").exists()


def test_main_input_not_found(tmp_path):
    assert main([str(tmp_path / "nope.i"), "--model", "model"]) == 1


def test_main_unknown_model(tmp_path):
    # export fails resolving the model -> main catches it and returns 1.
    assert main([str(_INPUT), "--model", "does_not_exist", "--output-dir", str(tmp_path)]) == 1


def test_main_rejects_bad_jobs(tmp_path):
    # --jobs < 1 is a usage error (argparse SystemExit via parser.error).
    with pytest.raises(SystemExit):
        main([str(_INPUT), "--model", "model", "--output-dir", str(tmp_path), "-j", "0"])


# ---------------------------------------------------------------------------
# Artifact-plan drift guard: what plan_export_artifacts predicts must equal what
# a real compile actually generates -- .pt2 graphs + meta.json (+ the .i stub at
# the CLI level). This is the single check that keeps the emit-order predictors
# in lockstep with the compile_model call sites.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inp", "model", "derivs"),
    [
        (_INPUT, "model", ()),
        (_INPUT, "model", (":",)),
        (_COMPOSED_INPUT, "model", ()),
        (_COMPOSED_INPUT, "model", (":",)),
    ],
    ids=["forward", "forward+d", "composed", "composed+d"],
)
def test_planned_equals_produced(tmp_path, inp, model, derivs):
    from neml2.cli.aoti_export import export_model_for_aoti, plan_export_artifacts

    predicted = plan_export_artifacts(inp, model, derivatives=derivs).artifacts
    reported: list[str] = []
    export_model_for_aoti(inp, model, tmp_path, derivatives=derivs, progress_cb=reported.append)

    # Schema v10 layout: the shared metadata.json sits at the artifact root while
    # the .pt2 binaries live under <device>/<dtype>/. Gather both, keyed by bare
    # name (the predictor + progress stream both speak bare filenames).
    leaf = tmp_path / "cpu" / "float64"
    on_disk = {p.name for p in tmp_path.iterdir() if p.is_file()}
    on_disk |= {p.name for p in leaf.iterdir() if p.is_file()}
    # Ordered prediction matches the ordered progress stream (serial run) ...
    assert reported == predicted
    # ... and the exact set of files on disk (.pt2 graphs + the metadata.json).
    assert set(predicted) == on_disk


def test_planned_equals_produced_includes_stub(tmp_path):
    """At the CLI level the standalone ``_aoti.i`` stub is generated too; the
    progress stream ends with it and the file exists on disk."""
    reported: list[str] = []
    stub = compile_and_emit_stub(_INPUT, tmp_path, model="model", progress_cb=reported.append)
    assert stub.exists()
    assert reported[-1] == "model_aoti.i"
    # The stub is the last generated file, after every .pt2 and the metadata.json.
    assert reported == ["model.pt2", "metadata.json", "model_aoti.i"]


# ---------------------------------------------------------------------------
# Parallel segment compilation
# ---------------------------------------------------------------------------


def test_parallel_matches_serial(tmp_path):
    """Compiling a multi-segment model with jobs>1 must yield byte-identical
    metadata and the same set of artifacts as a serial (jobs=1) compile --
    segment metadata is reassembled in segment order regardless of completion
    order."""
    from neml2.cli.aoti_export import export_model_for_aoti

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    meta_serial = export_model_for_aoti(_COMPOSED_INPUT, "model", serial_dir, jobs=1)
    # jobs (8) intentionally exceeds the segment count (3) -- the pool is capped at
    # the number of segments, so this must still match the serial output exactly.
    meta_parallel = export_model_for_aoti(_COMPOSED_INPUT, "model", parallel_dir, jobs=8)

    # Metadata identical (order-insensitive JSON compare).
    assert json.dumps(meta_serial, sort_keys=True) == json.dumps(meta_parallel, sort_keys=True)

    # Same artifacts on disk (compare names, not .pt2 bytes -- Inductor output
    # can differ trivially run-to-run). Gather the shared metadata.json at the
    # root plus the <device>/<dtype>/ .pt2 binaries (artifact-root layout).
    def _artifact_names(root: Path) -> set[str]:
        names = {p.name for p in root.iterdir() if p.is_file()}
        names |= {p.name for p in (root / "cpu" / "float64").iterdir() if p.is_file()}
        return names

    assert _artifact_names(serial_dir) == _artifact_names(parallel_dir)


def test_jobs_ignored_for_single_segment(tmp_path):
    """A single-segment (forward-only) model has nothing to parallelize; jobs>1
    falls back to the serial path and produces identical output."""
    from neml2.cli.aoti_export import export_model_for_aoti

    j1_dir = tmp_path / "j1"
    j4_dir = tmp_path / "j4"
    j1_dir.mkdir()
    j4_dir.mkdir()

    meta_j1 = export_model_for_aoti(_INPUT, "model", j1_dir, jobs=1)
    meta_j4 = export_model_for_aoti(_INPUT, "model", j4_dir, jobs=4)

    assert json.dumps(meta_j1, sort_keys=True) == json.dumps(meta_j4, sort_keys=True)

    def _artifact_names(root: Path) -> set[str]:
        names = {p.name for p in root.iterdir() if p.is_file()}
        names |= {p.name for p in (root / "cpu" / "float64").iterdir() if p.is_file()}
        return names

    assert _artifact_names(j1_dir) == _artifact_names(j4_dir)


# ---------------------------------------------------------------------------
# Multi-device grid orchestration (export_model_multidevice)
# ---------------------------------------------------------------------------


def test_multidevice_matches_single_device_export(tmp_path):
    """A single-device grid run must produce metadata + artifacts byte-identical
    to a direct per-device export_model_for_aoti (the CLI now routes through the
    grid orchestrator, so this pins that they agree)."""
    from neml2.cli.aoti_export import export_model_for_aoti, export_model_multidevice

    direct_root = tmp_path / "direct"
    direct_root.mkdir(parents=True)
    grid_root = tmp_path / "grid"
    grid_root.mkdir()

    # Both paths use the artifact-root layout: shared metadata.json at
    # the root, .pt2 binaries under <device>/<dtype>/.
    meta_direct = export_model_for_aoti(_COMPOSED_INPUT, "model", direct_root, derivatives=[":"])
    meta_grid = export_model_multidevice(
        _COMPOSED_INPUT, "model", grid_root, ["cpu"], derivatives=[":"]
    )

    # export_model_multidevice returns a single shared meta dict,
    # identical to the direct single-device export.
    assert json.dumps(meta_grid, sort_keys=True) == json.dumps(meta_direct, sort_keys=True)
    assert {p.name for p in (grid_root / "cpu" / "float64").iterdir() if p.is_file()} == {
        p.name for p in (direct_root / "cpu" / "float64").iterdir() if p.is_file()
    }


def test_multidevice_grid_parallel_matches_serial_and_tags_device(tmp_path):
    """The grid pool (jobs>1) matches serial output and reports device-tagged
    progress; over-provisioned jobs are capped at the grid size."""
    from neml2.cli.aoti_export import export_model_multidevice

    serial_dir = tmp_path / "s"
    par_dir = tmp_path / "p"
    serial_dir.mkdir()
    par_dir.mkdir()

    reported: list[str] = []
    meta_serial = export_model_multidevice(_COMPOSED_INPUT, "model", serial_dir, ["cpu"], jobs=1)
    meta_par = export_model_multidevice(
        _COMPOSED_INPUT, "model", par_dir, ["cpu"], jobs=8, progress_cb=reported.append
    )

    # Single shared meta dict per run; serial and parallel agree.
    assert json.dumps(meta_serial, sort_keys=True) == json.dumps(meta_par, sort_keys=True)
    # Every reported .pt2 is device-tagged (single-device -> "cpu/"); the shared
    # metadata.json is written once at the root and reported untagged.
    assert reported
    assert all(name.startswith("cpu/") for name in reported if name.endswith(".pt2"))
    assert "metadata.json" in reported
