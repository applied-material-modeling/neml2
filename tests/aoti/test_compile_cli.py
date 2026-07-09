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
``compile_and_emit_stub`` helper both emit one artifact folder per device
(``<out>/<model>/<device>/``) plus a single standalone ``<out>/<model>_aoti.i``
stub that points at the folder via an absolute ``artifact_path``. Also exercises
the two cheap error exits (missing input, unknown model).

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
    assert (tmp_path / "model" / "cpu" / "model_meta.json").exists()

    # The shim points at the artifact folder via artifact_path; the old
    # per-device `meta =` field is gone.
    stub_text = stub.read_text()
    assert "artifact_path" in stub_text
    assert "\n    meta =" not in stub_text


def test_compile_and_emit_stub_layout(tmp_path):
    stub = compile_and_emit_stub(_INPUT, tmp_path, model="model")
    assert stub == tmp_path / "model_aoti.i"
    assert stub.exists()
    assert (tmp_path / "model" / "cpu" / "model_meta.json").exists()


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

    on_disk = {p.name for p in tmp_path.iterdir() if p.is_file()}
    # Ordered prediction matches the ordered progress stream (serial run) ...
    assert reported == predicted
    # ... and the exact set of files on disk (.pt2 graphs + the meta.json).
    assert set(predicted) == on_disk


def test_planned_equals_produced_includes_stub(tmp_path):
    """At the CLI level the standalone ``_aoti.i`` stub is generated too; the
    progress stream ends with it and the file exists on disk."""
    reported: list[str] = []
    stub = compile_and_emit_stub(_INPUT, tmp_path, model="model", progress_cb=reported.append)
    assert stub.exists()
    assert reported[-1] == "model_aoti.i"
    # The stub is the last generated file, after every .pt2 and the meta.json.
    assert reported == ["model.pt2", "model_meta.json", "model_aoti.i"]


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
    # can differ trivially run-to-run).
    serial_files = {p.name for p in serial_dir.iterdir() if p.is_file()}
    parallel_files = {p.name for p in parallel_dir.iterdir() if p.is_file()}
    assert serial_files == parallel_files


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
    assert {p.name for p in j1_dir.iterdir() if p.is_file()} == {
        p.name for p in j4_dir.iterdir() if p.is_file()
    }
