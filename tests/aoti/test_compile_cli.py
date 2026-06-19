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

from pathlib import Path

from neml2.cli.aoti_compile import compile_and_emit_stub, main

_INPUT = Path(__file__).parent / "forward_single" / "model.i"


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
