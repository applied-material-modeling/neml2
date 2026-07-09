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

"""Device resolution in the ``AOTIModel`` HIT shim's ``from_hit``.

The shim reads an ``artifact_path`` (absolute or input-relative) and loads the
``<artifact_path>/<device>/*_meta.json`` for ``torch.get_default_device()``.
These tests drive its resolution branches with crafted stubs + empty/odd
artifact folders, so none reach the C++ binding (the error / ambiguity checks
fire first) and no AOTI compile is needed. The happy path is covered by
``tests/aoti``'s stub-loading test.
"""

from __future__ import annotations

import pytest
import torch

from neml2 import load_input


def _write_stub(tmp_path, artifact_path_value: str):
    stub = tmp_path / "model_aoti.i"
    # INTERIM (nmhit-backslash): the HIT lexer rejects backslashes in quoted
    # strings, so a Windows path string fails to parse. Normalize to forward
    # slashes here. Revert this once nmhit accepts backslashes in quoted strings
    # (the real fix); see also the matching workaround in cli/aoti_compile.py.
    artifact_path_value = artifact_path_value.replace("\\", "/")
    stub.write_text(
        "[Models]\n"
        "  [model]\n"
        "    type = AOTIModel\n"
        f"    artifact_path = '{artifact_path_value}'\n"
        "  []\n"
        "[]\n"
    )
    return stub


def test_missing_device_subfolder_errors(tmp_path):
    # Artifact folder exists but has no subfolder for the default device.
    (tmp_path / "model").mkdir()
    stub = _write_stub(tmp_path, str(tmp_path / "model"))
    with pytest.raises(FileNotFoundError, match="no artifact compiled for device"):
        load_input(str(stub)).get_model("model")


def test_relative_artifact_path_resolved_against_input_dir(tmp_path):
    # A relative artifact_path resolves against the stub's directory.
    (tmp_path / "model").mkdir()
    stub = _write_stub(tmp_path, "./model")
    with pytest.raises(FileNotFoundError, match="no artifact compiled for device"):
        load_input(str(stub)).get_model("model")


def test_multiple_metas_in_device_folder_errors(tmp_path):
    dev = torch.get_default_device().type
    device_dir = tmp_path / "model" / dev
    device_dir.mkdir(parents=True)
    (device_dir / "a_meta.json").write_text("{}")
    (device_dir / "b_meta.json").write_text("{}")
    stub = _write_stub(tmp_path, str(tmp_path / "model"))
    with pytest.raises(RuntimeError, match="multiple"):
        load_input(str(stub)).get_model("model")
