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

"""Artifact-root resolution in the ``AOTIModel`` HIT shim (schema v10).

The shim reads an ``artifact_path`` (absolute or input-relative) pointing at the
artifact ROOT folder produced by ``neml2-compile`` -- one shared
``metadata.json`` at its root plus per-``<device>/<dtype>/`` ``.pt2`` binaries.
``AOTIModel.__init__`` reads ``<artifact_root>/metadata.json`` (raising
``FileNotFoundError`` if absent, before touching the C++ binding), then hands the
root + the ambient ``torch.get_default_device()`` / ``torch.get_default_dtype()``
to the C++ ctor, which selects the ``<device>/<dtype>/`` leaf (and raises a clear
error listing the available leaves when the requested one is missing). There is
no per-device meta scan or ``[Solvers]`` handling in the shim anymore -- solver
config rides in ``metadata.json`` and is applied by the C++ ctor.

These tests drive the resolution branches with crafted stubs + empty/odd
artifact folders; none reach a real AOTI compile. The happy path is covered by
``tests/aoti``'s stub-loading test.
"""

from __future__ import annotations

import json

import pytest
import torch

from neml2 import load_input
from neml2.aoti._shim import AOTIModel


def _write_stub(tmp_path, artifact_path_value: str):
    stub = tmp_path / "model_aoti.i"
    stub.write_text(
        "[Models]\n"
        "  [model]\n"
        "    type = AOTIModel\n"
        f"    artifact_path = '{artifact_path_value}'\n"
        "  []\n"
        "[]\n"
    )
    return stub


def _default_leaf() -> tuple[str, str]:
    """The (device, dtype) subfolder names for the ambient torch defaults."""
    device = torch.get_default_device().type
    dtype = str(torch.get_default_dtype()).removeprefix("torch.")
    return device, dtype


def _write_metadata(root) -> None:
    """A minimal self-describing metadata.json at the artifact root (schema v10)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(
        json.dumps({"schema_version": 10, "type": "composed", "inputs": [], "outputs": []})
    )


def test_missing_metadata_errors_before_binding(tmp_path):
    # Artifact folder exists but has no metadata.json at its root: the shim's
    # __init__ raises FileNotFoundError before it ever reaches the C++ binding.
    root = tmp_path / "model"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no metadata.json"):
        AOTIModel(root)


def test_missing_metadata_via_hit_errors(tmp_path):
    # Same, but through the HIT `from_hit` path: an absolute artifact_path at a
    # folder with no metadata.json surfaces the same FileNotFoundError.
    (tmp_path / "model").mkdir()
    stub = _write_stub(tmp_path, str(tmp_path / "model"))
    with pytest.raises(FileNotFoundError, match="no metadata.json"):
        load_input(str(stub)).get_model("model")


def test_missing_device_dtype_leaf_errors(tmp_path):
    # metadata.json is present (so __init__ passes), but there is no
    # <device>/<dtype>/ leaf for the ambient torch defaults -> the C++ ctor
    # raises, listing what IS available (here: none).
    root = tmp_path / "model"
    _write_metadata(root)
    stub = _write_stub(tmp_path, str(root))
    with pytest.raises(RuntimeError, match="no compiled artifact for device"):
        load_input(str(stub)).get_model("model")


def test_relative_artifact_path_resolved_against_input_dir(tmp_path):
    # A relative artifact_path resolves against the stub's directory. The
    # resolved folder has metadata.json but no matching leaf, so we reach (and
    # surface) the C++ ctor's leaf-not-found error -- proving the relative path
    # resolved to the right place.
    _write_metadata(tmp_path / "model")
    stub = _write_stub(tmp_path, "./model")
    with pytest.raises(RuntimeError, match="no compiled artifact for device"):
        load_input(str(stub)).get_model("model")


def test_leaf_not_found_message_lists_available_leaves(tmp_path):
    # When a sibling leaf exists but not the requested (device, dtype), the
    # error names the compiled leaves so the user knows what to run instead.
    device, dtype = _default_leaf()
    root = tmp_path / "model"
    _write_metadata(root)
    # Create a decoy leaf under a bogus dtype so the "Available:" list is
    # non-empty and distinct from the requested (device, dtype).
    decoy_dtype = "float32" if dtype != "float32" else "float64"
    (root / device / decoy_dtype).mkdir(parents=True)
    with pytest.raises(RuntimeError) as excinfo:
        AOTIModel(root)
    msg = str(excinfo.value)
    assert "no compiled artifact for device" in msg
    assert f"{device}/{decoy_dtype}" in msg  # the available leaf is reported
