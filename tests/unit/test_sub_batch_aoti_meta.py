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

"""Schema-version handshake + sub-batch metadata emission.

Pins the structural contract between the Python writer
(``neml2.cli.aoti_export``) and the C++ loader
(``neml2/csrc/aoti/Model.cpp``):

* ``schema_version`` is the boundary the C++ side strictly checks at load
  time -- any breaking change to the JSON shape bumps this constant on
  both sides.
* The master-level ``inputs`` / ``outputs`` carry ``name`` + ``var_size``
  + ``var_type`` (and opt-in ``sub_batch_shape``); the Python shim's
  ``input_spec`` / ``output_spec`` reconstruction relies on them.
* Forward-segment per-input/output metadata collapses to ``name`` only --
  C++ derives per-output flat sizes from the live AOTI tensor shape
  rather than carrying them in JSON. Confirmed by a real export.
"""

from __future__ import annotations

from pathlib import Path

from neml2.cli.aoti_export import AOTI_META_SCHEMA_VERSION, _var_infos
from neml2.types import Scalar

# ---------- schema_version constant ----------


def test_schema_version_matches_dependencies_yaml():
    """The exported constant tracks the single source of truth in
    ``scripts/dependencies.yaml`` (kept in sync by ``scripts/dep_manager.py``);
    the C++ loader mirrors the same value and refuses any other, so a stale
    cache surfaces with a clear ``regenerate via neml2-compile`` message.

    Parsed with a regex rather than PyYAML: ``yaml`` is a build/dev-tool
    dependency (``dep_manager``), not part of the runtime test environment.
    ``aoti.schema_version`` is the only ``schema_version`` key in the file."""
    import re

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "scripts" / "dependencies.yaml").read_text()
    m = re.search(r'^\s*schema_version:\s*"?(\d+)"?', text, re.MULTILINE)
    assert m is not None, "schema_version not found in scripts/dependencies.yaml"
    assert AOTI_META_SCHEMA_VERSION == int(m.group(1))


# ---------- _var_infos default behaviour (no sub-batch) ----------


def test_var_infos_default_emits_name_size_type_and_base_shape():
    spec = {"x": Scalar}
    infos = _var_infos(spec)
    assert infos == [{"name": "x", "var_size": 1, "var_type": "Scalar", "base_shape": []}]


def test_var_infos_omits_empty_sub_batch_shape():
    spec = {"x": Scalar, "y": Scalar}
    infos = _var_infos(spec, sub_batch_shapes={"x": (), "y": (3,)})
    assert infos == [
        {"name": "x", "var_size": 1, "var_type": "Scalar", "base_shape": []},
        {
            "name": "y",
            "var_size": 1,
            "var_type": "Scalar",
            "base_shape": [],
            "sub_batch_shape": [3],
        },
    ]


# ---------- Integration: a real export emits the new fields ----------


_ELASTIC_HIT = """
[Tensors]
  [E]
    type = Python
    expr = 'Scalar(torch.tensor(2e5, dtype=torch.float64))'
  []
  [nu]
    type = Python
    expr = 'Scalar(torch.tensor(0.3, dtype=torch.float64))'
  []
[]
[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = 'E nu'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
"""


def test_exported_metadata_carries_schema_version(tmp_path):
    """``export_model_for_aoti`` writes `schema_version` to the JSON sidecar."""
    import json

    from neml2.cli.aoti_export import export_model_for_aoti

    hit_path = tmp_path / "elastic.i"
    hit_path.write_text(_ELASTIC_HIT)
    out_dir = tmp_path / "aoti"

    meta = export_model_for_aoti(hit_path, "model", out_dir, device="cpu")
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION

    # Schema v10: out_dir is the artifact ROOT -- a single shared metadata.json
    # sits there (no per-model name, no device/dtype in the envelope), while the
    # .pt2 binaries land under <device>/<dtype>/.
    assert "device" not in meta
    assert "dtype" not in meta
    assert (out_dir / "cpu" / "float64" / "model.pt2").exists()
    with open(out_dir / "metadata.json") as f:
        on_disk = json.load(f)
    assert on_disk["schema_version"] == AOTI_META_SCHEMA_VERSION


def test_forward_segment_metadata_carries_only_names(tmp_path):
    """Forward-segment per-input/output dicts collapse to ``{name}`` only.

    Master-level ``inputs`` / ``outputs`` still carry ``var_size`` /
    ``var_type`` because the Python shim reads them to rebuild
    ``input_spec`` / ``output_spec``.
    """
    import json

    from neml2.cli.aoti_export import export_model_for_aoti

    hit_path = tmp_path / "elastic.i"
    hit_path.write_text(_ELASTIC_HIT)
    out_dir = tmp_path / "aoti"

    export_model_for_aoti(hit_path, "model", out_dir, device="cpu")
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)

    # Master level keeps the full info, including the v5 base_shape.
    for info in (*meta["inputs"], *meta["outputs"]):
        assert set(info) >= {"name", "var_size", "var_type", "base_shape"}
        assert "sparsity" not in info

    # Segment level: only ``name``. ``param_inputs`` survives at the
    # segment-dict top level; here we only check the inputs/outputs lists.
    for seg in meta["segments"]:
        if seg["kind"] != "forward":
            continue
        for info in (*seg["inputs"], *seg["outputs"]):
            assert set(info) == {"name"}
