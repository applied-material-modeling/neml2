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

"""Schema-version handshake + sub-batch / sparsity metadata emission."""

from __future__ import annotations

import pytest

from neml2.cli.aoti_export import (
    AOTI_META_SCHEMA_VERSION,
    _sparsity_for_outputs,
    _var_infos,
)
from neml2.model import Model
from neml2.types import Scalar

# ---------- schema_version constant ----------


def test_schema_version_constant_is_int_two():
    """The supported version is `2` (bake-by-default + promoted parameters);
    C++ side mirrors this."""
    assert AOTI_META_SCHEMA_VERSION == 2


# ---------- _var_infos default behaviour (no sub-batch, no sparsity) ----------


def test_var_infos_default_emits_name_size_and_type():
    spec = {"x": Scalar}
    infos = _var_infos(spec)
    assert infos == [{"name": "x", "var_size": 1, "var_type": "Scalar"}]


def test_var_infos_omits_empty_sub_batch_shape():
    spec = {"x": Scalar, "y": Scalar}
    infos = _var_infos(spec, sub_batch_shapes={"x": (), "y": (3,)})
    assert infos == [
        {"name": "x", "var_size": 1, "var_type": "Scalar"},
        {"name": "y", "var_size": 1, "var_type": "Scalar", "sub_batch_shape": [3]},
    ]


def test_var_infos_sparsity_only_emitted_for_explicit_outputs():
    spec = {"y_dense": Scalar, "y_diag": Scalar}
    infos = _var_infos(spec, sparsity={"y_dense": {"x": "dense"}})
    assert infos == [
        {"name": "y_dense", "var_size": 1, "var_type": "Scalar", "sparsity": {"x": "dense"}},
        {"name": "y_diag", "var_size": 1, "var_type": "Scalar"},
    ]


# ---------- _sparsity_for_outputs from Model.list_deriv ----------


class _DenseReduce(Model):
    list_deriv = {("y", "x"): "dense"}
    input_spec = {"x": Scalar}
    output_spec = {"y": Scalar}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (x,) = inputs
        y = Scalar(x.data.sum(dim=-1))
        if v is None:
            return y
        return y, self.apply_chain_rule(v, "y", {"x": lambda V: V.sum(dim=-3)})


def test_sparsity_for_outputs_reads_model_list_deriv():
    m = _DenseReduce()
    s = _sparsity_for_outputs(m, ("y",), ("x",))
    assert s == {"y": {"x": "dense"}}


def test_sparsity_for_outputs_silent_for_diagonal_only_models():
    class _Diag(Model):
        input_spec = {"x": Scalar}
        output_spec = {"y": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            (x,) = inputs
            return Scalar(2.0 * x.data)

    assert _sparsity_for_outputs(_Diag(), ("y",), ("x",)) == {}


def test_sparsity_for_outputs_filters_to_requested_io():
    """Pairs naming variables not in the requested I/O are dropped."""

    class _MultiEdge(Model):
        list_deriv = {
            ("y1", "x1"): "dense",
            ("y2", "x2"): "dense",
        }
        input_spec = {"x1": Scalar, "x2": Scalar}
        output_spec = {"y1": Scalar, "y2": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            x1, x2 = inputs
            return Scalar(x1.data), Scalar(x2.data)

    m = _MultiEdge()
    # Restrict to y1 only — y2 pair drops out.
    s = _sparsity_for_outputs(m, ("y1",), ("x1", "x2"))
    assert s == {"y1": {"x1": "dense"}}
    # Restrict to x1 only — y2's pair drops because x2 isn't in inputs.
    s = _sparsity_for_outputs(m, ("y1", "y2"), ("x1",))
    assert s == {"y1": {"x1": "dense"}}


# ---------- Integration: a real export emits the new fields ----------


@pytest.mark.aoti_compile
def test_exported_metadata_carries_schema_version(tmp_path):
    """``export_model_for_aoti`` writes `schema_version` to the JSON sidecar."""
    import json

    from neml2.cli.aoti_export import export_model_for_aoti

    # Use the smallest possible HIT that round-trips: a stand-alone Scalar
    # LinearIsotropicElasticity-like leaf. The simplest existing fixture is
    # in tests/aoti/.
    hit = """
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
    hit_path = tmp_path / "elastic.i"
    hit_path.write_text(hit)
    out_dir = tmp_path / "aoti"

    meta = export_model_for_aoti(hit_path, "model", out_dir, device="cpu")
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION

    # And the sidecar JSON has it on disk too.
    with open(out_dir / "model_meta.json") as f:
        on_disk = json.load(f)
    assert on_disk["schema_version"] == AOTI_META_SCHEMA_VERSION


@pytest.mark.aoti_compile
def test_exported_metadata_omits_sparsity_for_diagonal_models(tmp_path):
    """Diagonal-only models don't pollute their JSON with sparsity fields."""
    import json

    from neml2.cli.aoti_export import export_model_for_aoti

    hit = """
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
    hit_path = tmp_path / "elastic2.i"
    hit_path.write_text(hit)
    out_dir = tmp_path / "aoti2"

    export_model_for_aoti(hit_path, "model", out_dir, device="cpu")
    with open(out_dir / "model_meta.json") as f:
        meta = json.load(f)
    for out_info in meta["outputs"]:
        assert "sparsity" not in out_info
        assert "sub_batch_shape" not in out_info
    for in_info in meta["inputs"]:
        assert "sub_batch_shape" not in in_info
        assert "sparsity" not in in_info
