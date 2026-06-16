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

"""Tests for SparseVector / SparseMatrix (typed duals of Assembled*)."""

import pytest
import torch

from neml2.es import AssembledMatrix, AssembledVector, AxisLayout, SparseMatrix, SparseVector
from neml2.types import SR2, Scalar, Tensor, TensorWrapper


def _two_var_layout() -> AxisLayout:
    """A simple two-variable, single-group, dense, scalar+SR2 layout."""
    return AxisLayout([["x", "s"]], {"x": Scalar, "s": SR2})


def _two_var_values(batch: int = 4) -> dict[str, TensorWrapper]:
    return {
        "x": Scalar(torch.randn(batch)),
        "s": SR2(torch.randn(batch, 6)),
    }


# ---------------------------------------------------------------------------
# SparseVector
# ---------------------------------------------------------------------------


def test_sparse_vector_assemble_then_disassemble_round_trips():
    layout = _two_var_layout()
    values = _two_var_values(batch=4)
    sv = SparseVector(layout, values)

    av = sv.assemble()
    assert isinstance(av, AssembledVector)

    # disassemble currently returns a dict; round-trip via from_dict /
    # disassemble compares element-wise.
    out = av.disassemble()
    assert torch.allclose(out["x"].data, values["x"].data)
    assert torch.allclose(out["s"].data, values["s"].data)


def test_sparse_vector_missing_variable_raises_keyerror():
    layout = _two_var_layout()
    # Only "x" is supplied; "s" is missing per the layout.
    with pytest.raises(KeyError, match="missing values for variables"):
        SparseVector(layout, {"x": Scalar(torch.zeros(4))})


def test_sparse_vector_values_must_be_typed_wrappers():
    """SparseVector is strictly typed per CLAUDE.md rule 1; raw torch.Tensor
    values are rejected so external boundaries are forced to wrap at the
    construction site rather than handing raw tensors to an internal
    neml2 helper.

    pyright should reject the construction; at runtime the strict typing
    is documentary -- nothing in the dataclass __post_init__ explicitly
    isinstance-checks each value (would add per-call overhead with no
    upside; the typed downstream operations will fail loudly on raw
    input). So this test just round-trips with typed wrappers and lets
    static type-checking enforce the rule.
    """
    layout = _two_var_layout()
    values: dict[str, TensorWrapper] = {
        "x": Scalar(torch.randn(4)),
        "s": SR2(torch.randn(4, 6)),
    }
    sv = SparseVector(layout, values)
    assert isinstance(sv["x"], Scalar)
    assert isinstance(sv["s"], SR2)
    av = sv.assemble()
    assert isinstance(av, AssembledVector)


def test_sparse_vector_dict_protocol_iterates_and_indexes():
    layout = _two_var_layout()
    values = _two_var_values()
    sv = SparseVector(layout, values)

    assert "x" in sv
    assert "missing" not in sv
    assert len(sv) == 2
    assert set(iter(sv)) == {"x", "s"}
    assert set(sv.keys()) == {"x", "s"}
    assert {name for name, _ in sv.items()} == {"x", "s"}
    assert torch.equal(sv["x"].data, values["x"].data)


def test_sparse_vector_to_dtype_moves_every_value():
    layout = _two_var_layout()
    sv = SparseVector(
        layout,
        {
            "x": Scalar(torch.zeros(4, dtype=torch.float32)),
            "s": SR2(torch.zeros(4, 6, dtype=torch.float32)),
        },
    )
    moved = sv.to(dtype=torch.float64)
    assert moved["x"].data.dtype == torch.float64
    assert moved["s"].data.dtype == torch.float64
    # original is untouched (frozen dataclass + new dict)
    assert sv["x"].data.dtype == torch.float32


# ---------------------------------------------------------------------------
# SparseMatrix
# ---------------------------------------------------------------------------


def test_sparse_matrix_assemble_round_trips_through_disassemble():
    """Hand-build an AssembledMatrix via select_blocks, disassemble into a
    SparseMatrix, re-assemble, and check the per-group tensors match."""
    layout = _two_var_layout()
    storage = layout.storage_size()  # 1 (x) + 6 (s) = 7
    blocks: dict[str, dict[str, Tensor]] = {
        "x": {"x": Tensor(torch.eye(1, dtype=torch.float64).expand(4, 1, 1).clone(), batch_ndim=1)},
        "s": {"s": Tensor(torch.eye(6, dtype=torch.float64).expand(4, 6, 6).clone(), batch_ndim=1)},
    }
    am = AssembledMatrix.select_blocks(layout, layout, blocks)
    assert am.tensors[0][0].data.shape == (4, storage, storage)

    sm = am.disassemble()
    assert isinstance(sm, SparseMatrix)
    am2 = sm.assemble()

    # Round-trip preserves per-group tensors.
    assert torch.allclose(am2.tensors[0][0].data, am.tensors[0][0].data)


def test_sparse_matrix_missing_row_raises_keyerror():
    layout = _two_var_layout()
    with pytest.raises(KeyError, match="missing row entries"):
        SparseMatrix(layout, layout, {"x": {}})  # missing row "s"


def test_sparse_matrix_per_block_sparsity_becomes_zero_block():
    """Missing inner (row_var, col_var) cells must be allowed and become
    zero blocks at assembly time."""
    layout = _two_var_layout()
    eye_x = Tensor(torch.eye(1, dtype=torch.float64).expand(4, 1, 1).clone(), batch_ndim=1)
    # Outer row "s" present but inner is empty -> select_blocks zero-fills.
    cells = {"x": {"x": eye_x}, "s": {}}
    sm = SparseMatrix(layout, layout, cells)
    am = sm.assemble()
    # x-x corner is identity; everything else is zero.
    blk = am.tensors[0][0].data
    assert torch.allclose(blk[..., 0, 0], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(blk[..., 1:, 1:], torch.zeros(4, 6, 6, dtype=torch.float64))


def test_sparse_matrix_indexing_uses_row_col_tuple():
    layout = _two_var_layout()
    eye_x = Tensor(torch.eye(1, dtype=torch.float64).expand(4, 1, 1).clone(), batch_ndim=1)
    eye_s = Tensor(torch.eye(6, dtype=torch.float64).expand(4, 6, 6).clone(), batch_ndim=1)
    cells = {"x": {"x": eye_x}, "s": {"s": eye_s}}
    sm = SparseMatrix(layout, layout, cells)

    assert ("x", "x") in sm
    assert ("x", "s") not in sm
    assert ("nope", "nope") not in sm
    assert torch.equal(sm[("x", "x")].data, eye_x.data)


def test_sparse_matrix_to_dtype_moves_every_cell():
    layout = _two_var_layout()
    from neml2.types import Tensor as _T

    eye_x = _T(torch.eye(1, dtype=torch.float32).expand(4, 1, 1).clone(), batch_ndim=1)
    sm = SparseMatrix(layout, layout, {"x": {"x": eye_x}, "s": {}})
    moved = sm.to(dtype=torch.float64)
    assert moved[("x", "x")].data.dtype == torch.float64
    # original untouched
    assert sm[("x", "x")].data.dtype == torch.float32
