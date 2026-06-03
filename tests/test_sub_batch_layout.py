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

"""AxisLayout sub-batch shape + IStructure + SchurComplement registration coverage."""

from __future__ import annotations

import pytest
import torch

from neml2.equation_systems import (
    AssembledVector,
    AxisLayout,
    IStructure,
)
from neml2.solvers import DenseLU, Newton, SchurComplement
from neml2.types import SR2, Scalar

# ---------- AxisLayout default (no sub-batch) ----------


def test_default_layout_is_dense_and_block_size_equals_storage():
    layout = AxisLayout([["x", "s"]], {"x": Scalar, "s": SR2})
    assert layout.istructure == IStructure.DENSE
    assert layout.storage_size() == 1 + 6
    assert layout.block_size() == 1 + 6
    assert layout.sub_batch_shape("x") == torch.Size([])
    assert layout.sub_batch_shape("s") == torch.Size([])


def test_layout_keeps_default_when_sub_batch_shapes_omitted():
    layout = AxisLayout([["a", "b"]], {"a": Scalar, "b": Scalar})
    # Equivalent layout via the constructor with an explicit empty dict.
    same = AxisLayout([["a", "b"]], {"a": Scalar, "b": Scalar}, sub_batch_shapes={})
    assert layout == same


# ---------- BLOCK detection ----------


def test_uniform_sub_batch_promotes_layout_to_block():
    """All vars sharing the same non-trivial sub-batch shape → BLOCK."""
    layout = AxisLayout(
        [["R", "N"]],
        {"R": Scalar, "N": Scalar},
        sub_batch_shapes={"R": torch.Size([100]), "N": torch.Size([100])},
    )
    assert layout.istructure == IStructure.BLOCK
    assert layout.sub_batch_shape("R") == torch.Size([100])
    assert layout.sub_batch_shape("N") == torch.Size([100])


def test_multi_dim_sub_batch_block_detection():
    layout = AxisLayout(
        [["x", "y"]],
        {"x": Scalar, "y": Scalar},
        sub_batch_shapes={"x": torch.Size([4, 5]), "y": torch.Size([4, 5])},
    )
    assert layout.istructure == IStructure.BLOCK


def test_mismatched_sub_batch_falls_back_to_dense():
    """Different sub-batch shapes across vars ⇒ DENSE."""
    layout = AxisLayout(
        [["x", "y"]],
        {"x": Scalar, "y": Scalar},
        sub_batch_shapes={"x": torch.Size([100]), "y": torch.Size([50])},
    )
    assert layout.istructure == IStructure.DENSE


def test_partial_sub_batch_falls_back_to_dense():
    """Some vars sub-batch, some not ⇒ DENSE."""
    layout = AxisLayout(
        [["a", "b"]],
        {"a": Scalar, "b": Scalar},
        sub_batch_shapes={"a": torch.Size([10])},  # b defaults to ()
    )
    assert layout.istructure == IStructure.DENSE


def test_explicit_istructure_overrides_inference():
    """A model with mixed sub-batch but known-diagonal cross-derivatives
    can declare BLOCK explicitly."""
    layout = AxisLayout(
        [["x"]],
        {"x": Scalar},
        sub_batch_shapes={"x": torch.Size([7])},
        istructure=IStructure.BLOCK,
    )
    assert layout.istructure == IStructure.BLOCK
    # And the override is preserved across with_sub_batch_shapes if
    # explicitly re-passed; the default inference path runs otherwise.
    rewrap = layout.with_sub_batch_shapes({"x": torch.Size([7])})
    assert rewrap.istructure == IStructure.BLOCK
    forced = layout.with_sub_batch_shapes({"x": torch.Size([7])}, istructure=IStructure.DENSE)
    assert forced.istructure == IStructure.DENSE


# ---------- with_sub_batch_shapes immutability ----------


def test_with_sub_batch_shapes_returns_new_layout_leaving_original_unchanged():
    layout = AxisLayout([["x"]], {"x": Scalar})
    assert layout.istructure == IStructure.DENSE
    rewrap = layout.with_sub_batch_shapes({"x": torch.Size([8])})
    assert layout.sub_batch_shape("x") == torch.Size([])
    assert layout.istructure == IStructure.DENSE
    assert rewrap.sub_batch_shape("x") == torch.Size([8])
    assert rewrap.istructure == IStructure.BLOCK


# ---------- AssembledVector roundtrip is unchanged by sub-batch metadata ----------


def test_assembled_vector_roundtrip_sub_batch_aware():
    """An AssembledVector built from per-site tensors flattens (sub_batch × base)
    into a single trailing dim and ``disassemble`` recovers the per-site shape.

    Per D-052: AssembledVector.from_dict packs ``(*dyn, *sub_batch, *base)`` into
    ``(*dyn, prod(sub_batch) * base_size)`` so the assembled vector shape is
    uniform per group regardless of variable sub-batch ndim. For Scalar
    (base_size=1) and sub_batch=(3,), the trailing axis is 3.
    """
    layout = AxisLayout(
        [["x"]],
        {"x": Scalar},
        sub_batch_shapes={"x": torch.Size([3])},
    )
    values = {"x": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)}
    assembled = AssembledVector.from_dict(layout, values)
    assert assembled.tensors[0].shape == torch.Size([2, 3])
    rt = assembled.disassemble()
    assert torch.equal(rt["x"], values["x"])


def test_assembled_vector_add_requires_matching_sub_batch_metadata():
    """Two AssembledVectors with mismatched sub-batch shapes can't be added."""
    layout_a = AxisLayout(
        [["x"]],
        {"x": Scalar},
        sub_batch_shapes={"x": torch.Size([3])},
    )
    layout_b = AxisLayout(
        [["x"]],
        {"x": Scalar},
        sub_batch_shapes={"x": torch.Size([5])},
    )
    values_a = {"x": torch.zeros(3, dtype=torch.float64)}
    values_b = {"x": torch.zeros(5, dtype=torch.float64)}
    va = AssembledVector.from_dict(layout_a, values_a)
    vb = AssembledVector.from_dict(layout_b, values_b)
    with pytest.raises(ValueError, match="different layouts"):
        _ = va + vb


# ---------- IStructure enum integration ----------


def test_istructure_is_string_friendly_enum():
    """The enum surfaces its values for HIT / error messages."""
    assert IStructure.DENSE.name == "DENSE"
    assert IStructure.BLOCK.name == "BLOCK"
    assert IStructure.DENSE.value == "dense"
    assert IStructure.BLOCK.value == "block"


# ---------- SchurComplement basic construction ----------


def test_schur_complement_constructs_with_default_sub_solvers():
    """SchurComplement() picks DenseLU for both sub-solvers (matches Newton's default).

    HIT inputs still spell `primary_solver`/`schur_solver` explicitly — the
    C++ side has no defaults — but programmatic Python construction stays
    terse for the dominant case. See test_schur.py for the deeper coverage.
    """
    s = SchurComplement()
    assert isinstance(s, SchurComplement)


def test_schur_complement_rejects_non_two_group_input():
    """A single-group A is rejected with a clear error."""
    from neml2.equation_systems import AssembledMatrix

    s = SchurComplement()
    layout = AxisLayout([["x"]], {"x": Scalar})
    A = AssembledMatrix(layout, layout, [[torch.eye(1).reshape(1, 1)]])
    b_vec = AssembledVector.from_dict(layout, {"x": torch.zeros(1, dtype=torch.float64)})
    with pytest.raises(ValueError, match="exactly 2 row groups"):
        s.solve(A, b_vec)


# ---------- Newton accepts either solver ----------


def test_newton_accepts_dense_lu_default():
    newton = Newton()
    assert isinstance(newton.linear_solver, DenseLU)


def test_newton_accepts_schur_complement():
    newton = Newton(linear_solver=SchurComplement())
    assert isinstance(newton.linear_solver, SchurComplement)


# ---------- DenseLU still handles sub-batch via leading-dim broadcast ----------


def test_dense_lu_solves_block_diagonal_system_via_broadcast():
    """A (*B, L, n, n) system with L blocks is solved as L independent dense LUs."""
    from neml2.equation_systems import AssembledMatrix

    layout = AxisLayout(
        [["x"]],
        {"x": Scalar},
        sub_batch_shapes={"x": torch.Size([4])},
    )
    assert layout.istructure == IStructure.BLOCK
    # Diagonal scaling matrix per site: A[b, L, 0, 0] = L+1 ⇒ x[L] = b[L] / (L+1)
    diag = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    A_block = diag.reshape(1, 4, 1, 1)
    b_block = torch.tensor([[2.0, 4.0, 9.0, 16.0]], dtype=torch.float64).reshape(1, 4, 1)
    A = AssembledMatrix(layout, layout, [[A_block]])
    b_vec = AssembledVector.from_dict(layout, {"x": b_block.reshape(1, 4)})
    x = DenseLU().solve(A, b_vec)
    assert isinstance(x, AssembledVector)
    # Expected x[L] = b[L] / (L+1)
    expected = (b_block.reshape(1, 4) / diag).reshape(1, 4)
    assert torch.allclose(x.tensors[0].reshape(1, 4), expected)


# ---------- Factory parsing ----------


def test_schur_complement_registers_through_native_factory():
    """``type = SchurComplement`` in HIT should construct via the registry."""
    from neml2.factory import _registry  # noqa: PLC2701 — testing registration

    assert _registry["SchurComplement"] is SchurComplement
