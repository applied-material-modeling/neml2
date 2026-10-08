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

"""Direct-construction coverage for the assembled vector/matrix/layout fold machinery.

These pin the assembled-object operations (``to_flat``/``from_flat`` round-trips, group
arithmetic, batch indexing, norms, and the matrix disassemble/transpose/matmul/arithmetic)
and the :class:`~neml2.es.axis_layout.AxisLayout` sizing helpers for a mixed BLOCK group
(a shared common sub-batch prefix + a folded extra axis) alongside a DENSE group (whole
sub-batch folded into the base) -- the plumbing the Schur/pyzag solvers walk -- without
needing a full solve. Companion to ``test_subbatch_fold.py`` (which pins the block-group
vector fold) and the regression solves (which drive the tangent converters end to end).
"""

from __future__ import annotations

import torch

from neml2.es.assembled import AssembledMatrix, AssembledVector, _common_prefix, norm, norm_sq
from neml2.es.axis_layout import AxisLayout
from neml2.types import SR2, Scalar, Tensor


def _mixed_layout(ng=2, ns=3):
    """Group 0 BLOCK {common grain, grain x slip}, group 1 DENSE {per-slip scalar}."""
    return AxisLayout(
        [["u_grain", "u_slip"], ["d_slip"]],
        {"u_grain": Scalar, "u_slip": Scalar, "d_slip": Scalar},
        sub_batch_shapes={
            "u_grain": torch.Size([ng]),
            "u_slip": torch.Size([ng, ns]),
            "d_slip": torch.Size([ns]),
        },
        structure=["block", "dense"],
    )


def _sample_vector(nb=2, ng=2, ns=3):
    layout = _mixed_layout(ng, ns)
    torch.manual_seed(0)
    values = {
        "u_grain": Scalar(torch.randn(nb, ng, dtype=torch.float64), 1).with_sub_batch_ndim(1),
        "u_slip": Scalar(torch.randn(nb, ng, ns, dtype=torch.float64), 1).with_sub_batch_ndim(2),
        "d_slip": Scalar(torch.randn(nb, ns, dtype=torch.float64), 1).with_sub_batch_ndim(1),
    }
    return layout, AssembledVector.from_dict(layout, values), values


# ---------------------------------------------------------------- AxisLayout


def test_axis_layout_sizing_helpers():
    """Group/var sizing helpers for a mixed BLOCK(+extra) / DENSE layout."""
    ng, ns = 2, 3
    layout = _mixed_layout(ng, ns)
    assert layout.ngroup == 2
    assert layout.nvar == 3
    assert layout.vars() == ("u_grain", "u_slip", "d_slip")
    # BLOCK group: common = grain, u_slip carries the extra slip axis.
    assert tuple(layout.group_common_sub_batch(0)) == (ng,)
    assert tuple(layout.var_extra_sub_batch(0, "u_slip")) == (ns,)
    assert tuple(layout.var_extra_sub_batch(0, "u_grain")) == ()
    assert tuple(layout.group_sub_batch_shape(0)) == (ng,)
    # DENSE group: no preserved intermediate; its sub-batch folds into the base.
    assert tuple(layout.group_common_sub_batch(1)) == ()
    assert layout.var_size("u_grain") == 1
    # group_flat_size: block grain*(grain-base + slip-folded); dense slip folded into base.
    assert layout.group_flat_size(0) == ng * (1 + ns)
    assert layout.group_flat_size(1) == ns


def test_chain_rule_equalize_and_matvec_edges():
    """equalize_tangent_K edge paths (empty, mixed k_ndim, k=0 passthrough, bad size)
    and the matvec helper."""
    import pytest

    from neml2.models.chain_rule import equalize_tangent_K, matvec

    # Empty list -> returned as-is.
    assert equalize_tangent_K([]) == []

    # A k_ndim==0 contribution alongside a differing full-K one: the scalar is
    # passed through, the K one is unchanged (already max).
    c0 = Scalar(torch.ones(2, dtype=torch.float64))  # k_ndim=0
    c1 = Scalar(torch.arange(3.0), k_ndim=1, k_state=("full",), k_pairing=(None,))
    out = equalize_tangent_K([c0, c1])
    assert out[0].k_ndim == 0
    assert tuple(out[1].data.shape[:1]) == (3,)

    # Contributions of DIFFERENT k_ndim extend max_k with the extra axis.
    a = Scalar(torch.randn(3, dtype=torch.float64), k_ndim=1, k_state=("full",), k_pairing=(None,))
    b = Scalar(
        torch.randn(3, 2, dtype=torch.float64),
        k_ndim=2,
        k_state=("full", "full"),
        k_pairing=(None, None),
    )
    out2 = equalize_tangent_K([a, b])
    assert tuple(out2[1].data.shape[:2]) == (3, 2)

    # Incompatible full-K storage sizes (neither is 1) must raise.
    big = Scalar(
        torch.randn(3, dtype=torch.float64), k_ndim=1, k_state=("full",), k_pairing=(None,)
    )
    bad = Scalar(
        torch.randn(2, dtype=torch.float64), k_ndim=1, k_state=("full",), k_pairing=(None,)
    )
    with pytest.raises(ValueError, match="incompatible storage"):
        equalize_tangent_K([big, bad])

    # matvec == batched (M @ v) on raw torch tensors.
    mt = torch.randn(2, 3, 4, dtype=torch.float64)
    vt = torch.randn(2, 4, dtype=torch.float64)
    got = matvec(mt, vt)
    torch.testing.assert_close(got, torch.einsum("bij,bj->bi", mt, vt))


def test_axis_layout_validation_and_misc_helpers():
    """Constructor validation plus the small sizing helpers on both structures."""
    import pytest

    # Missing var from specs.
    with pytest.raises(KeyError, match="missing from specs"):
        AxisLayout([["a"]], {"b": Scalar})
    # structure length mismatch.
    with pytest.raises(ValueError, match="one per group"):
        AxisLayout([["a"], ["b"]], {"a": Scalar, "b": Scalar}, structure=["dense"])
    # invalid structure entry.
    with pytest.raises(ValueError, match="'block' or 'dense'"):
        AxisLayout([["a"]], {"a": Scalar}, structure=["sparse"])  # type: ignore[list-item]
    # BLOCK group mixing a sub-batch-trivial var with a sub-batched one (nc==0 guard).
    bad = AxisLayout(
        [["a", "b"]],
        {"a": Scalar, "b": Scalar},
        sub_batch_shapes={"a": torch.Size(()), "b": torch.Size([3])},
        structure=["block"],
    )
    with pytest.raises(ValueError, match="sub-batch-trivial"):
        bad.group_common_sub_batch(0)

    layout = _mixed_layout(ng=2, ns=3)
    # var_extra_sub_batch on the DENSE group is empty.
    assert tuple(layout.var_extra_sub_batch(1, "d_slip")) == ()
    # block_size == storage_size (base-only).
    assert layout.block_size() == layout.storage_size()
    # group_intmd_ndim: BLOCK group carries its common axis, DENSE folds into base.
    assert layout.group_intmd_ndim(0) == 1
    assert layout.group_intmd_ndim(1) == 0


# ---------------------------------------------------------------- AssembledVector


def test_vector_from_dict_disassemble_roundtrip_mixed_groups():
    """BLOCK(+extra) and DENSE groups both round-trip through from_dict/disassemble."""
    _, vec, values = _sample_vector()
    back = vec.disassemble().values
    for name in ("u_grain", "u_slip", "d_slip"):
        got = back[name].data.reshape(values[name].data.shape)
        torch.testing.assert_close(got, values[name].data, rtol=0, atol=0)


def test_vector_to_flat_from_flat_roundtrip():
    """to_flat packs every group to one feature axis; from_flat recovers the groups."""
    layout, vec, _ = _sample_vector()
    flat = vec.to_flat()
    folded_width = sum(layout.group_flat_size(gi) for gi in range(layout.ngroup))
    assert flat.shape[-1] == folded_width
    rebuilt = AssembledVector.from_flat(layout, flat)
    for a, b in zip(vec.tensors, rebuilt.tensors, strict=True):
        torch.testing.assert_close(a.data, b.data, rtol=0, atol=0)
        assert a.sub_batch_ndim == b.sub_batch_ndim


def test_vector_arithmetic_and_group_and_batch():
    """__neg__ / __add__ / __sub__, single-group extraction, and batch slicing."""
    layout, vec, _ = _sample_vector(nb=4)
    doubled = vec + vec
    zero = vec - vec
    neg = -vec
    for gi in range(layout.ngroup):
        torch.testing.assert_close(doubled.tensors[gi].data, 2.0 * vec.tensors[gi].data)
        torch.testing.assert_close(zero.tensors[gi].data, torch.zeros_like(vec.tensors[gi].data))
        torch.testing.assert_close(neg.tensors[gi].data, -vec.tensors[gi].data)
    g0 = vec.group(0)
    assert g0.layout.ngroup == 1
    sliced = vec.batch[:2]
    assert sliced.tensors[0].data.shape[0] == 2


def test_vector_norm_matches_manual():
    """norm / norm_sq sum squares over base (and the BLOCK common sub-batch)."""
    _, vec, values = _sample_vector()
    terms = [
        (values[n].data ** 2).sum(dim=tuple(range(1, values[n].data.ndim)))
        for n in ("u_grain", "u_slip", "d_slip")
    ]
    expected = torch.stack(terms).sum(dim=0)
    torch.testing.assert_close(norm_sq(vec), expected, rtol=0, atol=1e-10)
    torch.testing.assert_close(norm(vec), torch.sqrt(expected), rtol=0, atol=1e-10)


# ---------------------------------------------------------------- AssembledMatrix (dense)


def _dense_layout(names_specs):
    names = [n for n, _ in names_specs]
    return AxisLayout([names], dict(names_specs), structure=["dense"])


def _dense_matrix(row_layout, col_layout, nb=2):
    """Hand-build a dense x dense AssembledMatrix: one block, shape (nb, row_tot, col_tot)."""
    rtot = row_layout.storage_size()
    ctot = col_layout.storage_size()
    torch.manual_seed(1)
    t = Tensor(torch.randn(nb, rtot, ctot, dtype=torch.float64), batch_ndim=1, sub_batch_ndim=0)
    return AssembledMatrix(row_layout, col_layout, [[t]])


def test_matrix_disassemble_transpose_arithmetic():
    """Dense AssembledMatrix: disassemble into typed cells, transpose, and arithmetic."""
    rl = _dense_layout([("a", Scalar), ("b", SR2)])
    cl = _dense_layout([("c", Scalar), ("d", Scalar)])
    m = _dense_matrix(rl, cl)
    sm = m.disassemble()
    assert set(sm.cells) == {"a", "b"}
    assert set(sm.cells["a"]) == {"c", "d"}
    mt = m.transpose()
    assert mt.row_layout is cl and mt.col_layout is rl
    torch.testing.assert_close(mt.tensors[0][0].data, m.tensors[0][0].data.transpose(-1, -2))
    torch.testing.assert_close((m + m).tensors[0][0].data, 2.0 * m.tensors[0][0].data)
    torch.testing.assert_close((m - m).tensors[0][0].data, torch.zeros_like(m.tensors[0][0].data))
    torch.testing.assert_close((-m).tensors[0][0].data, -m.tensors[0][0].data)


def test_matrix_matvec_matches_dense_matmul():
    """A @ x for dense groups equals the plain batched matrix-vector product."""
    sq = _dense_layout([("a", Scalar), ("b", Scalar)])  # square 2x2 so A@x is well-posed
    m = _dense_matrix(sq, sq)
    torch.manual_seed(2)
    x = AssembledVector.from_flat(sq, torch.randn(2, sq.storage_size(), dtype=torch.float64))
    y = m @ x
    expected = torch.einsum("bij,bj->bi", m.tensors[0][0].data, x.to_flat())
    torch.testing.assert_close(y.to_flat(), expected, rtol=0, atol=1e-10)


def test_matrix_batch_indexing():
    """am.batch[key] slices the dynamic axis of every block."""
    sq = _dense_layout([("a", Scalar), ("b", Scalar)])
    m = _dense_matrix(sq, sq, nb=5)
    sliced = m.batch[:3]
    assert sliced.tensors[0][0].data.shape[0] == 3


def test_matrix_select_blocks_roundtrips_disassemble():
    """select_blocks is the inverse of disassemble for dense layouts."""
    rl = _dense_layout([("a", Scalar), ("b", SR2)])
    cl = _dense_layout([("c", Scalar), ("d", Scalar)])
    m = _dense_matrix(rl, cl)
    cells = {rv: dict(inner) for rv, inner in m.disassemble().cells.items()}
    rebuilt = AssembledMatrix.select_blocks(rl, cl, cells)
    torch.testing.assert_close(rebuilt.tensors[0][0].data, m.tensors[0][0].data, rtol=0, atol=0)


def test_matrix_select_blocks_zero_fills_missing_pairs():
    """A (row_var, col_var) pair absent from the blocks dict is zero-filled."""
    rl = _dense_layout([("a", Scalar), ("b", Scalar)])
    cl = _dense_layout([("c", Scalar), ("d", Scalar)])
    m = _dense_matrix(rl, cl)
    cells = {rv: dict(inner) for rv, inner in m.disassemble().cells.items()}
    del cells["a"]["d"]  # drop one pair -> must come back as zeros
    rebuilt = AssembledMatrix.select_blocks(rl, cl, cells)
    full = rebuilt.tensors[0][0].data
    # var order a,b (rows 0,1) x c,d (cols 0,1): the (a,d) entry is element [...,0,1].
    assert full[..., 0, 1].abs().max() == 0.0
    assert full[..., 1, 1].abs().max() > 0.0  # (b,d) untouched


def test_matrix_select_blocks_rejects_block_structure():
    """select_blocks only supports dense layouts; a block layout raises."""
    import pytest

    blk = AxisLayout(
        [["g"]], {"g": Scalar}, sub_batch_shapes={"g": torch.Size([2])}, structure=["block"]
    )
    t = Tensor(torch.zeros(2, 1, 1, dtype=torch.float64), batch_ndim=1, sub_batch_ndim=0)
    with pytest.raises(NotImplementedError, match="dense"):
        AssembledMatrix.select_blocks(blk, blk, {"g": {"g": t}})


def test_matrix_select_blocks_empty_raises():
    """An empty blocks dict cannot size the zero-fill and must raise."""
    import pytest

    sq = _dense_layout([("a", Scalar), ("b", Scalar)])
    with pytest.raises(ValueError, match="empty blocks"):
        AssembledMatrix.select_blocks(sq, sq, {})


def test_matrix_matmul_matrix_matches_dense():
    """A @ B (matrix RHS) over dense groups equals the batched matmul."""
    sq = _dense_layout([("a", Scalar), ("b", Scalar)])
    a = _dense_matrix(sq, sq)
    torch.manual_seed(3)
    bt = Tensor(torch.randn(2, 2, 2, dtype=torch.float64), batch_ndim=1, sub_batch_ndim=0)
    b = AssembledMatrix(sq, sq, [[bt]])
    c = a @ b
    expected = a.tensors[0][0].data @ b.tensors[0][0].data
    torch.testing.assert_close(c.tensors[0][0].data, expected, rtol=0, atol=1e-10)


def test_matrix_matmul_error_branches():
    """Mismatched inner layouts and wrong RHS type raise."""
    import pytest

    sq = _dense_layout([("a", Scalar), ("b", Scalar)])
    other = _dense_layout([("a", Scalar), ("b", Scalar), ("c", Scalar)])
    m = _dense_matrix(sq, sq)
    x_bad = AssembledVector.from_flat(
        other, torch.zeros(2, other.storage_size(), dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="matching inner layouts"):
        _ = m @ x_bad
    with pytest.raises(TypeError):
        _ = m @ 5  # type: ignore[operator]  # neither vector nor matrix
    m_bad = _dense_matrix(other, other)
    with pytest.raises(ValueError, match="matching inner layouts"):
        _ = m @ m_bad


def test_matrix_per_instance_matvec_dense():
    """per_instance_matvec over dense groups matches A@x and Aᵀ@x; group mismatch raises."""
    import pytest

    sq = _dense_layout([("a", Scalar), ("b", Scalar)])
    m = _dense_matrix(sq, sq)
    torch.manual_seed(4)
    x = AssembledVector.from_flat(sq, torch.randn(2, sq.storage_size(), dtype=torch.float64))
    y = m.per_instance_matvec(x)
    torch.testing.assert_close(
        y.to_flat(),
        torch.einsum("bij,bj->bi", m.tensors[0][0].data, x.to_flat()),
        atol=1e-10,
        rtol=0,
    )
    yt = m.per_instance_matvec(x, transpose=True)
    torch.testing.assert_close(
        yt.to_flat(),
        torch.einsum("bji,bj->bi", m.tensors[0][0].data, x.to_flat()),
        atol=1e-10,
        rtol=0,
    )
    two_group = AxisLayout([["a"], ["b"]], {"a": Scalar, "b": Scalar}, structure=["dense", "dense"])
    x_bad = AssembledVector.from_flat(
        two_group, torch.zeros(2, two_group.storage_size(), dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="group"):
        m.per_instance_matvec(x_bad)


def test_matrix_transpose_rejects_two_intmd_axes():
    """A block carrying two intermediate sub-batch axes can't be base-swap transposed."""
    import pytest

    layout = AxisLayout(
        [["g"]], {"g": Scalar}, sub_batch_shapes={"g": torch.Size([2, 2])}, structure=["block"]
    )
    t = Tensor(torch.zeros(1, 2, 2, 1, 1, dtype=torch.float64), batch_ndim=1, sub_batch_ndim=2)
    m = AssembledMatrix(layout, layout, [[t]])
    with pytest.raises(NotImplementedError, match="at most one intermediate"):
        m.transpose()


def test_common_prefix_stops_at_first_divergence():
    """The paired-block common axis is the longest shared leading sub-batch prefix:
    equal leading axes accumulate; the first mismatch stops it (shorter wins)."""
    # Full match on the shorter shape (row common-only vs row+extra col).
    assert _common_prefix((2,), (2, 3)) == (2,)
    # Diverge at the second axis -> prefix truncates there (the break path).
    assert _common_prefix((2, 3), (2, 5)) == (2,)
    # Diverge at the very first axis -> empty common.
    assert _common_prefix((4, 3), (2, 3)) == ()
    # Two fully-equal shapes keep every axis.
    assert _common_prefix((2, 3), (2, 3)) == (2, 3)
