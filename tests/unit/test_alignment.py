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

"""Sub-batch alignment tests.

Locks in the C++-style ``align_intmd_dim`` contract on every typed-wrapper
binary operator and every free function: a *global* operand (no sub-batch)
combined with a *per-sub-batch-site* operand must broadcast correctly at
any dynamic batch size, with the result carrying the unified
``sub_batch_ndim``.

The regression backstop in ``test_crystal_plasticity.py``
(``test_taylor_cp_native_matches_cxx_gold_one_step``) covers the
``sub_batch_ndim == 0`` byte-identical case; this file covers the cross-ndim
matrix.
"""

from __future__ import annotations

import pytest
import torch

from neml2.types import (
    R2,
    SR2,
    SSR4,
    WR2,
    Rot,
    Scalar,
    align_sub_batch,
    compose,
    dev,
    drotate,
    drotate_self,
    exp,
    exp_map,
    inner,
    norm,
    outer,
    r2_from_sr2,
    r2_from_wr2,
    rotate,
    skew,
    sqrt,
    sym,
    tr,
    unit,
    vol,
)

DT = torch.float64

# Shapes to parametrize over: (dynamic batch, sub_batch shape).
# (1, ()) is the original Phase-10.8 byte-identical baseline; everything else
# is new coverage that catches the broadcast bug at B>1 and sub-batch > 0.
SHAPES = [
    pytest.param((1, ()), id="B1_global"),
    pytest.param((2, ()), id="B2_global"),
    pytest.param((1, (5,)), id="B1_per_crystal_5"),
    pytest.param((2, (5,)), id="B2_per_crystal_5"),
    pytest.param((2, (3, 7)), id="B2_per_site_3x7"),
]


def _make(cls, dynamic_batch, sub_batch_shape, seed):
    g = torch.Generator().manual_seed(seed)
    shape = (dynamic_batch, *sub_batch_shape, *cls.BASE_SHAPE)
    if cls is Scalar and not sub_batch_shape:
        shape = (dynamic_batch,)
    return cls(torch.randn(*shape, generator=g, dtype=DT), sub_batch_ndim=len(sub_batch_shape))


# ---------------------------------------------------------------------------
# align_sub_batch
# ---------------------------------------------------------------------------


def test_align_sub_batch_no_op_when_matching():
    a = _make(SR2, 2, (5,), 0)
    b = _make(SR2, 2, (5,), 1)
    [aa, bb], sb = align_sub_batch(a, b)
    assert aa is a and bb is b
    assert sb == 1


def test_align_sub_batch_pads_global_to_per_crystal():
    g = _make(SR2, 2, (), 0)  # (2, 6)
    p = _make(SR2, 2, (5,), 1)  # (2, 5, 6)
    [aa, bb], sb = align_sub_batch(g, p)
    assert sb == 1
    assert aa.data.shape == (2, 1, 6)  # padded
    assert bb.data.shape == (2, 5, 6)  # unchanged


def test_align_sub_batch_handles_n_ary():
    a = _make(SR2, 2, (), 0)
    b = _make(SR2, 2, (5,), 1)
    c = _make(SR2, 2, (3, 5), 2)
    [aa, bb, cc], sb = align_sub_batch(a, b, c)
    assert sb == 2
    assert aa.data.shape == (2, 1, 1, 6)
    assert bb.data.shape == (2, 1, 5, 6)
    assert cc.data.shape == (2, 3, 5, 6)


# ---------------------------------------------------------------------------
# Binary operators on every typed wrapper — global ⊕ per-crystal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name,op", [("+", lambda a, b: a + b), ("-", lambda a, b: a - b)])
@pytest.mark.parametrize("cls", [Scalar, SR2, SSR4, R2, Rot, WR2])
def test_additive_ops_align_sub_batch(cls, op_name, op):
    B, N = 2, 5
    g = _make(cls, B, (), 0)
    p = _make(cls, B, (N,), 1)
    result = op(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, *cls.BASE_SHAPE)


@pytest.mark.parametrize("cls", [SR2, SSR4, R2, Rot, WR2])
def test_scalar_product_aligns_sub_batch(cls):
    """Per-crystal wrapper × global Scalar."""
    B, N = 2, 5
    p = _make(cls, B, (N,), 0)
    s = _make(Scalar, B, (), 1)
    result = p * s
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, *cls.BASE_SHAPE)


@pytest.mark.parametrize("cls", [SR2, SSR4, R2, Rot, WR2])
def test_global_scalar_times_per_crystal_wrapper(cls):
    """Global Scalar × per-crystal wrapper — opposite operand order."""
    B, N = 2, 5
    g = _make(cls, B, (), 0)
    s = _make(Scalar, B, (N,), 1)
    result = g * s
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, *cls.BASE_SHAPE)


def test_r2_matmul_global_times_per_crystal():
    """The canonical broken eager broadcast — fixed at B>1 by align_sub_batch."""
    B, N = 2, 5
    g = _make(R2, B, (), 0)
    p = _make(R2, B, (N,), 1)
    result = g @ p
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3, 3)


def test_r2_matmul_per_crystal_times_global():
    B, N = 2, 5
    g = _make(R2, B, (), 0)
    p = _make(R2, B, (N,), 1)
    result = p @ g
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3, 3)


def test_ssr4_matmul_sr2_global_per_crystal():
    B, N = 2, 5
    T = _make(SSR4, B, (), 0)
    s = _make(SR2, B, (N,), 1)
    result = T @ s
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 6)


def test_ssr4_matmul_ssr4_per_crystal():
    B, N = 2, 5
    T = _make(SSR4, B, (), 0)
    U = _make(SSR4, B, (N,), 1)
    result = T @ U
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 6, 6)


# ---------------------------------------------------------------------------
# Reorder invariance: (g op p) == (p op g) up to operand-order sign
# ---------------------------------------------------------------------------


def test_additive_reorder_byte_identical():
    """``(g + p)`` and ``(p + g)`` produce the same tensor."""
    B, N = 2, 5
    g = _make(SR2, B, (), 0)
    p = _make(SR2, B, (N,), 1)
    a = g + p
    b = p + g
    assert torch.equal(a.data, b.data)
    assert a.sub_batch_ndim == b.sub_batch_ndim == 1


# ---------------------------------------------------------------------------
# Free functions in types.functions preserve / propagate sub_batch_ndim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_tr_propagates_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    result = tr(A)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb)


@pytest.mark.parametrize("shape", SHAPES)
def test_vol_propagates_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    result = vol(A)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb, 6)


@pytest.mark.parametrize("shape", SHAPES)
def test_dev_propagates_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    result = dev(A)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb, 6)


@pytest.mark.parametrize("shape", SHAPES)
def test_norm_propagates_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    result = norm(A)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb)


@pytest.mark.parametrize("shape", SHAPES)
def test_unit_propagates_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    result = unit(A, eps=1e-12)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb, 6)


@pytest.mark.parametrize("shape", SHAPES)
def test_sqrt_exp_propagate_sub_batch(shape):
    B, sb = shape
    A = _make(SR2, B, sb, 0)
    s = norm(A)  # already carries sub_batch
    assert sqrt(s).sub_batch_ndim == len(sb)
    assert exp(s).sub_batch_ndim == len(sb)


def test_inner_aligns_global_with_per_crystal():
    B, N = 2, 5
    g = _make(SR2, B, (), 0)
    p = _make(SR2, B, (N,), 1)
    result = inner(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N)


def test_outer_aligns_global_with_per_crystal():
    B, N = 2, 5
    g = _make(SR2, B, (), 0)
    p = _make(SR2, B, (N,), 1)
    result = outer(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 6, 6)


@pytest.mark.parametrize("shape", SHAPES)
def test_r2_from_sr2_preserves_sub_batch(shape):
    B, sb = shape
    s = _make(SR2, B, sb, 0)
    result = r2_from_sr2(s)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb, 3, 3)


@pytest.mark.parametrize("shape", SHAPES)
def test_r2_from_wr2_preserves_sub_batch(shape):
    B, sb = shape
    w = _make(WR2, B, sb, 0)
    result = r2_from_wr2(w)
    assert result.sub_batch_ndim == len(sb)
    assert result.data.shape == (B, *sb, 3, 3)


@pytest.mark.parametrize("shape", SHAPES)
def test_sym_skew_preserve_sub_batch(shape):
    B, sb = shape
    t = _make(R2, B, sb, 0)
    s = sym(t)
    w = skew(t)
    assert s.sub_batch_ndim == len(sb) and s.data.shape == (B, *sb, 6)
    assert w.sub_batch_ndim == len(sb) and w.data.shape == (B, *sb, 3)


def test_compose_aligns_global_with_per_crystal():
    B, N = 2, 5
    g = _make(Rot, B, (), 0)
    p = _make(Rot, B, (N,), 1)
    result = compose(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3)


def test_drotate_aligns_global_with_per_crystal():
    B, N = 2, 5
    g = _make(Rot, B, (), 0)
    p = _make(Rot, B, (N,), 1)
    result = drotate(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3, 3)


def test_drotate_self_aligns_global_with_per_crystal():
    B, N = 2, 5
    g = _make(Rot, B, (), 0)
    p = _make(Rot, B, (N,), 1)
    result = drotate_self(g, p)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3, 3)


@pytest.mark.parametrize("shape", SHAPES)
def test_exp_map_preserves_sub_batch(shape):
    B, sb = shape
    w = _make(WR2, B, sb, 0)
    r = exp_map(w)
    assert r.sub_batch_ndim == len(sb)
    assert r.data.shape == (B, *sb, 3)


def test_rotate_sym_aligns_global_R_with_per_crystal_SR2():
    B, N = 2, 5
    s = _make(SR2, B, (N,), 0)
    R = _make(R2, B, (), 1)
    result = rotate(s, R)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 6)


def test_rotate_skew_aligns_global_R_with_per_crystal_WR2():
    B, N = 2, 5
    w = _make(WR2, B, (N,), 0)
    R = _make(R2, B, (), 1)
    result = rotate(w, R)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 3)


def test_rotate_ssr4_aligns_global_R_with_per_crystal_SSR4():
    B, N = 2, 5
    T = _make(SSR4, B, (N,), 0)
    R = _make(R2, B, (), 1)
    result = rotate(T, R)
    assert result.sub_batch_ndim == 1
    assert result.data.shape == (B, N, 6, 6)


# ---------------------------------------------------------------------------
# Numerical agreement: aligned op == hand-aligned reference
# ---------------------------------------------------------------------------


def test_sr2_subtraction_matches_hand_aligned():
    B, N = 2, 5
    g = _make(SR2, B, (), 0)
    p = _make(SR2, B, (N,), 1)
    aligned = g - p
    # Hand reference: pre-pad g via sub_batch.unsqueeze before raw subtract.
    g_padded = g.sub_batch.unsqueeze(0, 1)
    reference = g_padded.data - p.data
    assert torch.equal(aligned.data, reference)


def test_r2_matmul_matches_hand_aligned():
    B, N = 2, 5
    g = _make(R2, B, (), 0)
    p = _make(R2, B, (N,), 1)
    aligned = g @ p
    g_padded = g.sub_batch.unsqueeze(0, 1)
    reference = g_padded.data @ p.data
    assert torch.equal(aligned.data, reference)
