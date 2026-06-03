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

"""Coverage for the region-view shape API (``t.batch`` / ``t.dynamic_batch`` /
``t.sub_batch`` / ``t.base``). The sub-batch-specific ops are exercised in
``test_sub_batch_shape.py``; this file covers the cross-region surface and
the new ``dynamic_batch`` / ``base`` / ``batch`` views."""

import pytest
import torch

from neml2.types import R2, SR2, SSR4, Scalar, Vec

# ---------- region shape introspection ----------


def test_views_partition_data_shape():
    """``dynamic_batch + sub_batch + base`` always reconstructs ``data.shape``."""
    s = SR2(torch.zeros(2, 3, 4, 6)).sub_batch.retag(2)
    parts = (*s.dynamic_batch.shape, *s.sub_batch.shape, *s.base.shape)
    assert torch.Size(parts) == s.data.shape
    # batch view spans dynamic + sub
    assert s.batch.shape == torch.Size([2, 3, 4])
    assert s.batch.ndim == s.dynamic_batch.ndim + s.sub_batch.ndim


def test_views_report_zero_ndim_when_region_empty():
    s = Scalar(torch.zeros(()))  # 0-dim
    assert s.batch.ndim == 0
    assert s.batch.shape == torch.Size([])
    assert s.dynamic_batch.ndim == 0
    assert s.sub_batch.ndim == 0
    assert s.base.ndim == 0


# ---------- dynamic_batch view ----------


def test_dynamic_batch_expand_prepends_axes_when_needed():
    """torch.expand semantics: pads region's ndim from the left."""
    fill = SR2(torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64))
    assert fill.dynamic_batch.shape == torch.Size([])
    e = fill.dynamic_batch.expand(20)
    assert e.data.shape == torch.Size([20, 6])
    assert e.sub_batch_ndim == 0
    # Every batch row carries the same base values.
    assert torch.equal(e.data[0], fill.data)
    assert torch.equal(e.data[19], fill.data)


def test_dynamic_batch_expand_multi_dim():
    fill = SR2(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    e = fill.dynamic_batch.expand(100, 10)
    assert e.data.shape == torch.Size([100, 10, 6])
    assert e.dynamic_batch.shape == torch.Size([100, 10])


def test_dynamic_batch_expand_preserves_existing_sub_batch():
    """``sub_batch_ndim`` and ``sub_batch_shape`` ride through ``dynamic_batch.expand``."""
    s = Scalar(torch.zeros(5)).sub_batch.retag(1)  # dyn=(), sub=(5,)
    e = s.dynamic_batch.expand(7)
    assert e.dynamic_batch.shape == torch.Size([7])
    assert e.sub_batch.shape == torch.Size([5])
    assert e.sub_batch_ndim == 1
    assert e.data.shape == torch.Size([7, 5])


def test_dynamic_batch_expand_rejects_shrinking():
    s = Scalar(torch.zeros(5, 3))
    with pytest.raises(ValueError, match="fewer dims than current"):
        s.dynamic_batch.expand(3)


def test_dynamic_batch_unsqueeze_does_not_bump_sub_batch_ndim():
    s = Scalar(torch.zeros(2, 3)).sub_batch.retag(1)  # dyn=(2,), sub=(3,)
    u = s.dynamic_batch.unsqueeze(0)
    assert u.data.shape == torch.Size([1, 2, 3])
    assert u.sub_batch_ndim == 1  # unchanged
    assert u.dynamic_batch.shape == torch.Size([1, 2])
    assert u.sub_batch.shape == torch.Size([3])


def test_dynamic_batch_squeeze_only_drops_size_one():
    s = Scalar(torch.zeros(1, 4))
    sq = s.dynamic_batch.squeeze(0)
    assert sq.data.shape == torch.Size([4])
    with pytest.raises(ValueError, match="expected 1"):
        Scalar(torch.zeros(3, 4)).dynamic_batch.squeeze(0)


def test_dynamic_batch_cat_along_leading_axis():
    a = Scalar(torch.tensor([1.0, 2.0]))
    b = Scalar(torch.tensor([10.0, 20.0, 30.0]))
    c = a.dynamic_batch.cat([b], dim=0)
    assert c.data.tolist() == [1.0, 2.0, 10.0, 20.0, 30.0]
    assert c.sub_batch_ndim == 0


# ---------- batch view (read-mostly) ----------


def test_batch_view_disallows_unsqueeze_expand_squeeze():
    s = Scalar(torch.zeros(2, 3))
    with pytest.raises(TypeError, match="ambiguous"):
        s.batch.unsqueeze(0)
    with pytest.raises(TypeError, match="ambiguous"):
        s.batch.expand(4, 2, 3)
    with pytest.raises(TypeError, match="ambiguous"):
        s.batch.squeeze(0)


def test_batch_view_cat_treats_combined_region_as_one_axis():
    """``batch.cat`` works because it doesn't change the sub-batch / dynamic split."""
    a = Scalar(torch.zeros(2, 3)).sub_batch.retag(1)  # dyn=(2,), sub=(3,)
    b = Scalar(torch.ones(2, 4)).sub_batch.retag(1)
    # Concat along the sub-batch axis (axis 1 in batch numbering) preserves both regions.
    c = a.batch.cat([b], dim=1)
    assert c.data.shape == torch.Size([2, 7])
    assert c.sub_batch_ndim == 1


# ---------- base view ----------


def test_base_transpose_swaps_default_trailing_axes():
    r = R2(torch.arange(9, dtype=torch.float64).reshape(3, 3))
    tr = r.base.transpose()
    assert torch.equal(tr.data, r.data.transpose(-2, -1))


def test_base_transpose_explicit_dims():
    """Explicit dim0/dim1 (region-relative) match the default."""
    r = R2(torch.arange(9, dtype=torch.float64).reshape(3, 3))
    tr_a = r.base.transpose(0, 1)
    tr_b = r.base.transpose(-2, -1)
    assert torch.equal(tr_a.data, tr_b.data)


def test_base_transpose_with_batch_axes():
    r = R2(torch.arange(2 * 3 * 3, dtype=torch.float64).reshape(2, 3, 3))
    tr = r.base.transpose()
    # Only the last two axes flip; the batch axis is untouched.
    assert tr.data.shape == torch.Size([2, 3, 3])
    assert torch.equal(tr.data[0], r.data[0].transpose(-2, -1))


def test_base_transpose_rejects_scalar_and_vec_wrappers():
    """``base.transpose`` requires BASE_NDIM >= 2."""
    with pytest.raises(TypeError, match="BASE_NDIM >= 2"):
        Scalar(torch.zeros(3, 4)).base.transpose()
    with pytest.raises(TypeError, match="BASE_NDIM >= 2"):
        Vec(torch.zeros(2, 3)).base.transpose()
    with pytest.raises(TypeError, match="BASE_NDIM >= 2"):
        SR2(torch.zeros(6)).base.transpose()


def test_base_transpose_ok_on_ssr4():
    """SSR4 has BASE_NDIM=2 so ``base.transpose`` is allowed."""
    c = SSR4(torch.arange(36, dtype=torch.float64).reshape(6, 6))
    tr = c.base.transpose()
    assert torch.equal(tr.data, c.data.transpose(-2, -1))


# ---------- chaining across views ----------


def test_chain_dynamic_batch_expand_then_sub_batch_unsqueeze():
    """View methods return wrappers, so consecutive view chains work cleanly."""
    fill = SR2(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    chained = fill.dynamic_batch.expand(20).sub_batch.unsqueeze(0)
    assert chained.data.shape == torch.Size([20, 1, 6])
    assert chained.dynamic_batch.shape == torch.Size([20])
    assert chained.sub_batch.shape == torch.Size([1])


def test_chain_preserves_concrete_wrapper_type():
    """``t.dynamic_batch.expand(N)`` typed as ``SR2`` (not ``TensorWrapper``)."""
    fill = SR2(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    e = fill.dynamic_batch.expand(5)
    assert isinstance(e, SR2)
    # And the chained .sub_batch.unsqueeze() also returns SR2
    u = e.sub_batch.unsqueeze(0)
    assert isinstance(u, SR2)


# ---------- region-relative dim resolution ----------


def test_dim_out_of_range_raises_index_error():
    s = Scalar(torch.zeros(2, 3)).sub_batch.retag(1)  # sub_ndim=1
    with pytest.raises(IndexError, match="sub_batch insert dim 5"):
        s.sub_batch.unsqueeze(5)
    with pytest.raises(IndexError, match="dynamic_batch insert dim 5"):
        s.dynamic_batch.unsqueeze(5)


def test_negative_dim_resolves_relative_to_region():
    """``dim=-1`` is the last axis of the region, not of ``data``."""
    s = Scalar(torch.zeros(2, 3, 4)).sub_batch.retag(2)  # dyn=(2,), sub=(3, 4)
    # Sub-batch -1 should refer to the trailing sub-batch axis (size 4).
    u = s.sub_batch.unsqueeze(-1)
    # New axis goes right after the existing sub-batch region.
    assert u.data.shape == torch.Size([2, 3, 4, 1])
    assert u.sub_batch.shape == torch.Size([3, 4, 1])
