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

import pytest
import torch

from neml2.types import SR2, SSR4, Scalar

# ---------- shape decomposition ----------


def test_default_sub_batch_is_zero():
    """Pre-sub-batch wrappers see batch_shape unchanged (backwards compat)."""
    s = Scalar(torch.zeros(2, 3))
    assert s.sub_batch_ndim == 0
    assert s.dynamic_batch_shape == torch.Size([2, 3])
    assert s.sub_batch_shape == torch.Size([])
    assert s.batch_shape == torch.Size([2, 3])
    assert s.base_shape == torch.Size([])

    r = SR2(torch.zeros(2, 3, 6))
    assert r.dynamic_batch_shape == torch.Size([2, 3])
    assert r.sub_batch_shape == torch.Size([])
    assert r.batch_shape == torch.Size([2, 3])
    assert r.base_shape == torch.Size([6])

    c = SSR4(torch.zeros(4, 6, 6))
    assert c.dynamic_batch_shape == torch.Size([4])
    assert c.sub_batch_shape == torch.Size([])
    assert c.batch_shape == torch.Size([4])
    assert c.base_shape == torch.Size([6, 6])


def test_sub_batch_retag_promotes_trailing_batch_dims():
    """`sub_batch.retag(k)` reinterprets the last k batch dims as sub-batch."""
    # Scalar (B=4, L1=2, L2=3) with sub_batch_ndim=2
    s = Scalar(torch.zeros(4, 2, 3)).sub_batch.retag(2)
    assert s.sub_batch_ndim == 2
    assert s.dynamic_batch_shape == torch.Size([4])
    assert s.sub_batch_shape == torch.Size([2, 3])
    assert s.batch_shape == torch.Size([4, 2, 3])
    assert s.base_shape == torch.Size([])

    # SR2 (B=5, L=7) with sub_batch_ndim=1
    r = SR2(torch.zeros(5, 7, 6)).sub_batch.retag(1)
    assert r.dynamic_batch_shape == torch.Size([5])
    assert r.sub_batch_shape == torch.Size([7])
    assert r.batch_shape == torch.Size([5, 7])
    assert r.base_shape == torch.Size([6])

    # SR2 with two leading dynamic dims and two sub-batch dims
    r2 = SR2(torch.zeros(2, 3, 4, 5, 6)).sub_batch.retag(2)
    assert r2.dynamic_batch_shape == torch.Size([2, 3])
    assert r2.sub_batch_shape == torch.Size([4, 5])
    assert r2.base_shape == torch.Size([6])


def test_sub_batch_retag_idempotent_returns_self():
    s = Scalar(torch.zeros(3, 4))
    assert s.sub_batch.retag(0) is s


def test_sub_batch_flatten_demotes_to_dynamic():
    s = Scalar(torch.zeros(2, 3, 4)).sub_batch.retag(2)
    f = s.sub_batch.flatten()
    assert f.sub_batch_ndim == 0
    assert f.dynamic_batch_shape == torch.Size([2, 3, 4])
    assert f.batch_shape == torch.Size([2, 3, 4])
    # data is the same tensor, not a copy
    assert f.data is s.data


def test_sub_batch_retag_validates_range():
    with pytest.raises(ValueError, match="non-negative"):
        Scalar(torch.zeros(2)).sub_batch.retag(-1)
    with pytest.raises(ValueError, match="exceeds available batch dims"):
        Scalar(torch.zeros(2)).sub_batch.retag(2)
    with pytest.raises(ValueError, match="exceeds available batch dims"):
        SR2(torch.zeros(3, 6)).sub_batch.retag(2)
    # Rank-0 Scalar has no batch dims at all.
    with pytest.raises(ValueError, match="exceeds available batch dims"):
        Scalar(torch.tensor(1.0)).sub_batch.retag(1)


# ---------- sub-batch ops ----------


def test_sub_batch_expand_at_adds_leading_axis():
    """`expand_at(N)` lifts a sub-batch-trivial wrapper to one with sub_batch_shape=(N,)."""
    s = Scalar(torch.tensor([10.0, 20.0]))  # shape (2,)
    e = s.sub_batch.expand_at(4)
    assert e.shape == torch.Size([2, 4])
    assert e.sub_batch_ndim == 1
    assert e.dynamic_batch_shape == torch.Size([2])
    assert e.sub_batch_shape == torch.Size([4])
    # Every sub-batch slot carries the same value.
    assert torch.equal(e.data[0], torch.full((4,), 10.0))
    assert torch.equal(e.data[1], torch.full((4,), 20.0))


def test_sub_batch_expand_at_inserts_at_chosen_position():
    s = Scalar(torch.zeros(2, 5)).sub_batch.retag(1)  # sb=(5,)
    e_front = s.sub_batch.expand_at(3, dim=0)
    assert e_front.sub_batch_shape == torch.Size([3, 5])
    assert e_front.dynamic_batch_shape == torch.Size([2])
    e_back = s.sub_batch.expand_at(3, dim=1)
    assert e_back.sub_batch_shape == torch.Size([5, 3])
    assert e_back.dynamic_batch_shape == torch.Size([2])


def test_sub_batch_expand_at_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="size must be positive"):
        Scalar(torch.zeros(2)).sub_batch.expand_at(0)


def test_sub_batch_unsqueeze_adds_size_one_axes():
    s = Scalar(torch.tensor([1.0, 2.0, 3.0])).sub_batch.retag(1)  # sb=(3,)
    front = s.sub_batch.unsqueeze(0)
    assert front.shape == torch.Size([1, 3])
    assert front.sub_batch_shape == torch.Size([1, 3])
    back = s.sub_batch.unsqueeze(-1)
    assert back.shape == torch.Size([3, 1])
    assert back.sub_batch_shape == torch.Size([3, 1])
    multi = s.sub_batch.unsqueeze(0, n=2)
    assert multi.shape == torch.Size([1, 1, 3])
    assert multi.sub_batch_ndim == 3


def test_sub_batch_unsqueeze_n_zero_is_identity():
    s = Scalar(torch.zeros(2))
    assert s.sub_batch.unsqueeze(0, n=0) is s


def test_sub_batch_diagonalize_scalar():
    """Trailing sub-batch axis becomes an (L, L) diagonal block — the KWN pattern."""
    s = Scalar(torch.tensor([2.0, 3.0, 5.0])).sub_batch.retag(1)  # sb=(3,)
    d = s.sub_batch.diagonalize()
    assert d.shape == torch.Size([3, 3])
    assert d.sub_batch_ndim == 2
    assert d.sub_batch_shape == torch.Size([3, 3])
    expected = torch.diag(torch.tensor([2.0, 3.0, 5.0]))
    assert torch.equal(d.data, expected)


def test_sub_batch_diagonalize_with_dynamic_batch():
    # (B=2, L=3) → (B=2, L=3, L=3)
    s = Scalar(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sub_batch.retag(1)
    d = s.sub_batch.diagonalize()
    assert d.shape == torch.Size([2, 3, 3])
    assert d.dynamic_batch_shape == torch.Size([2])
    assert d.sub_batch_shape == torch.Size([3, 3])
    assert torch.equal(d.data[0], torch.diag(torch.tensor([1.0, 2.0, 3.0])))
    assert torch.equal(d.data[1], torch.diag(torch.tensor([4.0, 5.0, 6.0])))


def test_sub_batch_diagonalize_requires_sub_batch_dim():
    with pytest.raises(ValueError, match="at least one sub-batch dim"):
        Scalar(torch.zeros(3)).sub_batch.diagonalize()


def test_sub_batch_diagonalize_not_implemented_for_non_scalar():
    with pytest.raises(NotImplementedError):
        SR2(torch.zeros(3, 6)).sub_batch.retag(1).sub_batch.diagonalize()


def test_sub_batch_cat_along_leading_axis():
    a = Scalar(torch.tensor([1.0, 2.0])).sub_batch.retag(1)  # sb=(2,)
    b = Scalar(torch.tensor([10.0, 20.0, 30.0])).sub_batch.retag(1)  # sb=(3,)
    c = a.sub_batch.cat([b], dim=0)
    assert c.shape == torch.Size([5])
    assert c.sub_batch_shape == torch.Size([5])
    assert c.data.tolist() == [1.0, 2.0, 10.0, 20.0, 30.0]


def test_sub_batch_cat_with_dynamic_batch():
    a = Scalar(torch.zeros(4, 2)).sub_batch.retag(1)
    b = Scalar(torch.ones(4, 3)).sub_batch.retag(1)
    c = a.sub_batch.cat([b], dim=0)
    assert c.shape == torch.Size([4, 5])
    assert c.dynamic_batch_shape == torch.Size([4])
    assert c.sub_batch_shape == torch.Size([5])


def test_sub_batch_cat_validates_inputs():
    a = Scalar(torch.zeros(3)).sub_batch.retag(1)
    b = Scalar(torch.zeros(3, 4)).sub_batch.retag(2)
    with pytest.raises(ValueError, match="mismatched sub_batch_ndim"):
        a.sub_batch.cat([b], dim=0)

    r = SR2(torch.zeros(3, 6)).sub_batch.retag(1)
    with pytest.raises(TypeError, match="mixed wrapper types"):
        a.sub_batch.cat([r], dim=0)  # type: ignore[list-item]


# ---------- KWN-style expand → diagonalize → Hadamard ----------


def test_kwn_per_bin_diagonal_derivative_kernel():
    """Reproduces the C++ pattern in RateLimitedPrecipitateGrowthRate.cxx:91-93.

    ``d(rate)/d(R) * diag(r_map)`` collapses to a per-bin Hadamard against
    the diagonal block, giving the block-diagonal derivative ``δ_{ij}·D_i``.
    """
    d_rate_dR = Scalar(torch.tensor([2.0, 3.0, 5.0, 7.0])).sub_batch.retag(1)  # sb=(4,)
    r_map = Scalar(torch.tensor([10.0, 20.0, 30.0, 40.0])).sub_batch.retag(1)
    diag_r = r_map.sub_batch.diagonalize()  # (L, L)
    # ``d_rate_dR.sub_batch.unsqueeze(-1)`` makes (L, 1) so it broadcasts
    # against (L, L) → (L, L) Hadamard.
    block = d_rate_dR.sub_batch.unsqueeze(-1).data * diag_r.data
    expected = torch.diag(torch.tensor([2.0 * 10.0, 3.0 * 20.0, 5.0 * 30.0, 7.0 * 40.0]))
    assert torch.equal(block, expected)


# ---------- pytree drop_field_names interaction ----------


def test_pytree_roundtrip_drops_sub_batch_ndim():
    """Per drop_field_names registration, pytree flatten/unflatten resets
    sub_batch_ndim to the dataclass default. The exported graph operates
    on raw tensors so the loss of the hint is harmless there."""
    from torch.utils import _pytree as pytree

    s = Scalar(torch.zeros(3, 4)).sub_batch.retag(1)
    assert s.sub_batch_ndim == 1
    leaves, spec = pytree.tree_flatten({"x": s})
    # Only the data tensor is a leaf.
    assert len(leaves) == 1
    assert isinstance(leaves[0], torch.Tensor)
    rt = pytree.tree_unflatten(leaves, spec)
    assert rt["x"].data.shape == torch.Size([3, 4])
    assert rt["x"].sub_batch_ndim == 0


def test_sub_batch_metadata_survives_to_device_roundtrip():
    s = Scalar(torch.zeros(3, 4)).sub_batch.retag(1)
    s2 = s.to(torch.float32)
    assert s2.sub_batch_ndim == 1
    assert s2.sub_batch_shape == torch.Size([4])
