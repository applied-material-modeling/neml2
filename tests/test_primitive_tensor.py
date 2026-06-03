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

"""Direct exercise of ``PrimitiveTensor``'s generic surface via a throwaway
subclass. The concrete primitives (``Scalar``, ``Vec``, ``SR2``, …) have
their own integration tests; this file isolates the inherited behavior so
regressions in ``_primitive.py`` are visible without the per-leaf noise.

The throwaway type ``_Foo`` has ``BASE_SHAPE = (4,)`` — different from any
real primitive — so accidental dispatch to a built-in type is also caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch

from neml2.types import Scalar
from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class _Foo(PrimitiveTensor):
    """Throwaway primitive for testing: base shape (4,), no special semantics."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (4,)


# Register once at import time. Pytest collects this file before the registry
# is queried; a duplicate register call later would raise.
register(_Foo)


# ---------- generic factories ----------


def test_zeros_uses_base_shape():
    f = _Foo.zeros(2, 3)
    assert f.data.shape == torch.Size([2, 3, 4])
    assert torch.all(f.data == 0)


def test_ones_uses_base_shape():
    f = _Foo.ones(5)
    assert f.data.shape == torch.Size([5, 4])
    assert torch.all(f.data == 1)


def test_full_uses_base_shape_and_value():
    f = _Foo.full(2, fill_value=7.5)
    assert f.data.shape == torch.Size([2, 4])
    assert torch.all(f.data == 7.5)


def test_empty_uses_base_shape():
    f = _Foo.empty(3)
    assert f.data.shape == torch.Size([3, 4])
    # contents undefined; just confirm shape


def test_factories_accept_dtype_device():
    f = _Foo.zeros(2, dtype=torch.float32)
    assert f.dtype == torch.float32
    g = _Foo.full(2, fill_value=1.0, device="cpu")
    assert g.device.type == "cpu"


def test_factories_preserve_concrete_type():
    """Factories return the concrete class, not the base."""
    assert type(_Foo.zeros(3)) is _Foo


# ---------- generic fill template ----------


def test_fill_with_exact_components():
    f = _Foo.fill(1.0, 2.0, 3.0, 4.0)
    assert f.data.shape == torch.Size([4])
    assert f.data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_fill_rejects_wrong_component_count():
    with pytest.raises(ValueError, match="expects 4 components"):
        _Foo.fill(1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="expects 4 components"):
        _Foo.fill(1.0, 2.0, 3.0, 4.0, 5.0)


# ---------- generic arithmetic operators ----------


def _foo(values, sub_batch_ndim=0):
    return _Foo(torch.tensor(values, dtype=torch.float64), sub_batch_ndim=sub_batch_ndim)


def test_add_same_type():
    a = _foo([1.0, 2.0, 3.0, 4.0])
    b = _foo([10.0, 20.0, 30.0, 40.0])
    r = a + b
    assert isinstance(r, _Foo)
    assert r.data.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_sub_same_type():
    a = _foo([10.0, 20.0, 30.0, 40.0])
    b = _foo([1.0, 2.0, 3.0, 4.0])
    assert (a - b).data.tolist() == [9.0, 18.0, 27.0, 36.0]


def test_mul_same_type():
    a = _foo([1.0, 2.0, 3.0, 4.0])
    b = _foo([2.0, 2.0, 2.0, 2.0])
    assert (a * b).data.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_truediv_same_type():
    a = _foo([2.0, 4.0, 6.0, 8.0])
    b = _foo([2.0, 2.0, 2.0, 2.0])
    assert (a / b).data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_neg():
    a = _foo([1.0, -2.0, 3.0, -4.0])
    assert (-a).data.tolist() == [-1.0, 2.0, -3.0, 4.0]


def test_scalar_interop_broadcasts_base():
    """A bare Scalar's data has base shape (); ``align_scalar_base`` pads it
    to broadcast against the (4,) base."""
    a = _foo([1.0, 2.0, 3.0, 4.0])
    s = Scalar(2.0)
    r = a * s
    assert isinstance(r, _Foo)
    assert r.data.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_scalar_interop_in_add_and_sub():
    a = _foo([1.0, 2.0, 3.0, 4.0])
    s = Scalar(10.0)
    assert (a + s).data.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert (a - s).data.tolist() == [-9.0, -8.0, -7.0, -6.0]


def test_scalar_division():
    a = _foo([10.0, 20.0, 30.0, 40.0])
    s = Scalar(10.0)
    assert (a / s).data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_returns_notimplemented_for_unknown_other():
    """Cross-type ops fall through to ``NotImplemented`` — the leaf class
    is responsible for any cross-type promotion it wants to support."""
    a = _foo([1.0, 2.0, 3.0, 4.0])
    assert a._binary("not a wrapper", lambda x, y: x + y) is NotImplemented


# ---------- sub-batch alignment carries through the generic ops ----------


def test_binary_op_aligns_sub_batch():
    # a has dyn=(2,), sub=() — sub_batch_ndim=0
    a = _foo([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    # b has dyn=(2,), sub=(3,) — sub_batch_ndim=1
    b_data = torch.tensor(
        [[[10.0, 10.0, 10.0, 10.0]] * 3] * 2,
        dtype=torch.float64,
    )
    b = _Foo(b_data, sub_batch_ndim=1)
    r = a + b
    assert r.sub_batch_ndim == 1
    assert r.data.shape == torch.Size([2, 3, 4])
    # Each (batch, sub) slot should be a + b[batch][sub] elementwise
    assert torch.equal(r.data[0][0], torch.tensor([11.0, 11.0, 11.0, 11.0], dtype=torch.float64))
    assert torch.equal(r.data[1][2], torch.tensor([12.0, 12.0, 12.0, 12.0], dtype=torch.float64))


# ---------- region views inherited from TensorWrapper still work ----------


def test_inherited_region_views():
    f = _Foo.zeros(2, 3).sub_batch.retag(1)  # dyn=(2,), sub=(3,)
    assert f.dynamic_batch.shape == torch.Size([2])
    assert f.sub_batch.shape == torch.Size([3])
    assert f.base.shape == torch.Size([4])
