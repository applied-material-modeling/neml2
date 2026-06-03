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

"""Scalar — a physically-meaningful 0-base-shape tensor.

`Scalar` is a wrapper (not a `torch.Tensor` alias) so cross-type operator
dispatch (`Scalar * SR2 → SR2`) is deterministic via Python's reflected-operator
protocol. With a bare `torch.Tensor` on the left, Python would never invoke
`SR2.__rmul__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar, overload

import torch

from neml2.types._base import TensorWrapper, align_sub_batch
from neml2.types._pytree import register

_WrapperT = TypeVar("_WrapperT", bound=TensorWrapper)


@dataclass(frozen=True, eq=False)
class Scalar(TensorWrapper):
    """Wraps a `torch.Tensor` of base shape ``()`` (i.e., one number per batch entry).

    ``data`` may be passed as a torch tensor (used as-is), a Python number,
    or anything `torch.as_tensor` accepts. Numeric literals are promoted to
    ``torch.float64`` unless ``dtype`` overrides it — the same default used
    everywhere else in NEML2, picked to match the precision of the typed
    wrappers' algebra and the input-file ``[Tensors]`` factory.
    """

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 0
    BASE_SHAPE: ClassVar[tuple[int, ...]] = ()

    def __init__(
        self,
        data,
        sub_batch_ndim: int = 0,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(data, torch.Tensor):
            if dtype is not None or device is not None:
                data = data.to(dtype=dtype, device=device)
        else:
            data = torch.as_tensor(data, dtype=dtype or torch.float64, device=device)
        # The class is `@dataclass(frozen=True)` so direct attribute writes
        # are forbidden; route through `object.__setattr__` to seat the
        # dataclass-declared fields.
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sub_batch_ndim", sub_batch_ndim)

    @classmethod
    def from_value(cls, x: float | int, *, like: TensorWrapper) -> Scalar:
        """Construct a Scalar inheriting dtype/device from an existing wrapper."""
        return cls(x, dtype=like.dtype, device=like.device)

    # ---- arithmetic with Scalar / float / int ----
    #
    # Every binary op routes through :func:`align_sub_batch` so a global
    # ``Scalar`` and a per-sub-batch-site ``Scalar`` combine cleanly at any
    # dynamic batch size (mirrors C++ ``utils::align_intmd_dim`` —
    # ``include/neml2/tensors/functions/utils.h:36``).

    def __add__(self, other) -> Scalar:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Scalar(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Scalar(self.data + other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    def __radd__(self, other) -> Scalar:
        return self.__add__(other)

    def __sub__(self, other) -> Scalar:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Scalar(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Scalar(self.data - other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    def __rsub__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(other - self.data, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    @overload
    def __mul__(self, other: Scalar | float | int) -> Scalar: ...
    @overload
    def __mul__(self, other: _WrapperT) -> _WrapperT: ...
    def __mul__(self, other):
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Scalar(aa.data * bb.data, sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Scalar(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        if isinstance(other, TensorWrapper):
            # ``Scalar * wrapper`` scales the wrapper. Delegate to the wrapper's
            # own (correctly base-shaped) scalar multiply — the same result the
            # reflected-operator protocol would reach, but stated explicitly so
            # ``Scalar * SR2 -> SR2`` and, crucially, ``Scalar * TensorWrapper``
            # in generic model code both type-check. Leaf authors can write the
            # scalar on either side.
            return other * self
        return NotImplemented

    def __rmul__(self, other) -> Scalar:
        # Mirrors __mul__: float/int → Scalar; the wrapper branch in __mul__
        # delegates to ``other * self`` and only triggers when ``other`` is a
        # non-Scalar TensorWrapper, which can't happen on the rmul path
        # (those wrappers' own __mul__ handles Scalar directly).
        return self.__mul__(other)

    def __truediv__(self, other) -> Scalar:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Scalar(aa.data / bb.data, sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Scalar(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    def __rtruediv__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(other / self.data, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    def __neg__(self) -> Scalar:
        return Scalar(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __abs__(self) -> Scalar:
        return Scalar(torch.abs(self.data), sub_batch_ndim=self.sub_batch_ndim)

    def __pow__(self, n: float | int) -> Scalar:
        return Scalar(torch.pow(self.data, n), sub_batch_ndim=self.sub_batch_ndim)


register(Scalar)
