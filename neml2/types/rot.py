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

"""Rot — rotation stored as modified Rodrigues parameters (MRPs).

Base shape ``(3,)``. The three components are ``n * tan(theta/4)`` where ``n``
is the rotation axis (unit vector) and ``theta`` the rotation angle. This is
the same convention as ``include/neml2/tensors/Rot.h``; it differs from the
standard Rodrigues parameters ``n * tan(theta/2)`` (which can be obtained by
the inverse map). The zero vector is the identity rotation.

Mathematical operations on rotations (composition, exponential map, Euler
matrix conversion, derivatives) live in :mod:`neml2.types.functions`
as free functions. The class itself only carries constructors, operators,
and shape/dim traits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch
from neml2.types._pytree import register
from neml2.types.scalar import Scalar


@dataclass(frozen=True, eq=False)
class Rot(TensorWrapper):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` in MRP packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    # ---- factories ----

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> Rot:
        """The identity rotation — the zero MRP vector."""
        return cls(torch.zeros(3, dtype=dtype, device=device))

    @classmethod
    def zeros(
        cls, *batch: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> Rot:
        return cls(torch.zeros(*batch, 3, dtype=dtype, device=device))

    # ---- operator overloads ----
    #
    # MRPs aren't a vector space in the rotation-composition sense, but the
    # underlying 3-vector storage supports the usual scalar arithmetic.
    # WR2ImplicitExponentialTimeIntegration's residual ``r = s - s_n.rotate(inc)``
    # uses ``-`` between Rots as elementwise 3-vector subtraction (the residual
    # is set to zero by the Newton solve, so the algebra is correct as long as
    # both sides are MRPs of the same orientation up to roundoff). Composition
    # itself goes through the free ``compose(Rot, Rot)`` function.

    def __add__(self, other) -> Rot:
        if isinstance(other, Rot):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data + align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __radd__(self, other) -> Rot:
        return self.__add__(other)

    def __sub__(self, other) -> Rot:
        if isinstance(other, Rot):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data - align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __neg__(self) -> Rot:
        return Rot(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __mul__(self, other: Scalar | float | int) -> Rot:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data * align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Rot(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: Scalar | float | int) -> Rot:
        return self.__mul__(other)

    def __truediv__(self, other) -> Rot:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Rot(aa.data / align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Rot(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented


register(Rot)
