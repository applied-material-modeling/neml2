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

"""Vec — 3-vector in Cartesian coordinates.

Base shape ``(3,)``. Mirrors ``include/neml2/tensors/Vec.h``. Used for
displacement / traction triples, position vectors, and any other Cartesian
3-vector quantity that isn't a Rodrigues rotation (``Rot``), a skew axial
vector (``WR2``), or a Miller-index triple (``MillerIndex``) — the storage
shape is identical to those three but the type tag keeps cross-type operator
dispatch unambiguous.

Methods on the class are limited to constructors, shape/dim traits, and
operators. Math-bearing operations (norm, dot, cross, rotation, etc.) live
in :mod:`functions` when needed; the migrated leaves so far only need the
basic algebra below plus inline ``stack``/index access on ``.data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch
from neml2.types._pytree import register
from neml2.types.scalar import Scalar


@dataclass(frozen=True, eq=False)
class Vec(TensorWrapper):
    """Wraps a `torch.Tensor` of shape ``(..., 3)``."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    # ---- factories ----

    @classmethod
    def zeros(
        cls, *batch: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> Vec:
        return cls(torch.zeros(*batch, 3, dtype=dtype, device=device))

    # ---- operator overloads ----
    #
    # Every binary op routes through :func:`align_sub_batch` so global and
    # per-sub-batch-site operands combine cleanly at any dynamic batch size
    # (mirrors C++ ``utils::align_intmd_dim``). Reflected operators are spelled
    # as explicit ``def`` methods (not ``__radd__ = __add__`` aliases) so the
    # concrete ``-> Vec`` return type narrows past the base's
    # ``-> TensorWrapper`` declaration for type-checkers.

    def __add__(self, other) -> Vec:
        if isinstance(other, Vec):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data + align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __radd__(self, other) -> Vec:
        return self.__add__(other)

    def __sub__(self, other) -> Vec:
        if isinstance(other, Vec):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data - align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __rsub__(self, other) -> Vec:
        # No float/int branch — Vec - scalar isn't meaningful for a 3-vector
        # (the SR2/WR2/R2 wrappers also omit it). Reflected from the Vec - Vec
        # path: if we get here, `other` failed isinstance(Vec) and there's
        # nothing sensible to do.
        return NotImplemented

    def __neg__(self) -> Vec:
        return Vec(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __mul__(self, other: Scalar | float | int) -> Vec:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data * align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Vec(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: Scalar | float | int) -> Vec:
        return self.__mul__(other)

    def __truediv__(self, other) -> Vec:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return Vec(aa.data / align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return Vec(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented


register(Vec)
