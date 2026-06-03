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

"""R2 — full second-order tensor (3, 3) over batch + sub-batch.

Mirrors ``include/neml2/tensors/R2.h``. Used for orientation matrices, Schmid
tensors (when the symmetric/skew split isn't taken), and any non-symmetric 3x3.

Methods on the class are limited to constructors, shape/dim traits, and
operators. Mathematical operations (det, inv, contraction with non-R2 types,
symmetric/skew projection, rotations) live in :mod:`functions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch
from neml2.types._pytree import register
from neml2.types.scalar import Scalar


@dataclass(frozen=True, eq=False)
class R2(TensorWrapper):
    """Wraps a `torch.Tensor` of shape ``(..., 3, 3)``."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 2
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3, 3)

    # ---- factories ----

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> R2:
        return cls(torch.eye(3, dtype=dtype, device=device))

    @classmethod
    def zeros(
        cls, *batch: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> R2:
        return cls(torch.zeros(*batch, 3, 3, dtype=dtype, device=device))

    # ---- operator overloads ----
    #
    # Every binary op routes through :func:`align_sub_batch` so global and
    # per-sub-batch-site operands combine cleanly at any dynamic batch size.
    # In particular ``R2 @ R2`` between a global ``(B, 3, 3)`` and per-crystal
    # ``(B, 5, 3, 3)`` is the canonical broken eager broadcast — alignment
    # pads the global to ``(B, 1, 3, 3)`` which matmul broadcasts to
    # ``(B, 5, 3, 3)`` correctly.

    def __add__(self, other) -> R2:
        if isinstance(other, R2):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data + align_scalar_base(bb.data, 2), sub_batch_ndim=sb)
        return NotImplemented

    def __radd__(self, other) -> R2:
        return self.__add__(other)

    def __sub__(self, other) -> R2:
        if isinstance(other, R2):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data - align_scalar_base(bb.data, 2), sub_batch_ndim=sb)
        return NotImplemented

    def __neg__(self) -> R2:
        return R2(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __mul__(self, other: Scalar | float | int) -> R2:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data * align_scalar_base(bb.data, 2), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return R2(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: Scalar | float | int) -> R2:
        return self.__mul__(other)

    def __truediv__(self, other) -> R2:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data / align_scalar_base(bb.data, 2), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return R2(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    def __matmul__(self, other):
        """``R2 @ R2 → R2`` (matrix product). Other operands fall through."""
        if isinstance(other, R2):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data @ bb.data, sub_batch_ndim=sb)
        return NotImplemented


register(R2)
