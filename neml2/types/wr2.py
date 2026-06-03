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

"""WR2 — skew-symmetric second-order tensor, stored as an axial 3-vector.

Base shape ``(3,)``. Mirrors ``include/neml2/tensors/WR2.h``. The packing
matches ``R2::skew(Vec)`` so the underlying 3x3 form is::

    [[ 0,  -w2,  w1],
     [ w2,  0,  -w0],
     [-w1,  w0,  0 ]]

where ``(w0, w1, w2)`` are the stored components. Used for vorticity (plastic
spin) and the rate variable inside the orientation exponential time
integration. Mathematical operations (``exp_map``, ``dexp_map``, conversion
to/from full ``R2``) live in :mod:`functions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch
from neml2.types._pytree import register
from neml2.types.scalar import Scalar


@dataclass(frozen=True, eq=False)
class WR2(TensorWrapper):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` storing the axial vector."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    # ---- factories ----

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> WR2:
        """The zero skew tensor — the additive identity (no canonical 'unit' skew)."""
        return cls(torch.zeros(3, dtype=dtype, device=device))

    @classmethod
    def zeros(
        cls, *batch: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> WR2:
        return cls(torch.zeros(*batch, 3, dtype=dtype, device=device))

    # ---- operator overloads ----
    #
    # Every binary op routes through :func:`align_sub_batch` so global and
    # per-sub-batch-site operands combine cleanly at any dynamic batch size
    # (mirrors C++ ``utils::align_intmd_dim``). In particular,
    # ``w - wp`` between a global vorticity ``(B, 3)`` and a per-crystal
    # plastic vorticity ``(B, 5, 3)`` aligns correctly at any ``B``.

    def __add__(self, other) -> WR2:
        if isinstance(other, WR2):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data + align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __radd__(self, other) -> WR2:
        return self.__add__(other)

    def __sub__(self, other) -> WR2:
        if isinstance(other, WR2):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data - align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __neg__(self) -> WR2:
        return WR2(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __mul__(self, other: Scalar | float | int) -> WR2:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data * align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return WR2(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: Scalar | float | int) -> WR2:
        return self.__mul__(other)

    def __truediv__(self, other) -> WR2:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return WR2(aa.data / align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return WR2(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented


register(WR2)
