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

"""SR2 — symmetric rank-2 tensor in Mandel packing.

Base shape ``(6,)``; packing ``[xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz, sqrt(2)*xy]``
to match ``include/neml2/tensors/SR2.h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch
from neml2.types._pytree import register
from neml2.types.scalar import Scalar


@dataclass(frozen=True, eq=False)
class SR2(TensorWrapper):
    """Wraps a `torch.Tensor` of shape ``(..., 6)`` in Mandel packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (6,)

    # ---- factories ----

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SR2:
        return cls(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))

    @classmethod
    def zeros(
        cls, *batch: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SR2:
        return cls(torch.zeros(*batch, 6, dtype=dtype, device=device))

    @classmethod
    def fill(
        cls,
        *components: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> SR2:
        """Build an SR2 from 1, 3, or 6 tensor components (mirrors C++ ``SR2::fill``).

        * 1 value ``a`` -> ``diag(a, a, a)``.
        * 3 values -> the three diagonal entries, zero shear.
        * 6 values ``s11 s22 s33 s23 s13 s12`` -> the full symmetric tensor; the
          three shear entries are scaled by ``sqrt(2)`` into Mandel storage.

        This is the native translation of the C++ ``FillSR2`` user tensor.
        """
        vals = [float(c) for c in components]
        if len(vals) == 1:
            data = [vals[0], vals[0], vals[0], 0.0, 0.0, 0.0]
        elif len(vals) == 3:
            data = [vals[0], vals[1], vals[2], 0.0, 0.0, 0.0]
        elif len(vals) == 6:
            s = 2.0**0.5
            data = [vals[0], vals[1], vals[2], s * vals[3], s * vals[4], s * vals[5]]
        else:
            raise ValueError(f"SR2.fill expects 1, 3, or 6 components, got {len(vals)}")
        return cls(torch.tensor(data, dtype=dtype, device=device))

    # ---- operator overloads ----
    #
    # Every binary op routes through :func:`align_sub_batch` so global and
    # per-sub-batch-site operands combine cleanly at any dynamic batch size.
    # Mirrors C++ ``utils::align_intmd_dim`` — every typed-tensor operator in
    # ``src/neml2/tensors/functions/operators.cxx`` does the same.

    def __add__(self, other) -> SR2:
        if isinstance(other, SR2):
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data + bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            # ``Scalar + SR2``: broadcast the Scalar across SR2's trailing
            # (6,) base. Mirrors the multiply path; see align_scalar_base
            # for the 0-d-aware unsqueeze.
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data + align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __radd__(self, other) -> SR2:
        return self.__add__(other)

    def __sub__(self, other) -> SR2:
        if isinstance(other, SR2):
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data - bb.data, sub_batch_ndim=sb)
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data - align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        return NotImplemented

    def __neg__(self) -> SR2:
        return SR2(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    def __mul__(self, other: Scalar | float | int) -> SR2:
        if isinstance(other, Scalar):
            # Scalar-mixed product: align sub-batch, then base-unsqueeze the
            # Scalar's data to broadcast against the SR2's trailing (6,) base.
            # Mirrors C++ operators.cxx:150-160 (Scalar unsqueezed across the base dim).
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data * align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return SR2(self.data * other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented  # type: ignore[return-value]

    def __rmul__(self, other: Scalar | float | int) -> SR2:
        return self.__mul__(other)

    def __truediv__(self, other) -> SR2:
        if isinstance(other, Scalar):
            [aa, bb], sb = align_sub_batch(self, other)
            return SR2(aa.data / align_scalar_base(bb.data, 1), sub_batch_ndim=sb)
        if isinstance(other, (float, int)):
            return SR2(self.data / other, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented


register(SR2)
