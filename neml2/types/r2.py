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

Arithmetic operators and ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. R2-specific:
:meth:`identity` (returns ``eye(3)``) and :meth:`__matmul__` for ``R2 @ R2``
matrix product (the only base-type that has square base shape and thus
supports the matrix-product operator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import align_sub_batch
from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class R2(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 3, 3)``."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 2
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3, 3)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> R2:
        return cls(torch.eye(3, dtype=dtype, device=device))

    def __matmul__(self, other):
        """``R2 @ R2 → R2`` (matrix product). Other operands fall through.

        ``align_sub_batch`` pads the LHS/RHS so e.g. a global ``(B, 3, 3)`` and
        a per-crystal ``(B, 5, 3, 3)`` matmul aligns to ``(B, 1, 3, 3)`` vs
        ``(B, 5, 3, 3)`` and broadcasts to ``(B, 5, 3, 3)``.
        """
        if isinstance(other, R2):
            [aa, bb], sb = align_sub_batch(self, other)
            return R2(aa.data @ bb.data, sub_batch_ndim=sb)
        return NotImplemented


register(R2)
