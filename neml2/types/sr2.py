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

Arithmetic operators and ``zeros``/``ones``/``full``/``empty`` factories are
inherited from :class:`PrimitiveTensor`. SR2-specific: :meth:`identity`
(returns the Mandel-packed unit tensor) and a Mandel-aware :meth:`fill`
override that accepts 1, 3, or 6 components with √2 shear scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class SR2(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 6)`` in Mandel packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    sub_batch_state: tuple = ()
    sub_batch_meta: tuple = ()
    k_ndim: int = 0
    k_state: tuple = ()
    k_pairing: tuple = ()
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (6,)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SR2:
        return cls(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))

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

        Overrides the generic :meth:`PrimitiveTensor.fill` to handle the short
        forms and the Mandel √2 scaling. The 6-component form is *not* a raw
        ``tensor([...]).reshape((6,))`` — the shear scaling matters.
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


register(SR2)
