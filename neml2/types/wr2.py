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

Arithmetic operators and ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. The only WR2-specific
factory is :meth:`identity`, which returns the zero skew (additive identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class WR2(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` storing the axial vector."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> WR2:
        """The zero skew tensor — the additive identity (no canonical 'unit' skew)."""
        return cls(torch.zeros(3, dtype=dtype, device=device))


register(WR2)
