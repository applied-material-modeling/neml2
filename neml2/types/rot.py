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

MRPs aren't a vector space in the rotation-composition sense, but the
underlying 3-vector storage supports the usual scalar arithmetic. The Newton
residual in ``WR2ImplicitExponentialTimeIntegration`` uses ``-`` between Rots
as elementwise 3-vector subtraction (correct as long as both sides are MRPs
of the same orientation up to roundoff). Composition itself goes through the
free :func:`compose` in :mod:`functions`.

Arithmetic operators and ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. The only Rot-specific
factory is :meth:`identity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class Rot(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` in MRP packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    sub_batch_state: tuple = ()
    sub_batch_meta: tuple = ()
    k_ndim: int = 0
    k_state: tuple = ()
    k_pairing: tuple = ()
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> Rot:
        """The identity rotation — the zero MRP vector."""
        return cls(torch.zeros(3, dtype=dtype, device=device))


register(Rot)
