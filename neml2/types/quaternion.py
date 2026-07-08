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

"""Quaternion — unit quaternion orientation, packed ``(w, x, y, z)``.

Base shape ``(4,)``. Mirrors ``include/neml2/tensors/Quaternion.h``. The scalar
part ``w`` is component 0; the vector part ``(x, y, z)`` is components 1-3. Used
mainly as the natural carrier for crystal-symmetry operator tables (see
:mod:`neml2.ops.symmetry`) and as an intermediate when sampling uniform
orientations.

Arithmetic operators and the ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. Math-bearing operations
(conversion to/from :class:`~neml2.types.MRP`, the rotation matrix, the
quaternion geodesic distance) live in :mod:`functions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class Quaternion(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 4)`` packed ``(w, x, y, z)``."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    sub_batch_state: tuple = ()
    sub_batch_meta: tuple = ()
    k_ndim: int = 0
    k_state: tuple = ()
    k_pairing: tuple = ()
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (4,)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> Quaternion:
        """The identity rotation — the quaternion ``(1, 0, 0, 0)``."""
        return cls(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))


register(Quaternion)
