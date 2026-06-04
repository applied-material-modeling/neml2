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

"""MillerIndex — a crystallographic direction or plane (h, k, l).

Base shape ``(3,)``. Mirrors ``include/neml2/tensors/MillerIndex.h``. The
stored components are the raw Miller indices (e.g. ``(1, 1, 0)`` for the
``[110]`` direction); conversion to a Cartesian lattice vector requires the
``CrystalGeometry`` lattice vectors and lives in :mod:`functions`.

The C++ class has no methods of its own — this Python wrapper is similarly
a pure labelled container, distinguished from a raw ``Vec`` only by type so
``[Tensors] type = Python`` expressions can spell ``MillerIndex(...)`` and
the eval namespace dispatches correctly. Stored as floating-point even
though the indices are integers — this matches the C++ side, which keeps
the storage in the model's default dtype to participate in autograd-friendly
expressions downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register


@dataclass(frozen=True, eq=False)
class MillerIndex(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` carrying Miller indices.

    Inherits all arithmetic and ``zeros``/``ones``/``full``/``empty``/``fill``
    factories from :class:`PrimitiveTensor`. No class-specific overrides — the
    C++ analogue has the same minimal surface.
    """

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)


register(MillerIndex)
