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

"""Boundary wrapper converting pyzag flat torch tensors to neml2-backed blocks."""

from __future__ import annotations

import torch
from pyzag.operators.base import BlockVector

from neml2.es import AssembledMatrix
from neml2.es.axis_layout import AxisLayout

from ._flat import _av_to_flat, _split_flat_to_av
from ._jacobian import NEML2BlockJacobian
from ._vector import NEML2BlockVector


class NEML2Wrapper:
    """Convert between pyzag flat torch tensors and neml2-backed block objects."""

    def __init__(self, layout: AxisLayout) -> None:
        self.layout = layout

    def wrap_vector(self, raw: torch.Tensor) -> NEML2BlockVector:
        """Flat torch ``(..., nflat)`` -> :class:`NEML2BlockVector`."""
        return NEML2BlockVector.from_av(_split_flat_to_av(raw, self.layout))

    def unwrap_vector(self, bv: BlockVector) -> torch.Tensor:
        """:class:`NEML2BlockVector` -> flat torch ``(..., nflat)``."""
        if not isinstance(bv, NEML2BlockVector):
            raise TypeError("NEML2Wrapper.unwrap_vector requires NEML2BlockVector.")
        return _av_to_flat(bv.to_av())

    def wrap_jacobian(self, diag: AssembledMatrix, sub: AssembledMatrix) -> NEML2BlockJacobian:
        """Wrap diag / sub ``AssembledMatrix`` blocks into an :class:`NEML2BlockJacobian`."""
        return NEML2BlockJacobian(diag, sub, self.layout)
