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

"""Typed helpers on neml2 assembled matrices / vectors for the pyzag backend.

Supported envelope: **at most one intermediate (sub-batch) dimension per block**
-- dense blocks (0 intmd), one-sided block x dense blocks (1 intmd), and paired
single-site block x block blocks (1 shared intmd). Blocks with two or more
intermediate axes (distinct row / col intmd, i.e. coupled sites) are rejected
with a clear error rather than silently mishandled. These helpers stay typed
(``Tensor.base.transpose`` / ``Tensor.batch[...]``); they do not reach around the
typed API with raw-tensor ops.
"""

from __future__ import annotations

from neml2.es import AssembledMatrix, AssembledVector


def _require_le_one_intmd(am: AssembledMatrix, what: str = "operator") -> None:
    """Raise if any block of ``am`` carries two or more intermediate axes."""
    for i, row in enumerate(am.tensors):
        for j, t in enumerate(row):
            if t.sub_batch_ndim >= 2:
                raise NotImplementedError(
                    "The neml2 pyzag backend supports at most one intermediate "
                    f"(sub-batch) dimension per block; {what} block [{i}][{j}] has "
                    f"sub_batch_ndim={t.sub_batch_ndim}. Distinct row/col intermediate "
                    "axes (coupled sites) are not yet supported."
                )


def _transpose_am(am: AssembledMatrix) -> AssembledMatrix:
    """Block transpose: swap row/col layouts and transpose each block's base axes.
    Rejects blocks with two or more intmd axes.
    """
    _require_le_one_intmd(am, "transpose")
    return AssembledMatrix(
        am.col_layout,
        am.row_layout,
        [
            [am.tensors[i][j].base.transpose() for i in range(am.row_layout.ngroup)]
            for j in range(am.col_layout.ngroup)
        ],
    )


def _select_dynamic_am(am: AssembledMatrix, key) -> AssembledMatrix:
    """Index or slice the leading dynamic (time) axis of every block as a view."""
    return AssembledMatrix(
        am.row_layout,
        am.col_layout,
        [[t.batch[key] for t in row] for row in am.tensors],
    )


def _select_dynamic_av(av: AssembledVector, key) -> AssembledVector:
    """Index or slice the leading dynamic (time) axis of every group tensor as a view."""
    return AssembledVector(av.layout, [t.batch[key] for t in av.tensors])
