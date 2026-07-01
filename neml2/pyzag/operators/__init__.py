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

"""pyzag block-operator backend backed by neml2 assembled types.

Implements pyzag's ``BlockVector`` / ``SolvableBlockOperator`` / ``BlockJacobian``
ABCs on top of neml2's ``AssembledVector`` / ``AssembledMatrix``, solving the
diagonal block with a cached batched LU (single group) or neml2's
``SchurComplement`` (two-group Schur split). Parallel cyclic reduction is
supported for single-group dense layouts; other layouts use the Thomas
factorization.
"""

from ._assembly import (
    _require_le_one_intmd,
    _select_dynamic_am,
    _select_dynamic_av,
    _transpose_am,
)
from ._cache import CachingLU
from ._flat import (
    _av_to_flat,
    _group_flat_size,
    _group_intmd_dim,
    _layout_flat_size,
    _split_flat_to_av,
)
from ._jacobian import NEML2BlockJacobian
from ._operator import NEML2SolvableBlockOperator
from ._vector import NEML2BlockVector
from ._wrapper import NEML2Wrapper

__all__ = [
    "CachingLU",
    "NEML2BlockVector",
    "NEML2SolvableBlockOperator",
    "NEML2BlockJacobian",
    "NEML2Wrapper",
    "_av_to_flat",
    "_split_flat_to_av",
    "_layout_flat_size",
    "_group_flat_size",
    "_group_intmd_dim",
    "_transpose_am",
    "_select_dynamic_am",
    "_select_dynamic_av",
    "_require_le_one_intmd",
]
