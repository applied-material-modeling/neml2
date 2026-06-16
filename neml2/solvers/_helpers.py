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

"""Private reduction / scaling helpers used by the line-search solver."""

from __future__ import annotations

import torch

from neml2.es import AssembledVector
from neml2.types import Tensor


def _dot(a: AssembledVector, b: AssembledVector) -> torch.Tensor:
    """Batched dot product across all assembled-vector groups.

    Per-group contributions reduce over BOTH base and sub_batch axes so
    the result has the dynamic-batch shape only. Mirrors ``norm_sq``'s
    reduction: per-group sub_batch axes get summed out before adding
    across groups, so a per-grain group's contribution doesn't broadcast
    grain into a global group's per-element value during the line search.
    """
    from neml2.types.functions import dot as _tensor_dot  # noqa: PLC0415

    total: torch.Tensor | None = None
    for ta, tb in zip(a.tensors, b.tensors, strict=True):
        contribution = _tensor_dot(ta.base, tb.base).data  # shape (*batch, *sub_batch)
        # Both ta and tb share batch/sub_batch by AssembledVector contract;
        # take ta's sub_batch_ndim as authoritative.
        if ta.sub_batch_ndim > 0:
            contribution = contribution.sum(dim=tuple(range(-ta.sub_batch_ndim, 0)))
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("Cannot dot empty AssembledVectors.")
    return total


def _scale_assembled(v: AssembledVector, alpha: torch.Tensor) -> AssembledVector:
    """Scale each group of an ``AssembledVector`` by per-element ``alpha``.

    ``alpha`` has shape ``(*B,)`` -- broadcast over each group's trailing
    base dim.
    """
    scaled = [
        Tensor(
            t.data * alpha.unsqueeze(-1), batch_ndim=t.batch_ndim, sub_batch_ndim=t.sub_batch_ndim
        )
        for t in v.tensors
    ]
    return AssembledVector(v.layout, scaled)


__all__ = ["_dot", "_scale_assembled"]
