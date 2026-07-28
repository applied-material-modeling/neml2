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

"""pyzag ``BlockVector`` backed by a neml2 ``AssembledVector``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from pyzag.operators.base import BlockVector

from neml2.es import AssembledVector
from neml2.es.axis_layout import AxisLayout
from neml2.types import Tensor
from neml2.types._boundary import to_torch

from ._flat import _group_intmd_dim


class NEML2BlockVector(BlockVector):
    """Block vector held as per-group torch tensors with neml2 layout metadata.

    Per-group torch tensors plus explicit ``intmd_dims`` mirror the dense
    backend pattern and let ``__setitem__`` / ``clone`` work in place. A neml2
    ``AssembledVector`` is materialized only at the neml2 boundary (via
    :meth:`to_av`), where the linear solvers and matmul consume it.
    """

    def __init__(
        self,
        raw_tensors: list[torch.Tensor],
        layout: AxisLayout,
        intmd_dims: list[int] | None = None,
    ) -> None:
        """Hold one raw ``(nblk, batch, *intmd, dofs)`` tensor per layout group.

        Args:
            raw_tensors: one tensor per group of ``layout`` (length must match ``layout.ngroup``).
            layout: the neml2 :class:`~neml2.es.axis_layout.AxisLayout` describing the groups.
            intmd_dims: number of intermediate (sub-batch/site) dims per group; defaults to
                each group's layout-declared value.
        """
        if intmd_dims is None:
            intmd_dims = [_group_intmd_dim(layout, g) for g in range(layout.ngroup)]
        if len(raw_tensors) != layout.ngroup:
            raise ValueError(
                f"NEML2BlockVector expects {layout.ngroup} per-group tensors, "
                f"got {len(raw_tensors)}."
            )
        if len(intmd_dims) != layout.ngroup:
            raise ValueError(
                f"intmd_dims length ({len(intmd_dims)}) must match layout.ngroup ({layout.ngroup})."
            )
        self.raw_tensors = list(raw_tensors)
        self.layout = layout
        self.intmd_dims = list(intmd_dims)

    def to_av(self) -> AssembledVector:
        """Materialize as a neml2 ``AssembledVector``."""
        tensors = [
            Tensor(t, batch_ndim=t.ndim - i - 1, sub_batch_ndim=i)
            for t, i in zip(self.raw_tensors, self.intmd_dims, strict=True)
        ]
        return AssembledVector(self.layout, tensors)

    @classmethod
    def from_av(cls, av: AssembledVector) -> NEML2BlockVector:
        """Construct from a neml2 ``AssembledVector``."""
        intmd_dims = [t.sub_batch_ndim for t in av.tensors]
        raw_tensors = [to_torch(t) for t in av.tensors]
        return cls(raw_tensors, av.layout, intmd_dims)

    @property
    def device(self) -> torch.device:
        """Device of the backing tensors."""
        return self.raw_tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the backing tensors."""
        return self.raw_tensors[0].dtype

    @property
    def nblk(self) -> int:
        """Number of blocks along the dynamic (time) axis."""
        return self.raw_tensors[0].shape[0]

    @property
    def batch_size(self) -> int:
        """Size of the plain (non-dynamic) batch axis."""
        return self.raw_tensors[0].shape[1]

    @property
    def block_size(self) -> int:
        """Total per-block degrees of freedom summed across all groups (incl. intmd dims)."""
        total = 0
        for t in self.raw_tensors:
            g = 1
            for d in t.shape[2:]:
                g *= d
            total += g
        return total

    def clone(self) -> NEML2BlockVector:
        """Deep copy, cloning every backing tensor."""
        return NEML2BlockVector([t.clone() for t in self.raw_tensors], self.layout, self.intmd_dims)

    def norm(self, dim: int = -1) -> torch.Tensor:
        """Combined L2 norm over the whole multi-group state, per block and batch."""
        per_group_sq = []
        for t, i in zip(self.raw_tensors, self.intmd_dims, strict=True):
            flat = t.flatten(start_dim=-(1 + i))
            per_group_sq.append(torch.norm(flat, dim=dim) ** 2)
        return torch.stack(per_group_sq, dim=0).sum(dim=0).sqrt()

    def flatten(self) -> torch.Tensor:
        """Concatenate all groups into one ``(batch, nblk * dofs)`` tensor.

        Each group is transposed to put the batch axis first, then flattened and
        concatenated along the feature axis -- the flat form pyzag's PCR path expects.
        """
        return torch.cat([t.transpose(0, 1).flatten(1) for t in self.raw_tensors], dim=-1)

    def where(self, mask: torch.Tensor, other: BlockVector) -> NEML2BlockVector:
        """Batchwise select: keep ``self`` where ``mask`` is true, else ``other``."""
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.where expects NEML2BlockVector.")
        out = []
        for t_self, t_other in zip(self.raw_tensors, other.raw_tensors, strict=True):
            shape = (1, -1) + (1,) * (t_self.ndim - 2)
            out.append(torch.where(mask.reshape(shape), t_self, t_other))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def scale_batches(self, factor: torch.Tensor) -> NEML2BlockVector:
        """Scale each plain-batch entry by the matching entry of ``factor``."""
        out = []
        for t in self.raw_tensors:
            shape = (1, -1) + (1,) * (t.ndim - 2)
            out.append(t * factor.reshape(shape))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def flip(self, dim: int) -> NEML2BlockVector:
        """Reverse every group along ``dim`` (used to walk time backward in the adjoint)."""
        return NEML2BlockVector(
            [t.flip(dim) for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def __neg__(self) -> NEML2BlockVector:
        """Elementwise negation."""
        return NEML2BlockVector([-t for t in self.raw_tensors], self.layout, self.intmd_dims)

    def __add__(self, other: BlockVector) -> NEML2BlockVector:
        """Elementwise sum of two matching-layout block vectors."""
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only add to NEML2BlockVector.")
        return NEML2BlockVector(
            [a + b for a, b in zip(self.raw_tensors, other.raw_tensors, strict=True)],
            self.layout,
            self.intmd_dims,
        )

    def __sub__(self, other: BlockVector) -> NEML2BlockVector:
        """Elementwise difference of two matching-layout block vectors."""
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only subtract NEML2BlockVector.")
        return NEML2BlockVector(
            [a - b for a, b in zip(self.raw_tensors, other.raw_tensors, strict=True)],
            self.layout,
            self.intmd_dims,
        )

    def __mul__(self, other: torch.Tensor | float | int) -> NEML2BlockVector:
        """Scale every group by a scalar or broadcastable tensor."""
        return NEML2BlockVector([t * other for t in self.raw_tensors], self.layout, self.intmd_dims)

    def __getitem__(self, idx: int | slice) -> NEML2BlockVector:
        """Slice along the dynamic (time) axis, keeping that axis (an int index is re-expanded)."""
        out = []
        for t in self.raw_tensors:
            sliced = t[idx]
            if isinstance(idx, int) or (sliced.ndim < t.ndim):
                sliced = sliced.unsqueeze(0)
            out.append(sliced)
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def __setitem__(self, idx: int | slice, value: BlockVector) -> None:
        """In-place write into the ``idx`` slice of the dynamic axis, per group."""
        if not isinstance(value, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only assign from NEML2BlockVector.")
        for t_self, t_val in zip(self.raw_tensors, value.raw_tensors, strict=True):
            t_self[idx] = t_val

    @classmethod
    def cat(cls, vectors: Sequence[BlockVector], dim: int = 0) -> NEML2BlockVector:
        """Concatenate matching-layout block vectors group-by-group along ``dim``."""
        if not vectors:
            raise ValueError("cat requires at least one vector")
        for v in vectors:
            if not isinstance(v, NEML2BlockVector):
                raise TypeError("All vectors must be NEML2BlockVector.")
        typed = cast(list[NEML2BlockVector], vectors)
        first = typed[0]
        out = []
        for g in range(first.layout.ngroup):
            out.append(torch.cat([v.raw_tensors[g] for v in typed], dim=dim))
        return NEML2BlockVector(out, first.layout, first.intmd_dims)

    @classmethod
    def zeros_like(cls, other: BlockVector) -> NEML2BlockVector:
        """A zero vector with the same layout, shapes, dtype, and device as ``other``."""
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.zeros_like requires NEML2BlockVector.")
        return NEML2BlockVector(
            [torch.zeros_like(t) for t in other.raw_tensors],
            other.layout,
            other.intmd_dims,
        )
