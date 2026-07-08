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
        return self.raw_tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.raw_tensors[0].dtype

    @property
    def nblk(self) -> int:
        return self.raw_tensors[0].shape[0]

    @property
    def batch_size(self) -> int:
        return self.raw_tensors[0].shape[1]

    @property
    def block_size(self) -> int:
        total = 0
        for t in self.raw_tensors:
            g = 1
            for d in t.shape[2:]:
                g *= d
            total += g
        return total

    def clone(self) -> NEML2BlockVector:
        return NEML2BlockVector([t.clone() for t in self.raw_tensors], self.layout, self.intmd_dims)

    def norm(self, dim: int = -1) -> torch.Tensor:
        """Combined L2 norm over the whole multi-group state, per block and batch."""
        per_group_sq = []
        for t, i in zip(self.raw_tensors, self.intmd_dims, strict=True):
            flat = t.flatten(start_dim=-(1 + i))
            per_group_sq.append(torch.norm(flat, dim=dim) ** 2)
        return torch.stack(per_group_sq, dim=0).sum(dim=0).sqrt()

    def flat_norm(self) -> torch.Tensor:
        """Combined cross-block L2 norm per batch over the whole state."""
        per_group_sq = []
        for t in self.raw_tensors:
            flat = t.transpose(0, 1).flatten(1)
            per_group_sq.append(torch.norm(flat, dim=-1) ** 2)
        return torch.stack(per_group_sq, dim=0).sum(dim=0).sqrt()

    def where(self, mask: torch.Tensor, other: BlockVector) -> NEML2BlockVector:
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.where expects NEML2BlockVector.")
        out = []
        for t_self, t_other in zip(self.raw_tensors, other.raw_tensors, strict=True):
            shape = (1, -1) + (1,) * (t_self.ndim - 2)
            out.append(torch.where(mask.reshape(shape), t_self, t_other))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def scale_batches(self, factor: torch.Tensor) -> NEML2BlockVector:
        out = []
        for t in self.raw_tensors:
            shape = (1, -1) + (1,) * (t.ndim - 2)
            out.append(t * factor.reshape(shape))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def flip(self, dim: int) -> NEML2BlockVector:
        return NEML2BlockVector(
            [t.flip(dim) for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def __neg__(self) -> NEML2BlockVector:
        return NEML2BlockVector([-t for t in self.raw_tensors], self.layout, self.intmd_dims)

    def __add__(self, other: BlockVector) -> NEML2BlockVector:
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only add to NEML2BlockVector.")
        return NEML2BlockVector(
            [a + b for a, b in zip(self.raw_tensors, other.raw_tensors, strict=True)],
            self.layout,
            self.intmd_dims,
        )

    def __sub__(self, other: BlockVector) -> NEML2BlockVector:
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only subtract NEML2BlockVector.")
        return NEML2BlockVector(
            [a - b for a, b in zip(self.raw_tensors, other.raw_tensors, strict=True)],
            self.layout,
            self.intmd_dims,
        )

    def __mul__(self, other: torch.Tensor | float | int) -> NEML2BlockVector:
        return NEML2BlockVector([t * other for t in self.raw_tensors], self.layout, self.intmd_dims)

    def __getitem__(self, idx: int | slice) -> NEML2BlockVector:
        out = []
        for t in self.raw_tensors:
            sliced = t[idx]
            if isinstance(idx, int) or (sliced.ndim < t.ndim):
                sliced = sliced.unsqueeze(0)
            out.append(sliced)
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def __setitem__(self, idx: int | slice, value: BlockVector) -> None:
        if not isinstance(value, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only assign from NEML2BlockVector.")
        for t_self, t_val in zip(self.raw_tensors, value.raw_tensors, strict=True):
            t_self[idx] = t_val

    @classmethod
    def cat(cls, vectors: Sequence[BlockVector], dim: int = 0) -> NEML2BlockVector:
        if not vectors:
            raise ValueError("cat requires at least one vector")
        for v in vectors:
            if not isinstance(v, NEML2BlockVector):
                raise TypeError("All vectors must be NEML2BlockVector.")
        first = vectors[0]
        out = []
        for g in range(first.layout.ngroup):
            out.append(torch.cat([v.raw_tensors[g] for v in vectors], dim=dim))
        return NEML2BlockVector(out, first.layout, first.intmd_dims)

    @classmethod
    def zeros_like(cls, other: BlockVector) -> NEML2BlockVector:
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.zeros_like requires NEML2BlockVector.")
        return NEML2BlockVector(
            [torch.zeros_like(t) for t in other.raw_tensors],
            other.layout,
            other.intmd_dims,
        )
