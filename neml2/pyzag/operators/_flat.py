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

"""Flat-tensor packing between pyzag's boundary and neml2 ``AssembledVector``."""

from __future__ import annotations

from math import prod

import torch

from neml2.es import AssembledVector
from neml2.es._helpers import _storage_size
from neml2.es.axis_layout import AxisLayout
from neml2.types import Tensor
from neml2.types._boundary import to_torch


def _group_intmd_dim(layout: AxisLayout, g: int) -> int:
    """Number of intermediate (sub-batch) dims for group ``g`` (0 for dense)."""
    if layout.structure[g] != "block":
        return 0
    return len(layout.group_sub_batch_shape(g))


def _group_flat_size(layout: AxisLayout, g: int) -> int:
    """Flat DOF count of group ``g`` (sub-batch extent times base storage)."""
    names = layout.groups[g]
    structure = layout.structure[g]
    total = 0
    for name in names:
        base = _storage_size(layout.type_of(name))
        if structure == "block":
            total += base
        else:
            sub = prod(int(s) for s in layout.sub_batch_shape(name)) or 1
            total += base * sub
    if structure == "block":
        sub_g = prod(int(s) for s in layout.group_sub_batch_shape(g)) or 1
        total *= sub_g
    return total


def _layout_flat_size(layout: AxisLayout) -> int:
    """Total flat DOF count summed across all groups."""
    return sum(_group_flat_size(layout, g) for g in range(layout.ngroup))


def _av_to_flat(av: AssembledVector) -> torch.Tensor:
    """Flatten an ``AssembledVector`` to a single ``(*dyn, nflat)`` tensor."""
    parts = [to_torch(t).flatten(start_dim=t.batch_ndim) for t in av.tensors]
    return torch.cat(parts, dim=-1)


def _split_flat_to_av(flat: torch.Tensor, layout: AxisLayout) -> AssembledVector:
    """Inverse of :func:`_av_to_flat`: split into per-group ``AssembledVector`` tensors."""
    dyn_dim = flat.ndim - 1
    group_sizes = [_group_flat_size(layout, g) for g in range(layout.ngroup)]
    group_flats = torch.split(flat, group_sizes, dim=-1)
    tensors = []
    for g, gflat in enumerate(group_flats):
        if layout.structure[g] == "block":
            sub_shape = tuple(int(s) for s in layout.group_sub_batch_shape(g))
            sub_numel = prod(sub_shape) or 1
            base = gflat.shape[-1] // sub_numel
            reshaped = gflat.reshape(*gflat.shape[:-1], *sub_shape, base)
            tensors.append(Tensor(reshaped, batch_ndim=dyn_dim, sub_batch_ndim=len(sub_shape)))
        else:
            tensors.append(Tensor(gflat, batch_ndim=dyn_dim, sub_batch_ndim=0))
    return AssembledVector(layout, tensors)
