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

"""Variable-group layout for the equation-system blocks.

:class:`AxisLayout` ties a tuple of variable groups to their tensor
types and per-variable sub-batch shapes. :class:`~AssembledVector` and
:class:`~AssembledMatrix` carry one of these to know how to disassemble
their flat per-group tensors back into named variables.

Per-group ``SubBatchStructure`` (v2-parity, mirrors v2's ``IStructure``)
-------------------------------------------------------------------

Each variable group declares ``structure: SubBatchStructure`` -- either
``"block"`` or ``"dense"``:

* ``"block"`` -- the group's sub_batch axes are PRESERVED as
  intermediate dims on the per-group assembled tensor. Storage shape
  is ``(*dyn, *sub_batch, group_storage_size)`` for a vector or
  ``(*dyn, *intmd_row, *intmd_col, row_storage, col_storage)`` for a
  matrix block. Used when sub_batch axes carry per-site independence
  the solver should preserve (per-grain in polycrystal, per-cell in
  finite-volume, per-bin in KWN). The implicit block matmul reduces
  via ``inner / sum_sub_batch`` along the intmd axes for an inner
  group whose ``structure == "block"``.
* ``"dense"`` -- the group's sub_batch axes are FOLDED into the base
  storage. Shape is ``(*dyn, group_storage_size_with_sub_folded)``
  for a vector. Used for groups whose sub_batch axes don't represent
  per-site independence (typically global unknowns / forces).

The user specifies ``structure`` in the HIT ``[EquationSystems]`` block
(``structure = 'block dense'`` -- space-separated, one token per group, in
group order). Defaults to all ``"dense"`` when omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from neml2.types import TensorWrapper

from ._helpers import _storage_size

#: Per-group structural flag for sub_batch axes. Mirrors v2's
#: ``AxisLayout::IStructure``.
SubBatchStructure: TypeAlias = Literal["block", "dense"]


@dataclass(frozen=True)
class AxisLayout:
    """Ordered variable groups, their tensor types, per-variable
    sub-batch shapes, and per-group :data:`SubBatchStructure`.
    """

    groups: tuple[tuple[str, ...], ...]
    specs: dict[str, type[TensorWrapper]]
    sub_batch_shapes: dict[str, torch.Size]
    structure: tuple[SubBatchStructure, ...]

    def __init__(
        self,
        groups: list[list[str]] | tuple[tuple[str, ...], ...],
        specs: dict[str, type[TensorWrapper]],
        sub_batch_shapes: dict[str, torch.Size] | None = None,
        structure: tuple[SubBatchStructure, ...] | list[SubBatchStructure] | None = None,
    ) -> None:
        normalized = tuple(tuple(group) for group in groups)
        missing = [name for group in normalized for name in group if name not in specs]
        if missing:
            raise KeyError(f"AxisLayout variables missing from specs: {missing}")
        sub = {
            name: torch.Size(sub_batch_shapes.get(name, ()))
            if sub_batch_shapes is not None
            else torch.Size(())
            for group in normalized
            for name in group
        }
        if structure is None:
            structure_tuple: tuple[SubBatchStructure, ...] = ("dense",) * len(normalized)
        else:
            structure_tuple = tuple(structure)
            if len(structure_tuple) != len(normalized):
                raise ValueError(
                    f"AxisLayout: structure has {len(structure_tuple)} entries, expected "
                    f"{len(normalized)} (one per group)."
                )
            for k in structure_tuple:
                if k not in ("block", "dense"):
                    raise ValueError(
                        f"AxisLayout: structure entries must be 'block' or 'dense', got {k!r}."
                    )
        object.__setattr__(self, "groups", normalized)
        object.__setattr__(self, "specs", dict(specs))
        object.__setattr__(self, "sub_batch_shapes", sub)
        object.__setattr__(self, "structure", structure_tuple)

    def with_sub_batch_shapes(
        self,
        sub_batch_shapes: dict[str, torch.Size],
    ) -> AxisLayout:
        """Return a new layout with updated sub-batch shapes (frozen replacement)."""
        return AxisLayout(self.groups, self.specs, sub_batch_shapes, self.structure)

    def sub_layout(self, index: int) -> AxisLayout:
        """Single-group sub-layout containing only ``self.groups[index]``."""
        group = self.groups[index]
        specs = {name: self.specs[name] for name in group}
        sub_batch = {name: self.sub_batch_shapes[name] for name in group}
        return AxisLayout([list(group)], specs, sub_batch, (self.structure[index],))

    @property
    def ngroup(self) -> int:
        return len(self.groups)

    @property
    def nvar(self) -> int:
        return sum(len(group) for group in self.groups)

    def vars(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group)

    def group_size(self, index: int) -> int:
        return sum(self.var_size(name) for name in self.groups[index])

    def storage_size(self) -> int:
        return sum(self.group_size(i) for i in range(self.ngroup))

    def var_size(self, name: str) -> int:
        return _storage_size(self.specs[name])

    def sub_batch_shape(self, name: str) -> torch.Size:
        """Per-variable sub-batch shape (empty when the var is sub-batch-trivial)."""
        return self.sub_batch_shapes.get(name, torch.Size(()))

    def group_sub_batch_shape(self, index: int) -> torch.Size:
        """Common sub-batch shape across every variable in group ``index``.

        For a BLOCK group every variable must share the same sub_batch_shape
        (otherwise the block tensor can't be a single rectangular tensor).
        Raises on disagreement. For a DENSE group the per-variable shapes
        may differ — they get folded into base on assembly — and this
        method returns the FIRST variable's shape for shape inference only.
        """
        group = self.groups[index]
        if not group:
            return torch.Size(())
        first = self.sub_batch_shapes.get(group[0], torch.Size(()))
        if self.structure[index] == "block":
            for name in group[1:]:
                other = self.sub_batch_shapes.get(name, torch.Size(()))
                if tuple(other) != tuple(first):
                    raise ValueError(
                        f"AxisLayout: BLOCK group {index} variables disagree on "
                        f"sub_batch_shape -- {group[0]!r}={tuple(first)} vs "
                        f"{name!r}={tuple(other)}. BLOCK groups must share one "
                        "sub_batch_shape across all variables."
                    )
        return first

    def block_size(self) -> int:
        """Per-(dynamic-batch, sub-batch-site) storage size."""
        return self.storage_size()

    def type_of(self, name: str) -> type[TensorWrapper]:
        return self.specs[name]


__all__ = ["AxisLayout", "SubBatchStructure"]
