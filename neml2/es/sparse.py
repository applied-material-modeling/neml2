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

"""Per-variable typed wrappers around equation-system state.

``SparseVector`` and ``SparseMatrix`` are the typed duals of
:class:`~neml2.es.assembled.AssembledVector` and
:class:`~neml2.es.assembled.AssembledMatrix`. They wrap the per-variable
form (``dict`` keyed by variable name with typed-wrapper values) plus the
:class:`~neml2.es.axis_layout.AxisLayout` that names the variables, so
``(layout, values)`` rides as one object across API boundaries instead
of as two arguments that callers have to keep in sync.

The pair is symmetric and round-trip-clean::

    SparseVector(layout, values).assemble()             -> AssembledVector
    AssembledVector.disassemble()                       -> SparseVector
    SparseMatrix(row_layout, col_layout, cells).assemble() -> AssembledMatrix
    AssembledMatrix.disassemble()                       -> SparseMatrix

Use the ``Sparse*`` forms at user-facing API boundaries
(``ModelNonlinearSystem.initialize``, the ``deterministic.ipynb`` /
``statistical.ipynb`` notebooks, the pyzag adapter, anywhere a
``(layout, dict)`` pair would otherwise be passed separately); use
``Assembled*`` inside the Newton hot path and at the AOTI export
contract surface, where per-group raw tensors are the canonical
representation.

The ``__post_init__`` validation catches missing-variable bugs at
construction time rather than letting them surface as ``KeyError`` deep
in :func:`~neml2.es.assembled._build_group_block` once a ``forward``
actually tries to assemble the state.

Note on ``SparseMatrix`` vs ``ChainRuleDict``: although both have the
shape ``dict[str, dict[str, TensorWrapper-ish]]``, ``ChainRuleDict``
holds leading-K typed tangents (``k_ndim > 0``, the chain-rule seed
direction lives on the outer axes) while ``SparseMatrix.cells`` holds
the assembled per-cell :class:`~neml2.types.Tensor` blocks that
``AssembledMatrix.tensors[i][j]`` produces -- K has already been folded
into trailing base via :func:`_tangent_block_to_trailing_k`. They are
different objects despite the dict-of-dict shape coincidence; do not
unify them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neml2.types import Tensor, TensorWrapper

from .assembled import AssembledMatrix, AssembledVector
from .axis_layout import AxisLayout

if TYPE_CHECKING:
    from collections.abc import ItemsView


@dataclass(frozen=True)
class SparseVector:
    """Per-variable typed-wrapper vector; the typed dual of :class:`AssembledVector`.

    ``values`` maps each variable name listed in ``layout.vars()`` to its
    typed value (a :class:`~neml2.types.TensorWrapper` subclass instance).
    Construction validates that every layout variable is covered.

    Per CLAUDE.md rule 1: ``values`` is strictly typed -- raw
    ``torch.Tensor`` is rejected. External boundaries that have raw
    tensors (the pyzag adapter, AOTI tracer fixtures, user code in
    notebooks / tests) wrap with the appropriate ``TensorWrapper``
    subclass *at the construction site*, not by handing raw tensors to
    an internal neml2 helper.
    """

    layout: AxisLayout
    values: Mapping[str, TensorWrapper]

    def __post_init__(self) -> None:
        # Catch missing-variable bugs here rather than deep in assembly.
        missing = sorted(set(self.layout.vars()) - set(self.values.keys()))
        if missing:
            raise KeyError(
                f"SparseVector: missing values for variables {missing}; "
                f"layout declares {list(self.layout.vars())}, "
                f"got {sorted(self.values.keys())}."
            )

    def assemble(self) -> AssembledVector:
        """Stack ``values`` into per-group tensors via the layout's grouping."""
        return AssembledVector.from_dict(self.layout, self.values)

    def to(self, *args, **kwargs) -> SparseVector:
        """Move every value to a new device / dtype; returns a new SparseVector."""
        return SparseVector(
            self.layout,
            {name: val.to(*args, **kwargs) for name, val in self.values.items()},
        )

    def __getitem__(self, name: str) -> TensorWrapper:
        return self.values[name]

    def __contains__(self, name: object) -> bool:
        return name in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def items(self) -> ItemsView[str, TensorWrapper]:
        return self.values.items()

    def keys(self):
        return self.values.keys()


@dataclass(frozen=True)
class SparseMatrix:
    """Per-(row_var, col_var) cell map; the typed dual of :class:`AssembledMatrix`.

    ``cells[row_var][col_var]`` is the assembled per-cell
    :class:`~neml2.types.Tensor` block that
    :attr:`AssembledMatrix.tensors[i][j]` holds -- K has already been
    folded into trailing base via
    :func:`~neml2.es._helpers._tangent_block_to_trailing_k`. Construction
    validates that the outer keys cover every row variable in
    ``row_layout``; missing inner ``(row_var, col_var)`` entries are
    allowed and become zero blocks at assembly time (per-block sparsity
    is normal in chain-rule derivatives).
    """

    row_layout: AxisLayout
    col_layout: AxisLayout
    cells: Mapping[str, Mapping[str, Tensor]]

    def __post_init__(self) -> None:
        missing = sorted(set(self.row_layout.vars()) - set(self.cells.keys()))
        if missing:
            raise KeyError(
                f"SparseMatrix: missing row entries for variables {missing}; "
                f"row_layout declares {list(self.row_layout.vars())}, "
                f"got {sorted(self.cells.keys())}."
            )

    def assemble(self) -> AssembledMatrix:
        """Walk row x col groups and pack into an :class:`AssembledMatrix`.

        Delegates to :meth:`AssembledMatrix.select_blocks`, which handles
        per-block sparsity (missing entries -> zero blocks).
        """
        # select_blocks wants a plain ``dict[str, dict[str, Tensor]]``;
        # unwrap our nested Mapping so it doesn't matter whether the
        # caller used a regular dict or some other Mapping type.
        plain: dict[str, dict[str, Tensor]] = {
            row: dict(inner) for row, inner in self.cells.items()
        }
        return AssembledMatrix.select_blocks(self.row_layout, self.col_layout, plain)

    def to(self, *args, **kwargs) -> SparseMatrix:
        """Move every cell to a new device / dtype; returns a new SparseMatrix."""
        return SparseMatrix(
            self.row_layout,
            self.col_layout,
            {
                row: {col: cell.to(*args, **kwargs) for col, cell in inner.items()}
                for row, inner in self.cells.items()
            },
        )

    def __getitem__(self, key: tuple[str, str]) -> Tensor:
        row, col = key
        return self.cells[row][col]

    def __contains__(self, key: object) -> bool:
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        row, col = key
        return row in self.cells and col in self.cells[row]


__all__ = ["SparseVector", "SparseMatrix"]
