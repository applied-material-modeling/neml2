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

r"""``SquareMatrix`` user-tensor: an ``n x n`` matrix built from structured coefficients.

A ``[Tensors]`` block type that assembles an ``n x n`` matrix and returns it as a
:class:`~neml2.types.Scalar` carried on two sub-batch axes ``(n, n)`` -- the
representation a slip-system **interaction matrix** uses (e.g. the
``interaction_matrix`` parameter of :class:`DislocationInteractionStrengthMap`). A
latent-hardening matrix can be specified **either** fully (``dense``) **or** from a
handful of physical coefficients mapped onto the ``n x n`` grid by a slip-family
partition (``block`` / ``diagonal_blocks``) -- the Franciosi picture (self / coplanar /
collinear / glissile / Hirth / Lomer).

Supported ``fill`` values and how ``data`` / ``blocks`` map onto the matrix (the fill
pattern is ``fill`` rather than ``type`` because neml2 reserves ``type`` to select the
object):

- ``identity`` -- the ``n x n`` identity (``data`` / ``blocks`` ignored).
- ``zero`` -- all zeros.
- ``diagonal`` -- ``data`` has length ``n``; ``data[i]`` goes on entry ``(i, i)``.
- ``diagonal_blocks`` -- ``blocks`` and ``data`` have equal length; each block ``k``
  fills ``blocks[k]`` consecutive diagonal entries with the single scalar ``data[k]``
  (a block-constant diagonal). ``sum(blocks) == n``.
- ``block`` -- ``data`` has length ``len(blocks)**2``; ``data`` (row-major over the
  block grid) sets one coefficient per slip-family pair: ``data[bi*nblocks + bj]`` fills
  the whole submatrix of rows in family ``bi`` x columns in family ``bj``. ``sum(blocks)
  == n``. This is the usual way to turn ~6 Franciosi coefficients into an ``n x n``
  matrix (e.g. ``m = 24``, ``blocks = '12 12'`` for a two-family lattice).
- ``dense`` -- ``data`` has length ``n**2`` (row-major, ``M[i, j] = data[i*n + j]``);
  the full matrix is given explicitly.

Geometry-agnostic: it never reads the lattice. The ``blocks`` partition and the
slip-system ordering (which system is index ``i``) are the user's responsibility and
must match the crystal geometry the matrix is paired with.

Ordering: ``dense`` and ``block`` read ``data`` row-major (``M[i, j] = data[i*n + j]``).
For a symmetric interaction matrix (the common case) the ordering is immaterial.
"""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..factory import register_neml2_object
from ..schema import HitSchema, option
from ..types import Scalar

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile


@register_neml2_object("SquareMatrix")
class SquareMatrix:
    """Assemble an ``n x n`` matrix (as a sub-batched :class:`~neml2.types.Scalar`)
    from structured coefficients."""

    SECTION: ClassVar[str] = "Tensors"

    hit = HitSchema(
        option(
            "m",
            int,
            "Matrix dimension n (e.g. the number of slip systems). Optional when 'blocks' "
            "is given -- then n is their sum.",
            default=0,
        ),
        option(
            "fill",
            str,
            "Fill pattern: 'identity', 'zero', 'diagonal', 'diagonal_blocks', 'block', or 'dense'.",
            default="dense",
        ),
        option(
            "data",
            list,
            "Coefficients; required length depends on 'type' (n for diagonal, "
            "len(blocks) for diagonal_blocks, len(blocks)**2 for block, n**2 for dense).",
            default=[],
        ),
        option(
            "blocks",
            list,
            "Block sizes (slip-family partition) summing to n; used by 'diagonal_blocks' "
            "and 'block'.",
            default=[],
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Any:
        del factory  # geometry-agnostic: nothing is resolved against the input file
        kind = node.param_optional_str("fill", "dense")
        data = list(node.param_list_float("data")) if node.find("data") is not None else []
        blocks = (
            [int(b) for b in node.param_list_int("blocks")]
            if node.find("blocks") is not None
            else []
        )
        # 'm' is optional when 'blocks' is given -- the size is then their sum.
        n = int(node.param_optional_int("m", 0))
        if n <= 0:
            if not blocks:
                raise ValueError(
                    "SquareMatrix: 'm' (matrix dimension) is required unless 'blocks' is given."
                )
            n = sum(blocks)
        mat = cls._build(n, kind, data, blocks)
        return Scalar(mat).sub_batch.retag(2)

    @staticmethod
    def _build(n: int, kind: str, data: list[float], blocks: list[int]) -> torch.Tensor:
        """Return the ``(n, n)`` float64 matrix for the requested fill pattern."""
        if n <= 0:
            raise ValueError(f"SquareMatrix: 'm' must be positive, got {n}.")
        dtype = torch.float64

        if kind == "identity":
            return torch.eye(n, dtype=dtype)
        if kind == "zero":
            return torch.zeros(n, n, dtype=dtype)
        if kind == "diagonal":
            if len(data) != n:
                raise ValueError(
                    f"SquareMatrix type 'diagonal': 'data' must have length m={n}, got {len(data)}."
                )
            return torch.diag(torch.tensor(data, dtype=dtype))
        if kind == "diagonal_blocks":
            cls_ = SquareMatrix
            cls_._check_blocks(blocks, n)
            if len(data) != len(blocks):
                raise ValueError(
                    f"SquareMatrix type 'diagonal_blocks': 'data' and 'blocks' must have "
                    f"equal length, got {len(data)} and {len(blocks)}."
                )
            diag = torch.cat(
                [torch.full((b,), float(d), dtype=dtype) for d, b in zip(data, blocks, strict=True)]
            )
            return torch.diag(diag)
        if kind == "block":
            cls_ = SquareMatrix
            cls_._check_blocks(blocks, n)
            nb = len(blocks)
            if len(data) != nb * nb:
                raise ValueError(
                    f"SquareMatrix type 'block': 'data' must have length "
                    f"len(blocks)**2={nb * nb}, got {len(data)}."
                )
            offsets = [0, *accumulate(blocks)]
            mat = torch.empty(n, n, dtype=dtype)
            for bi in range(nb):
                for bj in range(nb):
                    mat[offsets[bi] : offsets[bi + 1], offsets[bj] : offsets[bj + 1]] = float(
                        data[bi * nb + bj]
                    )
            return mat
        if kind == "dense":
            if len(data) != n * n:
                raise ValueError(
                    f"SquareMatrix type 'dense': 'data' must have length m**2={n * n}, "
                    f"got {len(data)}."
                )
            return torch.tensor(data, dtype=dtype).reshape(n, n)
        raise ValueError(
            f"SquareMatrix: invalid type {kind!r}; expected one of 'identity', 'zero', "
            "'diagonal', 'diagonal_blocks', 'block', 'dense'."
        )

    @staticmethod
    def _check_blocks(blocks: list[int], n: int) -> None:
        if not blocks:
            raise ValueError("SquareMatrix: 'blocks' is required for this type.")
        if any(b <= 0 for b in blocks):
            raise ValueError(f"SquareMatrix: 'blocks' entries must be positive, got {blocks}.")
        if sum(blocks) != n:
            raise ValueError(
                f"SquareMatrix: 'blocks' must sum to m={n}, got {blocks} (sum {sum(blocks)})."
            )


__all__ = ["SquareMatrix"]
