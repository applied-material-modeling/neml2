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

"""Cached batched LU factorization shared by the LU-based linear solvers."""

from __future__ import annotations

from collections.abc import Callable

import torch


class LUCache:
    """Cache one batched LU factorization, reused across solves of the same matrix.

    Skips the repeated ``lu_factor`` when a matrix is solved against several
    right-hand sides in a row (a :class:`~neml2.solvers.SchurComplement`'s two
    primary solves, or a Thomas sweep reusing a diagonal block).

    The source tensor is held **by reference** and keyed by identity plus its
    ``_version``: holding it alive keeps the key safe (a freed tensor's ``id`` can
    be recycled, so ``id()`` alone could return a stale factorization), and the
    ``_version`` check invalidates on an in-place edit. ``factor_fn`` selects the
    routine (default :func:`torch.linalg.lu_factor`; a CUDA caller may inject a
    guarded variant).
    """

    def __init__(self, factor_fn: Callable = torch.linalg.lu_factor) -> None:
        """Start with an empty cache; ``factor_fn`` performs the factorization."""
        self._factor_fn = factor_fn
        self._src: torch.Tensor | None = None
        self._version: int | None = None
        self._lu: torch.Tensor | None = None
        self._piv: torch.Tensor | None = None

    def factor(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached ``(LU, pivots)`` of ``matrix``, refactoring on change."""
        if matrix is not self._src or matrix._version != self._version:
            self._lu, self._piv = self._factor_fn(matrix)
            self._src = matrix
            self._version = matrix._version
        assert self._lu is not None and self._piv is not None
        return self._lu, self._piv

    def invalidate(self) -> None:
        """Drop the cached factorization and release the held source tensor."""
        self._src = None
        self._version = None
        self._lu = None
        self._piv = None


__all__ = ["LUCache"]
