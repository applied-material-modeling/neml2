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

"""Shared type aliases for chain-rule sensitivity propagation.

Tangents in transit through the framework are ordinary typed wrappers
(``Scalar`` / ``SR2`` / ``R2`` / ...) carrying one or more leading K
(seed-direction) axes -- a tangent of type ``T`` *is* a ``T``, with
``k_ndim > 0`` and ``k_state`` / ``k_pairing`` recording the per-K-axis
storage layout (see :class:`neml2.types._base.TensorWrapper`).

Per the v2-parity refactor, the declarative ``EdgeInfo`` / ``list_deriv``
machinery has been removed. The chain rule no longer dispatches on
labels -- it relies on positional ``k_pairing`` metadata to know when a
K axis is paired with a sub_batch axis (cheap broadcast diagonal) vs
when it has been exposed to per-site enumeration.

When a leaf's action would contract a paired sub_batch axis (e.g. a
``sum`` / ``mean`` / ``inner`` that mixes per-site data), it must call
``fullify`` from :mod:`neml2.types.functions` first. The exposing
reductions (``sum_sub_batch``, ``sum`` / ``mean`` over a K-paired axis)
handle the common case automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from torch import Tensor

from neml2.types._base import TensorWrapper

# Tangents flow as ordinary typed wrappers with the K (seed-direction) axis or
# axes as the leftmost batch dims of ``data``.
TangentAction: TypeAlias = TensorWrapper
ChainRuleDict: TypeAlias = dict[str, dict[str, TensorWrapper]]
ChainRuleAction: TypeAlias = Callable[..., TensorWrapper]

SecondOrderTangentAction: TypeAlias = TensorWrapper
SecondOrderChainRuleDict: TypeAlias = dict[str, dict[str, dict[str, TensorWrapper]]]
SecondOrderChainRuleAction: TypeAlias = Callable[..., TensorWrapper]


def equalize_tangent_K(contributions: list[TensorWrapper]) -> list[TensorWrapper]:
    """Tile compact-K contributions to match the max-K contribution.

    Apply-chain-rule accumulates contributions for the same seed leaf across
    multiple input edges. Different edges can produce different K storage
    widths -- one edge may carry a tangent whose K axes have been exposed
    (full) by a reducing op, while a parallel edge with no sub_batch
    interaction carries an un-exposed (broadcast) tangent with the same
    K_ndim but smaller storage.

    Summing tensors of different storage K widths along the leading K axes is
    shape-incompatible. The fix is to tile the lower-K contributions to the
    common max along each K axis. A compact contribution represents a response
    that does not depend on the per-site index (definitionally -- it came from
    a chain with no sub_batch interaction), so tiling preserves semantics.

    Returns a list of contributions all with the same leading K storage shape.
    No-op when every contribution already matches the max (the common case
    once ``align_k`` + ``combine_k_state`` have already aligned).
    """
    if not contributions:
        return contributions
    k_sizes = [
        tuple(int(s) for s in c.data.shape[: c.k_ndim]) if c.k_ndim > 0 else ()
        for c in contributions
    ]
    if all(ks == k_sizes[0] for ks in k_sizes):
        return contributions
    max_k = list(k_sizes[0])
    for ks in k_sizes[1:]:
        for i, s in enumerate(ks):
            if i < len(max_k):
                if s > max_k[i]:
                    max_k[i] = s
            else:
                max_k.append(s)
    # If any contribution carries a paired-broadcast K axis that needs
    # to be expanded to a parallel contribution's full K, eye-expand it
    # via :func:`~neml2.types.functions.fullify` BEFORE the per-axis
    # ``expand`` below. Plain expand on a paired-broadcast axis duplicates
    # the diagonal storage across all K positions, which is the WRONG
    # semantic -- the paired-broadcast convention stores the eye
    # diagonal, and the correct expanded form is eye(K_size) on the
    # (K, sub_pair) pair, not a constant row. Fullify is exactly that
    # eye expansion + state demotion (broadcast → full, pairing → None),
    # so callers downstream see a proper full-storage K axis that the
    # subsequent ``expand`` (for OTHER axes) can treat uniformly.
    from ..types.functions import fullify  # noqa: PLC0415

    needs_fullify: list[bool] = []
    for c, cur in zip(contributions, k_sizes, strict=True):
        if c.k_ndim == 0:
            needs_fullify.append(False)
            continue
        nf = False
        for i in range(c.k_ndim):
            if (
                cur[i] != max_k[i]
                and cur[i] == 1
                and c.k_state[i] == "broadcast"
                and c.k_pairing[i] is not None
            ):
                nf = True
                break
        needs_fullify.append(nf)
    contributions = [
        fullify(c) if nf else c for c, nf in zip(contributions, needs_fullify, strict=True)
    ]

    out: list[TensorWrapper] = []
    for c in contributions:
        if c.k_ndim == 0:
            out.append(c)
            continue
        cur = list(c.data.shape[: c.k_ndim])
        if all(cur[i] == max_k[i] for i in range(c.k_ndim)):
            out.append(c)
            continue
        target = list(c.data.shape)
        for i in range(c.k_ndim):
            if cur[i] != max_k[i]:
                if cur[i] != 1:
                    raise ValueError(
                        f"equalize_tangent_K: K axis {i} has incompatible storage "
                        f"size {cur[i]} (target {max_k[i]}); align upstream."
                    )
                target[i] = max_k[i]
        new_data = c.data.expand(target).contiguous()
        out.append(c._rewrap(new_data, sub_batch_ndim=c.sub_batch_ndim))
    return out


def matvec(M: Tensor, v: Tensor) -> Tensor:
    """Fused matrix-vector product without ``einsum``.

    ``M @ v`` where the trailing dims are ``(..., n_row, n_col)`` and
    ``(..., n_col)`` respectively. Implemented as
    ``(M @ v.unsqueeze(-1)).squeeze(-1)`` so Inductor can fuse the trailing
    pointwise ops around the matmul.
    """
    return (M @ v.unsqueeze(-1)).squeeze(-1)


__all__ = [
    "TangentAction",
    "ChainRuleDict",
    "ChainRuleAction",
    "SecondOrderTangentAction",
    "SecondOrderChainRuleDict",
    "SecondOrderChainRuleAction",
    "matvec",
    "equalize_tangent_K",
]
