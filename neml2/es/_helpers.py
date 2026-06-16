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

"""Raw-tensor helpers shared across the equation-system layer."""

from __future__ import annotations

from math import prod

import torch

from neml2.types import Tensor, TensorWrapper
from neml2.types._base import KStateFlag, SubBatchStateFlag


def _storage_size(type_cls: type[TensorWrapper]) -> int:
    return prod(type_cls.BASE_SHAPE) if type_cls.BASE_SHAPE else 1


def _batch_shape(
    tensor: torch.Tensor | TensorWrapper,
    type_cls: type[TensorWrapper],
) -> torch.Size:
    """Strip the type's base-shape suffix to recover ``(*dyn, *sub_batch)``."""
    shape = tensor.shape
    if type_cls.BASE_NDIM == 0:
        return shape
    return shape[: -type_cls.BASE_NDIM]


def _flatten_base(
    tensor: torch.Tensor | TensorWrapper,
    type_cls: type[TensorWrapper],
) -> Tensor:
    """Collapse all base axes into one trailing axis."""
    if isinstance(tensor, TensorWrapper):
        return Tensor.from_typed(tensor).flatten_base()
    if type_cls.BASE_NDIM == 0:
        raw = tensor.unsqueeze(-1)
        return Tensor(raw, batch_ndim=raw.ndim - 1, sub_batch_ndim=0)
    raw = tensor.reshape(*tensor.shape[: -type_cls.BASE_NDIM], _storage_size(type_cls))
    return Tensor(raw, batch_ndim=raw.ndim - 1, sub_batch_ndim=0)


def _expanded_identity_seed(primal: TensorWrapper) -> TensorWrapper:
    """Canonical no-classification seed for the v2-parity chain rule.

    For each sub_batch axis of the primal, emit a broadcast K-paired axis
    (size 1, paired with that sub axis) -- the "diagonal" of the per-site
    Jacobian, stored compactly. For the base region, emit a single full K
    axis of size ``prod(base_shape)`` enumerating the base-direction
    components. No classification, no labels, no compact/perturb dial.

    Layout::

        data.shape = (1*N_sub, K_base, *dyn_shape, *sub_data_shape, *BASE_SHAPE)
                     ^^^^^^^^^^^^^^^^
                     k_ndim axes total

    where ``sub_data_shape[i] = 1`` for every sub axis (paired-broadcast
    storage). ``k_state[i] = "broadcast"`` for paired-sub K axes and
    ``"full"`` for the single K_base axis. ``k_pairing[i] = i`` for paired-sub
    K axes and ``None`` for K_base.

    For a primal with no sub_batch axes, the seed is just K_base. For a
    primal with no base axes (Scalar with base_size=()), K_base is omitted
    so the seed is just the paired K axes.
    """
    type_cls = type(primal)
    base_size = _storage_size(type_cls)
    has_base = type_cls.BASE_NDIM > 0 or base_size > 1
    n_sub = primal.sub_batch_ndim
    sub_shape = tuple(int(s) for s in primal.sub_batch_shape)
    # NB: do NOT cast dyn_shape entries to int(); during AOTI / torch.export
    # tracing the dyn batch dim is a ``SymInt`` and ``int(s)`` collapses it
    # to its example value (e.g. 2), triggering the "marked batch as dynamic
    # but specialized to a constant" error. The downstream torch.zeros /
    # torch.ones / eye.expand happily accept SymInts.
    dyn_shape = tuple(primal.dynamic_batch_shape)
    dtype = primal.dtype
    device = primal.device
    # Special case: Scalar with no sub_batch (e.g. equivalent_plastic_strain
    # in isoharden). k_paired_count=0 and would-be k_base_count=0 → k_ndim=0,
    # which makes the seed indistinguishable from a primal in the chain rule.
    # Force a single K axis of size 1 so the chain rule can track this seed.
    force_k_base = not has_base and n_sub == 0

    # K_paired axes (one per sub axis, broadcast size 1).
    k_paired_count = n_sub
    # K_base axis (size = base_size if base region is non-trivial OR
    # forced for the Scalar-no-sub seed case).
    k_base_count = 1 if (has_base or force_k_base) else 0
    k_ndim = k_paired_count + k_base_count

    # Build storage shape:
    #   leading K_paired axes (each size 1)
    #   K_base axis (size base_size or omitted)
    #   dyn axes (full size)
    #   sub axes (storage size 1 for paired-broadcast)
    #   base axes (BASE_SHAPE)
    # K_paired axes store size 1 (broadcast). Sub axes are at their full
    # extent in storage; the broadcast is on the K side (the size-1 K_paired
    # axis pairs with the full sub axis to form an eye diagonal).
    k_paired_storage = (1,) * k_paired_count
    # K_base storage axis: size base_size for real-base types; size 1 for the
    # force_k_base case (Scalar with no sub_batch, so base_size=1). Always
    # include the axis when k_base_count > 0 so storage and metadata agree.
    k_base_storage = (base_size,) if (has_base or force_k_base) else ()
    sub_data_shape = sub_shape
    full_shape = (
        *k_paired_storage,
        *k_base_storage,
        *dyn_shape,
        *sub_data_shape,
        *type_cls.BASE_SHAPE,
    )

    if has_base:
        # Build K_base eye on the base axis as the identity over base components.
        # When BASE_NDIM > 1 (e.g. R2 = (3,3)), unflatten K_base into the base
        # shape via reshape.
        eye = torch.eye(base_size, dtype=dtype, device=device)
        # Reshape eye to (base_size, *BASE_SHAPE) where the second axis-group
        # tiles base directions.
        eye = eye.reshape((base_size, *type_cls.BASE_SHAPE))
        # Insert K_paired leading singleton axes + dyn singletons + sub singletons
        # to align with full_shape.
        # Currently eye has shape (base_size, *BASE_SHAPE). We need to insert
        # k_paired_count leading 1s + dyn_ndim 1s + n_sub 1s after K_base
        # before BASE_SHAPE.
        reshape_target = (
            *((1,) * k_paired_count),
            base_size,
            *((1,) * len(dyn_shape)),
            *((1,) * n_sub),
            *type_cls.BASE_SHAPE,
        )
        eye = eye.reshape(reshape_target)
        # Sub axes stored size 1 in eye (since K_base axis enumerates base
        # directions). Expand to sub_shape (full sub extent) so storage matches
        # the paired-broadcast convention (size-1 K_paired against size-N sub).
        data = eye.expand(full_shape).contiguous()
    else:
        # No real base region. Either (a) Scalar primal with no sub
        # (force_k_base path → full_shape has a leading K_base=1 axis) or
        # (b) wrapper with only sub_batch → full_shape is purely
        # K_paired (size-1 each). In both cases the seed is uniformly 1
        # along the K axes (a degenerate-but-consistent identity).
        data = torch.ones(full_shape, dtype=dtype, device=device)

    # Build K state and pairing.
    state_paired: tuple[KStateFlag, ...] = ("broadcast",) * k_paired_count
    state_base: tuple[KStateFlag, ...] = ("full",) * k_base_count
    k_state: tuple[KStateFlag, ...] = state_paired + state_base
    pairing_paired: tuple[int | None, ...] = tuple(range(k_paired_count))
    pairing_base: tuple[int | None, ...] = (None,) * k_base_count
    k_pairing = pairing_paired + pairing_base

    # Sub_batch state: paired-broadcast since K_paired matches storage.
    sb_state: tuple[SubBatchStateFlag, ...]
    sb_meta: tuple[int, ...]
    if n_sub > 0:
        sb_state = ("broadcast",) * n_sub
        sb_meta = sub_shape
    else:
        sb_state = ()
        sb_meta = ()

    return type_cls(
        data,
        sub_batch_ndim=n_sub,
        sub_batch_state=sb_state,
        sub_batch_meta=sb_meta,
        k_ndim=k_ndim,
        k_state=k_state,
        k_pairing=k_pairing,
    )


def _tangent_block_to_trailing_k(block: TensorWrapper) -> Tensor:
    """Convert a leading-K typed tangent block to trailing-K ``Tensor``.

    The wrapper has K as the leftmost batch dim(s). Assembly consumes
    trailing-K blocks; we move K to the end, flatten base into a single
    row dim, and return a Tensor with the region split set from the
    wrapper's metadata.

    K materialisation: assembly needs the FULL ``K = prod(sub_extents) *
    base_size`` (one direction per (sub_site, base_component) pair); the
    chain-rule tangent carries compact-K (paired-broadcast K_paired axes
    of storage size 1). We :func:`~neml2.types.functions.fullify` the
    tangent here so paired-broadcast K_paired axes expand to per-site
    enumeration before the trailing-K move.
    """
    if not isinstance(block, TensorWrapper):
        raise TypeError(
            f"_tangent_block_to_trailing_k expects a TensorWrapper, got {type(block).__name__}"
        )
    # Fullify: expand every K-paired broadcast axis to its enumerated form.
    from neml2.types.functions import fullify  # noqa: PLC0415

    block = fullify(block)
    # Materialise any still-broadcast sub axes (typically none after fullify
    # demoted the paired-broadcast sub axes, but defensively materialise the
    # rest so trailing-K assembly sees fully-expanded sub_batch storage).
    if block.sub_batch_state and any(s == "broadcast" for s in block.sub_batch_state):
        block = block.materialize()
    base_ndim = type(block).BASE_NDIM
    sub_ndim = block.sub_batch_ndim
    k_ndim = block.k_ndim
    if k_ndim == 0:
        # No K -- treat the leading axis as K of size 1 for trailing form.
        dyn_ndim = block.data.ndim - sub_ndim - base_ndim
        moved = block.data.unsqueeze(-1)  # (*dyn, *sub, *base, 1)
    elif k_ndim == 1:
        dyn_ndim = block.data.ndim - 1 - sub_ndim - base_ndim
        moved = block.data.movedim(0, -1)  # (*dyn, *sub, *base, K)
    else:
        # Permute leading K axes to trailing as a contiguous group, THEN
        # reshape them into one. The two-step (permute + reshape on adjacent
        # trailing dims) is symbolically equivalent to the prior
        # ``reshape(K_flat, *rest).movedim(0, -1)`` but avoids torch.export's
        # collapse-view stride check on a leading-K layout: the check fires
        # the Min(K_base*B, K_base^2*B) == K_base*B guard (trivially true
        # for B>0 but not symbolically provable) when the reshape sees a
        # non-trivial leading stride pattern.
        k_shape = block.data.shape[:k_ndim]
        K_flat = prod(k_shape)
        # Move K group from leading to trailing: (K_1..K_n, *rest) -> (*rest, K_1..K_n)
        perm = (*range(k_ndim, block.data.ndim), *range(k_ndim))
        permuted = block.data.permute(perm).contiguous()
        # Now trailing k_ndim axes are contiguous; collapse them to one.
        moved = permuted.reshape(*permuted.shape[:-k_ndim], K_flat)
        dyn_ndim = moved.ndim - 1 - sub_ndim - base_ndim
    if base_ndim == 0:
        raw = moved.unsqueeze(-2)
    elif base_ndim == 1:
        raw = moved
    else:
        K = moved.shape[-1]
        raw = moved.reshape(*moved.shape[: moved.ndim - 1 - base_ndim], -1, K)
    return Tensor(raw, batch_ndim=dyn_ndim, sub_batch_ndim=sub_ndim)


__all__ = [
    "_storage_size",
    "_batch_shape",
    "_flatten_base",
    "_expanded_identity_seed",
    "_tangent_block_to_trailing_k",
]
