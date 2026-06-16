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

"""Common base for the typed tensor wrapper classes.

`TensorWrapper` carries the shared shape decomposition + region views +
``_rewrap`` machinery. Each concrete wrapper is a
``@dataclass(frozen=True, eq=False)`` with these fields:

- ``data: torch.Tensor`` -- the underlying storage.
- ``sub_batch_ndim: int = 0`` -- count of trailing batch axes that act
  as a structured "sub-batch" region (the Python-native analogue of the
  C++ ``intmd_dim``).
- ``sub_batch_state: tuple[SubBatchStateFlag, ...] = ()`` -- per-axis
  ``"full"`` / ``"broadcast"`` storage mode (see :mod:`neml2.chain_rule`).
- ``sub_batch_meta: tuple[int, ...] = ()`` -- per-axis logical extent
  consulted only when the axis is in broadcast mode.
- ``k_ndim: int = 0`` -- count of leading K (seed-direction) axes
  carried by a chain-rule tangent. Primal wrappers have ``k_ndim == 0``.
- ``k_state: tuple[KStateFlag, ...] = ()`` -- per-K-axis ``"full"`` /
  ``"broadcast"`` storage mode. Length must equal ``k_ndim``.
- ``k_pairing: tuple[int | None, ...] = ()`` -- per-K-axis pairing.
  ``k_pairing[i] = j`` means K axis i is paired with sub_batch axis j;
  the seed for that pair is the eye-shape diagonal on (K_i, sub_j).
  ``None`` means the K axis is a base-direction enumerator with no
  sub pairing. Length must equal ``k_ndim``.

Shape decomposition
-------------------

A wrapper's tensor shape splits into FOUR regions::

    data.shape == (*K, *dynamic_batch_shape, *sub_batch_shape, *base_shape)
                  └─ K ─┘└── leading ──┘└── middle (static) ──┘└─ BASE_NDIM ─┘

- ``base_shape`` is fixed by the wrapper type (e.g. ``(6,)`` for SR2).
- ``sub_batch_shape`` is the next ``sub_batch_ndim`` trailing batch axes.
  These broadcast like batch dims in forward ops but encode per-site
  structure (cells, slip systems, grains) that the chain-rule machinery
  treats specially.
- ``dynamic_batch_shape`` is everything between K and sub_batch (the
  variable dim count that ``torch.export`` traces as dynamic).
- ``K`` is the ``k_ndim`` leading axes carried by a chain-rule tangent.
  Primal values have ``k_ndim == 0`` and so this region is absent.

The combined ``batch_shape`` (``dynamic + sub_batch``) matches the C++
``batch_dim() = dynamic_dim + intmd_dim`` invariant. The K region sits
ENTIRELY LEFT OF ``batch_shape`` -- region-view computations use
``data.ndim - BASE_NDIM - sub_batch_ndim`` for the right edge of dyn
and subtract ``k_ndim`` from the left to find dyn's left edge.

Region-view API
---------------

Shape-manipulating ops are exposed through region-view properties --
``t.batch``, ``t.dynamic_batch``, ``t.sub_batch``, ``t.base`` -- and a
``t.k`` view over the K region. Each view is computed on access (no
storage, no pytree registration) and its methods return a fresh wrapper,
so chaining works.

The K region is read-mostly today; tangent-side reshapes are driven by
the chain-rule primitives in :mod:`neml2.types.functions` (``fullify``,
``sum_sub_batch``, exposing reductions), not by direct K-view mutation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast, overload

import torch

if TYPE_CHECKING:
    from typing_extensions import Self

_WT = TypeVar("_WT", bound="TensorWrapper")

#: Per-sub-batch-axis storage mode (see :mod:`neml2.chain_rule`).
#:
#: - ``"full"`` -- ``data.shape`` at this axis equals the logical sub-batch
#:   extent. The legacy storage mode for every primal value.
#: - ``"broadcast"`` -- ``data.shape`` at this axis is 1, but the logical
#:   extent (carried in :attr:`TensorWrapper.sub_batch_meta`) is whatever
#:   downstream primals will broadcast against.
SubBatchStateFlag: TypeAlias = Literal["full", "broadcast"]

#: Per-K-axis storage mode (mirrors :data:`SubBatchStateFlag` semantics on the
#: K region).
#:
#: - ``"full"`` -- ``data.shape`` at this K axis equals the K-direction count
#:   the axis enumerates (e.g. ``base_size`` for a K_base axis, or the paired
#:   sub-batch extent for a K_paired axis that's been exposed by a reducing op).
#: - ``"broadcast"`` -- ``data.shape`` at this K axis is 1. For a K_paired
#:   axis (``k_pairing[i] is not None``) the broadcast is the "diagonal" of the
#:   Jacobian on that (K, sub) pair: per-site value at (K_i=k, sub_pair=g) is
#:   ``data[k=0, sub_pair=g] * delta(k==g)``, materialised by :func:`fullify`.
KStateFlag: TypeAlias = Literal["full", "broadcast"]


class TensorWrapper:
    """Shared shape decomposition + region views + ``_rewrap`` machinery
    for the typed tensor wrappers.

    Subclasses (``@dataclass(frozen=True, eq=False)``) declare the seven
    instance fields listed in the module docstring; the trailing
    ``BASE_NDIM`` dimensions of ``data`` have shape ``BASE_SHAPE`` (in
    Mandel packing where applicable).
    """

    data: torch.Tensor
    sub_batch_ndim: int
    sub_batch_state: tuple[SubBatchStateFlag, ...]
    sub_batch_meta: tuple[int, ...]
    k_ndim: int
    k_state: tuple[KStateFlag, ...]
    k_pairing: tuple[int | None, ...]
    BASE_NDIM: ClassVar[int]
    BASE_SHAPE: ClassVar[tuple[int, ...]]

    def __init__(
        self,
        data: torch.Tensor | TensorWrapper,
        sub_batch_ndim: int = 0,
        sub_batch_state: tuple[SubBatchStateFlag, ...] = (),
        sub_batch_meta: tuple[int, ...] = (),
        k_ndim: int = 0,
        k_state: tuple[KStateFlag, ...] = (),
        k_pairing: tuple[int | None, ...] = (),
    ) -> None:
        # Subclasses are ``@dataclass(frozen=True, eq=False)``; the dataclass
        # decorator generates the real ``__init__`` on each one. This stub
        # exists for type-checkers viewing callers like ``t_cls(tensor)`` with
        # ``t_cls: type[TensorWrapper]``. The ``data`` arg is typed as
        # ``torch.Tensor | TensorWrapper`` because ``__post_init__`` unwraps a
        # nested wrapper (used by ComposedModel's re-wrap-before-dispatch).
        raise NotImplementedError("TensorWrapper is abstract; instantiate a concrete subclass.")

    # Arithmetic operator stubs (concrete impls live on PrimitiveTensor).

    def __neg__(self) -> TensorWrapper:
        raise NotImplementedError

    def __add__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __radd__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __sub__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __rsub__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __mul__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __rmul__(self, other) -> TensorWrapper:
        raise NotImplementedError

    def __post_init__(self) -> None:
        """Defend against double-wrapping AND enforce K-region invariants.

        Double-wrap unwrap
        ------------------

        ``ComposedModel`` re-wraps each child's inputs via ``type_cls(state[n])``
        before dispatch. When the child is itself a ``ComposedModel`` (or
        ``ImplicitUpdate``) whose own dispatch then re-wraps, the same value
        gets wrapped twice -- yielding e.g. ``SR2(SR2(tensor))`` where
        ``self.data`` is an ``SR2`` instead of a ``torch.Tensor``. This hook
        unwraps a same-type wrapper input to its inner ``data`` so the
        double-wrap is a no-op.

        K-region invariants (Appendix A of v2-parity plan)
        --------------------------------------------------

        - ``len(k_state) == k_ndim``
        - ``len(k_pairing) == k_ndim``
        - each ``k_state[i] in {"full", "broadcast"}``
        - each ``k_pairing[i]`` is ``None`` or in ``[0, sub_batch_ndim)``
        - a paired-broadcast K axis (``k_state[i] == "broadcast"`` AND
          ``k_pairing[i] is not None``) MUST have ``data.shape`` size 1
          at the K axis position (positional invariant for the
          eye-diagonal storage convention).
        """
        if isinstance(self.data, TensorWrapper):
            inner = self.data
            if not isinstance(inner, type(self)):
                raise TypeError(
                    f"Cannot wrap a {type(inner).__name__} as a {type(self).__name__}; "
                    "wrapper types must match. Pass `inner.data` instead of the wrapper."
                )
            object.__setattr__(self, "data", inner.data)

        # K-region invariants.
        k_ndim = self.k_ndim
        if len(self.k_state) != k_ndim:
            raise ValueError(f"k_state length ({len(self.k_state)}) must equal k_ndim ({k_ndim})")
        if len(self.k_pairing) != k_ndim:
            raise ValueError(
                f"k_pairing length ({len(self.k_pairing)}) must equal k_ndim ({k_ndim})"
            )
        for i, s in enumerate(self.k_state):
            if s not in ("full", "broadcast"):
                raise ValueError(f"k_state[{i}] must be 'full' or 'broadcast', got {s!r}")
        for i, p in enumerate(self.k_pairing):
            if p is None:
                continue
            if not isinstance(p, int):
                raise ValueError(f"k_pairing[{i}] must be int or None, got {type(p).__name__}")
            if p < 0 or p >= self.sub_batch_ndim:
                raise ValueError(
                    f"k_pairing[{i}]={p} out of range for sub_batch_ndim={self.sub_batch_ndim}"
                )
            if self.k_state[i] == "broadcast":
                # Paired-broadcast invariant: data.shape at K axis i must be 1.
                if self.data.shape[i] != 1:
                    raise ValueError(
                        f"k_pairing[{i}]={p} with k_state[{i}]='broadcast' requires "
                        f"data.shape[{i}] == 1 (got {self.data.shape[i]}); the paired-"
                        "broadcast convention stores the eye diagonal as a size-1 axis."
                    )

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Total tensor rank (``len(shape)``) -- equals
        ``k_ndim + dynamic_batch_ndim + sub_batch_ndim + BASE_NDIM``."""
        return self.data.ndim

    # ---- shape decomposition ----

    @property
    def base_shape(self) -> torch.Size:
        if self.BASE_NDIM == 0:
            return torch.Size(())
        return self.data.shape[-self.BASE_NDIM :]

    @property
    def batch_shape(self) -> torch.Size:
        """All non-K, non-base dims, i.e. dynamic + sub-batch."""
        return self.data.shape[self.k_ndim : self.data.ndim - self.BASE_NDIM]

    @property
    def sub_batch_shape(self) -> torch.Size:
        n = self.sub_batch_ndim
        if n == 0:
            return torch.Size(())
        bsh = self.batch_shape
        raw = bsh[len(bsh) - n :]
        # Logical extent: ``data.shape`` for "full" axes, ``sub_batch_meta``
        # for "broadcast" axes. Empty state tuple = legacy all-"full".
        state = self.sub_batch_state
        if not state:
            return raw
        meta = self.sub_batch_meta
        return torch.Size(meta[i] if state[i] == "broadcast" else raw[i] for i in range(n))

    @property
    def dynamic_batch_shape(self) -> torch.Size:
        n = self.sub_batch_ndim
        bsh = self.batch_shape
        if n == 0:
            return bsh
        return bsh[: len(bsh) - n]

    @property
    def k_shape(self) -> torch.Size:
        """The K-region storage shape (raw ``data.shape[:k_ndim]``).

        ``"broadcast"`` axes here are size 1 in storage; the logical
        extent of a paired-broadcast K axis equals the paired sub axis's
        extent (recovered via :attr:`sub_batch_shape` and
        :attr:`k_pairing`).
        """
        if self.k_ndim == 0:
            return torch.Size(())
        return self.data.shape[: self.k_ndim]

    # ---- region views ----

    @property
    def batch(self: Self) -> BatchView[Self]:
        return BatchView(self)

    @property
    def dynamic_batch(self: Self) -> DynamicBatchView[Self]:
        return DynamicBatchView(self)

    @property
    def sub_batch(self: Self) -> SubBatchView[Self]:
        return SubBatchView(self)

    @property
    def base(self: Self) -> BaseView[Self]:
        return BaseView(self)

    @property
    def k(self: Self) -> KView[Self]:
        """Read-only view over the K region (leading ``k_ndim`` axes)."""
        return KView(self)

    # ---- internal: clone-with-new-tensor-and-metadata ----

    def with_sub_batch_ndim(self, sub_batch_ndim: int) -> Self:
        """Return a structurally-identical wrapper with the declared
        ``sub_batch_ndim`` overridden.

        Same storage, same K layout. Per-axis ``sub_batch_state`` /
        ``sub_batch_meta`` are reset to empty since the axis count is
        changing -- callers that need a labelled state must re-attach it.

        Use this to normalize a wrapper that crossed a layer boundary
        with the wrong sub axis count (e.g. a predictor output that lost
        its declared per-site axes) instead of unwrapping to raw data
        and re-constructing.
        """
        if sub_batch_ndim == self.sub_batch_ndim:
            return self
        return self._rewrap(self.data, sub_batch_ndim=sub_batch_ndim)

    def _rewrap(
        self,
        data: torch.Tensor,
        *,
        sub_batch_ndim: int,
        sub_batch_state: tuple[SubBatchStateFlag, ...] | None = None,
        sub_batch_meta: tuple[int, ...] | None = None,
        k_ndim: int | None = None,
        k_state: tuple[KStateFlag, ...] | None = None,
        k_pairing: tuple[int | None, ...] | None = None,
    ) -> Self:
        """Build a structurally-identical wrapper around new ``data``.

        Per-field inheritance rule:

        - ``sub_batch_state`` / ``sub_batch_meta``: inherit from ``self``
          iff ``sub_batch_ndim`` is unchanged; otherwise default to empty.
        - ``k_ndim``: when ``None``, inherit from ``self``. When passed
          explicitly, ``k_state`` and ``k_pairing`` must also be passed
          (no implicit reset to empty across a K-rank change -- forces
          callers to be explicit about the new K shape).
        - ``k_state`` / ``k_pairing``: inherit from ``self`` iff the
          effective ``k_ndim`` matches ``self.k_ndim`` AND the caller
          didn't pass either explicitly.

        Forcing per-field independent inheritance prevents an op that
        updates state from accidentally dropping K metadata.
        """
        same_sb = sub_batch_ndim == self.sub_batch_ndim
        if sub_batch_state is None:
            sub_batch_state = self.sub_batch_state if same_sb else ()
        if sub_batch_meta is None:
            sub_batch_meta = self.sub_batch_meta if same_sb else ()

        new_k_ndim = self.k_ndim if k_ndim is None else k_ndim
        if k_ndim is not None and (k_state is None or k_pairing is None):
            # Explicit k_ndim change requires explicit state + pairing.
            if k_state is None:
                k_state = ()
            if k_pairing is None:
                k_pairing = ()
        same_k = new_k_ndim == self.k_ndim
        if k_state is None:
            k_state = self.k_state if same_k else ()
        if k_pairing is None:
            k_pairing = self.k_pairing if same_k else ()

        new: Self = object.__new__(type(self))
        object.__setattr__(new, "data", data)
        object.__setattr__(new, "sub_batch_ndim", sub_batch_ndim)
        object.__setattr__(new, "sub_batch_state", sub_batch_state)
        object.__setattr__(new, "sub_batch_meta", sub_batch_meta)
        object.__setattr__(new, "k_ndim", new_k_ndim)
        object.__setattr__(new, "k_state", k_state)
        object.__setattr__(new, "k_pairing", k_pairing)
        # Run __post_init__ to enforce invariants on the fresh instance.
        new.__post_init__()
        return new

    def materialize(self: Self) -> Self:
        """Force every ``"broadcast"`` sub-batch axis into ``"full"`` storage.

        Cheap when ``sub_batch_state`` is already empty or all-``"full"``
        (no-op return). Does NOT touch the K region -- K materialisation
        is the job of :func:`fullify`.
        """
        state = self.sub_batch_state
        if not state or all(s == "full" for s in state):
            return self  # type: ignore[return-value]
        n = self.sub_batch_ndim
        sb_start = self.data.ndim - self.BASE_NDIM - n
        target = list(self.data.shape)
        meta = self.sub_batch_meta
        for i, s in enumerate(state):
            if s == "broadcast":
                target[sb_start + i] = int(meta[i])
        new_data = self.data.expand(target).contiguous()
        return self._rewrap(
            new_data,
            sub_batch_ndim=n,
            sub_batch_state=("full",) * n,
            sub_batch_meta=tuple(target[sb_start : sb_start + n]),
        )

    def to(self, *args, **kwargs) -> Self:
        moved = self.data.to(*args, **kwargs)
        if moved is self.data:
            return self  # type: ignore[return-value]
        return self._rewrap(moved, sub_batch_ndim=self.sub_batch_ndim)


# ---------------------------------------------------------------------------
# Region views
# ---------------------------------------------------------------------------


class _RegionView(Generic[_WT]):
    """Common machinery for the region views."""

    _REGION: ClassVar[str]

    __slots__ = ("_w",)

    def __init__(self, wrapper: _WT) -> None:
        self._w: _WT = wrapper

    def _bounds(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def shape(self) -> torch.Size:
        start, length = self._bounds()
        if length == 0:
            return torch.Size(())
        return self._w.data.shape[start : start + length]

    @property
    def ndim(self) -> int:
        return self._bounds()[1]

    def _resolve_dim(self, dim: int) -> int:
        n = self.ndim
        if dim < 0:
            dim += n
        if dim < 0 or dim >= n:
            raise IndexError(f"{self._REGION} dim {dim} out of range for {self._REGION}_ndim={n}")
        start, _ = self._bounds()
        return start + dim

    def _resolve_insert_dim(self, dim: int) -> int:
        n = self.ndim
        if dim < 0:
            dim += n + 1
        if dim < 0 or dim > n:
            raise IndexError(
                f"{self._REGION} insert dim {dim} out of range for {self._REGION}_ndim={n}"
            )
        start, _ = self._bounds()
        return start + dim


class _MutableRegionView(_RegionView[_WT]):
    """Common shape-changing ops shared by ``batch`` / ``dynamic_batch`` / ``sub_batch``."""

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        raise NotImplementedError

    def unsqueeze(self, dim: int = -1, n: int = 1) -> _WT:
        if n < 0:
            raise ValueError(f"{self._REGION}.unsqueeze: n must be non-negative, got {n}")
        if n == 0:
            return self._w
        insert_at = self._resolve_insert_dim(dim)
        new_data = self._w.data
        for _ in range(n):
            new_data = new_data.unsqueeze(insert_at)
        new_sb = self._new_sub_batch_ndim(axes_added=n)
        return self._w._rewrap(new_data, sub_batch_ndim=new_sb)

    def squeeze(self, dim: int = -1) -> _WT:
        axis = self._resolve_dim(dim)
        if self._w.data.shape[axis] != 1:
            raise ValueError(
                f"{self._REGION}.squeeze: axis {dim} has size "
                f"{self._w.data.shape[axis]}, expected 1"
            )
        new_data = self._w.data.squeeze(axis)
        new_sb = self._new_sub_batch_ndim(axes_removed=1)
        return self._w._rewrap(new_data, sub_batch_ndim=new_sb)

    def expand(self, *shape: int) -> _WT:
        cur = self.shape
        target = tuple(shape)
        if len(cur) > len(target):
            raise ValueError(
                f"{self._REGION}.expand: target {target} has fewer dims than "
                f"current {tuple(cur)}; squeeze first if dropping axes is intended"
            )
        pad = len(target) - len(cur)
        start, length = self._bounds()
        new_data = self._w.data
        for _ in range(pad):
            new_data = new_data.unsqueeze(start)
        before = self._w.data.shape[:start]
        after = self._w.data.shape[start + length :]
        full_target = (*before, *target, *after)
        new_data = new_data.expand(full_target).contiguous()
        new_sb = self._new_sub_batch_ndim(axes_added=pad)
        return self._w._rewrap(new_data, sub_batch_ndim=new_sb)

    def __getitem__(self, key) -> _WT:
        """Index / slice within this region only, preserving wrapper metadata.

        ``key`` addresses the region's axes only; pre-region and
        post-region axes pass through as ``:``. Supports ``Ellipsis``
        (expands to ``slice(None)`` fills), ``None`` (inserts a new
        region axis), integer indexing (drops an axis), and slicing
        (preserves rank). The wrapper's ``sub_batch_ndim`` is adjusted
        for any added (``None``) or removed (``int``) region axes; K
        metadata propagates via :meth:`_rewrap`.

        Examples (Scalar with ``sub_batch_ndim=1``, sub axis size N)::

            V.sub_batch[:-1]    # narrow sub axis to size N-1, drops trailing entry
            V.sub_batch[1:]     # narrow sub axis to size N-1, drops leading entry
            V.sub_batch[0]      # pick the first sub-batch site (drops sub axis)
            V.sub_batch[None]   # add a new size-1 sub axis at the front
        """
        if not isinstance(key, tuple):
            key = (key,)
        start, length = self._bounds()
        # Expand any ``Ellipsis`` to ``slice(None)`` fills so the user
        # can pass ``...`` to mean "all the region's other axes". Mirror
        # the convention from the dynamic-base ``Tensor`` region views.
        if Ellipsis in key:
            n_ellipsis = sum(1 for k in key if k is Ellipsis)
            if n_ellipsis > 1:
                raise IndexError(f"{self._REGION}.__getitem__: at most one Ellipsis allowed")
            explicit_axes = sum(1 for k in key if k is not Ellipsis and k is not None)
            n_fill = length - explicit_axes
            if n_fill < 0:
                raise IndexError(
                    f"{self._REGION}.__getitem__: key has more indices "
                    f"({explicit_axes}) than region ndim ({length})"
                )
            idx = key.index(Ellipsis)
            key = key[:idx] + (slice(None),) * n_fill + key[idx + 1 :]
        # Build the full data-side key: pre-region ``:``, user's key,
        # post-region ``:``. The post-region length captures the base
        # axes that pass through untouched.
        pre = (slice(None),) * start
        post = (slice(None),) * (self._w.data.ndim - start - length)
        full_key = pre + key + post
        new_data = self._w.data[full_key]
        added = sum(1 for k in key if k is None)
        removed = sum(1 for k in key if isinstance(k, int))
        new_sb = self._new_sub_batch_ndim(axes_added=added, axes_removed=removed)
        return self._w._rewrap(new_data, sub_batch_ndim=new_sb)


class BatchView(_MutableRegionView[_WT]):
    """Read-mostly view into ``dynamic + sub_batch``."""

    _REGION = "batch"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        # batch sits between K and base.
        return (w.k_ndim, w.data.ndim - w.BASE_NDIM - w.k_ndim)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        return self._w.sub_batch_ndim

    def unsqueeze(self, dim: int = -1, n: int = 1) -> _WT:
        raise TypeError(
            "batch.unsqueeze is ambiguous; use t.dynamic_batch.unsqueeze(...) or "
            "t.sub_batch.unsqueeze(...)"
        )

    def squeeze(self, dim: int = -1) -> _WT:
        raise TypeError(
            "batch.squeeze is ambiguous; use t.dynamic_batch.squeeze(...) "
            "or t.sub_batch.squeeze(...)"
        )

    def expand(self, *shape: int) -> _WT:
        raise TypeError(
            "batch.expand is ambiguous; use t.dynamic_batch.expand(...) or t.sub_batch.expand(...)"
        )


class DynamicBatchView(_MutableRegionView[_WT]):
    """View into the dynamic batch region. Ops preserve ``sub_batch_ndim``."""

    _REGION = "dynamic_batch"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        return (w.k_ndim, w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim - w.k_ndim)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        return self._w.sub_batch_ndim


class SubBatchView(_MutableRegionView[_WT]):
    """View into the sub-batch region. Shape-changing ops adjust ``sub_batch_ndim``."""

    _REGION = "sub_batch"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        return (sb_start, w.sub_batch_ndim)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        return self._w.sub_batch_ndim + axes_added - axes_removed

    def unsqueeze(self, dim: int = -1, n: int = 1) -> _WT:
        if n < 0:
            raise ValueError(f"sub_batch.unsqueeze: n must be non-negative, got {n}")
        if n == 0:
            return self._w
        insert_at = self._resolve_insert_dim(dim)
        sub_pos = insert_at - self._bounds()[0]
        new_data = self._w.data
        for _ in range(n):
            new_data = new_data.unsqueeze(insert_at)
        new_sb = self._new_sub_batch_ndim(axes_added=n)
        old_state = self._w.sub_batch_state
        if not old_state:
            return self._w._rewrap(new_data, sub_batch_ndim=new_sb)
        # New axes are size 1 in data; mark them ``"broadcast"`` with a
        # placeholder meta of 1. A subsequent op against a sized primal
        # picks up the actual logical extent via
        # :func:`combine_sub_batch_state`.
        head = old_state[:sub_pos]
        tail = old_state[sub_pos:]
        new_state: tuple[SubBatchStateFlag, ...] = (
            head + (cast("SubBatchStateFlag", "broadcast"),) * n + tail
        )
        old_meta = self._w.sub_batch_meta
        new_meta = old_meta[:sub_pos] + (1,) * n + old_meta[sub_pos:]
        return self._w._rewrap(
            new_data,
            sub_batch_ndim=new_sb,
            sub_batch_state=new_state,
            sub_batch_meta=new_meta,
        )

    def squeeze(self, dim: int = -1) -> _WT:
        axis = self._resolve_dim(dim)
        if self._w.data.shape[axis] != 1:
            raise ValueError(
                f"sub_batch.squeeze: axis {dim} has size {self._w.data.shape[axis]}, expected 1"
            )
        sub_pos = axis - self._bounds()[0]
        new_data = self._w.data.squeeze(axis)
        new_sb = self._new_sub_batch_ndim(axes_removed=1)
        old_state = self._w.sub_batch_state
        if not old_state:
            return self._w._rewrap(new_data, sub_batch_ndim=new_sb)
        new_state = old_state[:sub_pos] + old_state[sub_pos + 1 :]
        old_meta = self._w.sub_batch_meta
        new_meta = old_meta[:sub_pos] + old_meta[sub_pos + 1 :]
        return self._w._rewrap(
            new_data,
            sub_batch_ndim=new_sb,
            sub_batch_state=new_state,
            sub_batch_meta=new_meta,
        )

    def expand_at(self, size: int, dim: int = 0) -> _WT:
        if size <= 0:
            raise ValueError(f"sub_batch.expand_at: size must be positive, got {size}")
        insert_at = self._resolve_insert_dim(dim)
        new_data = self._w.data.unsqueeze(insert_at).expand(
            *self._w.data.shape[:insert_at], size, *self._w.data.shape[insert_at:]
        )
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim + 1)

    def diagonalize(self) -> _WT:
        if self._w.sub_batch_ndim == 0:
            raise ValueError(
                "sub_batch.diagonalize requires at least one sub-batch dim; got sub_batch_ndim=0"
            )
        if self._w.BASE_NDIM != 0:
            raise NotImplementedError(
                "sub_batch.diagonalize for BASE_NDIM > 0 wrappers is not "
                "implemented; no in-scope caller needs it."
            )
        new_data = torch.diag_embed(self._w.data)
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim + 1)

    def retag(self, ndim: int) -> _WT:
        """Return a view of the same ``data`` with ``sub_batch_ndim=ndim``."""
        if ndim < 0:
            raise ValueError(f"sub_batch.retag: ndim must be non-negative, got {ndim}")
        w = self._w
        batch_ndim = w.data.ndim - w.BASE_NDIM - w.k_ndim
        if ndim > batch_ndim:
            raise ValueError(
                f"sub_batch.retag: ndim={ndim} exceeds available batch dims "
                f"({batch_ndim}); tensor shape={tuple(w.data.shape)}, "
                f"BASE_NDIM={w.BASE_NDIM}, k_ndim={w.k_ndim}"
            )
        if ndim == w.sub_batch_ndim:
            return w
        return w._rewrap(w.data, sub_batch_ndim=ndim)

    def flatten(self) -> _WT:
        """Demote the sub-batch region to dynamic batch (``retag(0)``)."""
        return self.retag(0)


class BaseView(_RegionView[_WT]):
    """View into the base region."""

    _REGION = "base"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        return (w.data.ndim - w.BASE_NDIM, w.BASE_NDIM)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> _WT:
        if self._w.BASE_NDIM < 2:
            raise TypeError(
                f"base.transpose requires BASE_NDIM >= 2; "
                f"{type(self._w).__name__} has BASE_NDIM={self._w.BASE_NDIM}"
            )
        ax0 = self._resolve_dim(dim0)
        ax1 = self._resolve_dim(dim1)
        new_data = self._w.data.transpose(ax0, ax1)
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim)


class KView(_RegionView[_WT]):
    """Read-only view over the K (seed-direction) region.

    Exposes ``.shape`` and ``.ndim`` for the leading ``k_ndim`` axes;
    mutation of the K region is driven by the typed chain-rule primitives
    in :mod:`neml2.types.functions` (``fullify``, the exposing reductions
    on ``sum`` / ``mean`` / ``sum_sub_batch``).
    """

    _REGION = "k"

    def _bounds(self) -> tuple[int, int]:
        return (0, self._w.k_ndim)


# ---------------------------------------------------------------------------
# Sub-batch alignment + state combination
# ---------------------------------------------------------------------------


_W1 = TypeVar("_W1", bound=TensorWrapper)
_W2 = TypeVar("_W2", bound=TensorWrapper)
_W3 = TypeVar("_W3", bound=TensorWrapper)


def wrap_like(
    result_cls: type[_WT],
    data: torch.Tensor,
    source: TensorWrapper,
) -> _WT:
    """Construct ``result_cls`` around ``data`` with metadata copied from
    ``source``. Used by type-changing free functions (``tr(SR2) -> Scalar``,
    ``vol(SR2) -> SR2``, ...) whose output keeps the source wrapper's
    ``sub_batch_ndim``, per-axis storage state, AND K region verbatim.
    """
    return result_cls(
        data,
        sub_batch_ndim=source.sub_batch_ndim,
        sub_batch_state=source.sub_batch_state,
        sub_batch_meta=source.sub_batch_meta,
        k_ndim=source.k_ndim,
        k_state=source.k_state,
        k_pairing=source.k_pairing,
    )


def drop_sub_batch_state_axes(
    state: tuple[SubBatchStateFlag, ...],
    meta: tuple[int, ...],
    axes: Sequence[int],
) -> tuple[tuple[SubBatchStateFlag, ...], tuple[int, ...]]:
    """Drop the given sub-batch-relative axes from ``state`` and ``meta``."""
    if not state:
        return (), ()
    kill = set(axes)
    new_state = cast(
        "tuple[SubBatchStateFlag, ...]",
        tuple(s for i, s in enumerate(state) if i not in kill),
    )
    new_meta = tuple(m for i, m in enumerate(meta) if i not in kill)
    return new_state, new_meta


def combine_sub_batch_state(
    *wrappers: TensorWrapper,
) -> tuple[tuple[SubBatchStateFlag, ...], tuple[int, ...]]:
    """Combine per-axis state + meta across binary/n-ary op operands.

    Returns ``(state, meta)``. Caller has already passed the inputs
    through :func:`align_sub_batch` so they share a common
    ``sub_batch_ndim`` and aligned axis positions.

    Per-axis rule:

    - Output axis i is ``"broadcast"`` iff EVERY tangent input axis i
      is ``"broadcast"``. Compact tangent representation is preserved.
    - Otherwise output axis i is ``"full"``.

    Primals (empty ``sub_batch_state``) don't degrade the broadcast flag
    -- they contribute their logical extent for meta inference but
    cannot promote a tangent. Returns ``((), ())`` when no input carries
    non-empty state (legacy lane).
    """
    if not wrappers:
        return (), ()
    sb_ndim = wrappers[0].sub_batch_ndim
    if sb_ndim == 0:
        return (), ()
    have_any = any(getattr(w, "sub_batch_state", ()) for w in wrappers)
    if not have_any:
        return (), ()
    state_out: list[SubBatchStateFlag] = []
    meta_out: list[int] = []
    for i in range(sb_ndim):
        all_broadcast = True
        any_tangent = False
        extent = 0
        for w in wrappers:
            w_state = getattr(w, "sub_batch_state", ())
            w_meta = getattr(w, "sub_batch_meta", ())
            sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
            if not w_state:
                axis_extent = int(w.data.shape[sb_start + i])
            else:
                any_tangent = True
                axis_state = w_state[i]
                axis_extent = (
                    int(w_meta[i]) if axis_state == "broadcast" else int(w.data.shape[sb_start + i])
                )
                if axis_state != "broadcast":
                    all_broadcast = False
            if extent in (0, 1):
                extent = axis_extent
        flag: SubBatchStateFlag = "broadcast" if (any_tangent and all_broadcast) else "full"
        state_out.append(flag)
        meta_out.append(extent)
    return tuple(state_out), tuple(meta_out)


def align_scalar_base(scalar_data: torch.Tensor, base_ndim: int) -> torch.Tensor:
    """Pad a Scalar's data with ``base_ndim`` trailing length-1 axes so it
    broadcasts against a higher-base-rank wrapper.

    Skips unsqueezes when ``scalar_data`` is 0-d (broadcasts against any
    shape natively, and unsqueeze on a 0-d Constant IR node crashes
    Inductor's AOTI lowering).
    """
    if scalar_data.ndim == 0:
        return scalar_data
    for _ in range(base_ndim):
        scalar_data = scalar_data.unsqueeze(-1)
    return scalar_data


@overload
def align_sub_batch(a: _W1, b: _W2, /) -> tuple[tuple[_W1, _W2], int]: ...
@overload
def align_sub_batch(a: _W1, b: _W2, c: _W3, /) -> tuple[tuple[_W1, _W2, _W3], int]: ...
@overload
def align_sub_batch(*wrappers: TensorWrapper) -> tuple[tuple[TensorWrapper, ...], int]: ...
def align_sub_batch(*wrappers: TensorWrapper) -> tuple[tuple[TensorWrapper, ...], int]:
    """Pad each wrapper's ``sub_batch_ndim`` up to the common max.

    Direct Python analogue of C++ ``utils::align_intmd_dim``. For each
    input wrapper whose ``sub_batch_ndim < smax``, returns a view whose
    ``data`` has ``(smax - sub_batch_ndim)`` size-1 axes inserted at the
    start of its sub-batch region.

    Returns ``(aligned, smax)``.
    """
    if not wrappers:
        return (), 0
    smax = max(w.sub_batch_ndim for w in wrappers)
    if all(w.sub_batch_ndim == smax for w in wrappers):
        return tuple(wrappers), smax

    peer_state: list[SubBatchStateFlag] = ["full"] * smax
    peer_meta: list[int] = [0] * smax
    have_state = any(getattr(w, "sub_batch_state", ()) for w in wrappers)
    if have_state:
        for w in wrappers:
            w_state = getattr(w, "sub_batch_state", ())
            if not w_state:
                continue
            w_meta = getattr(w, "sub_batch_meta", ())
            offset = smax - w.sub_batch_ndim
            for i, s in enumerate(w_state):
                if peer_state[offset + i] == "full":
                    peer_state[offset + i] = s
                if peer_meta[offset + i] == 0:
                    extent = (
                        w_meta[i]
                        if s == "broadcast"
                        else int(w.data.shape[w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim + i])
                    )
                    peer_meta[offset + i] = int(extent)

    def _aligned(w: TensorWrapper) -> TensorWrapper:
        if w.sub_batch_ndim == smax:
            return w
        if w.data.ndim == 0:
            padded = w._rewrap(w.data, sub_batch_ndim=smax, sub_batch_state=(), sub_batch_meta=())
        else:
            padded = w.sub_batch.unsqueeze(0, smax - w.sub_batch_ndim)
        if have_state:
            offset = smax - w.sub_batch_ndim
            w_state = getattr(w, "sub_batch_state", ())
            w_meta = getattr(w, "sub_batch_meta", ())
            tail_state = w_state if w_state else ("full",) * w.sub_batch_ndim
            tail_meta = (
                w_meta
                if w_meta
                else tuple(
                    int(w.data.shape[w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim + i])
                    for i in range(w.sub_batch_ndim)
                )
            )
            head: tuple[SubBatchStateFlag, ...] = tuple(
                cast("SubBatchStateFlag", "broadcast" if peer_meta[i] > 0 else "full")
                for i in range(offset)
            )
            new_state = head + tuple(tail_state)
            new_meta = tuple(peer_meta[:offset]) + tuple(tail_meta)
            padded = padded._rewrap(
                padded.data,
                sub_batch_ndim=smax,
                sub_batch_state=cast("tuple[SubBatchStateFlag, ...]", new_state),
                sub_batch_meta=new_meta,
            )
        return padded

    return tuple(_aligned(w) for w in wrappers), smax


# ---------------------------------------------------------------------------
# K-region alignment + state combination (v2-parity plan §Appendix A)
# ---------------------------------------------------------------------------


@overload
def align_k(a: _W1, b: _W2, /) -> tuple[tuple[_W1, _W2], int]: ...
@overload
def align_k(a: _W1, b: _W2, c: _W3, /) -> tuple[tuple[_W1, _W2, _W3], int]: ...
@overload
def align_k(*wrappers: TensorWrapper) -> tuple[tuple[TensorWrapper, ...], int]: ...
def align_k(*wrappers: TensorWrapper) -> tuple[tuple[TensorWrapper, ...], int]:
    """Pad each wrapper's ``k_ndim`` up to the common max by inserting
    size-1 leading K axes on lower-rank operands.

    Padded axes default to ``state="broadcast"`` and ``pairing=None``
    (an empty K direction -- no per-site interpretation, broadcasts
    against any peer K dim).
    """
    if not wrappers:
        return (), 0
    kmax = max(w.k_ndim for w in wrappers)
    if all(w.k_ndim == kmax for w in wrappers):
        return tuple(wrappers), kmax

    def _aligned(w: TensorWrapper) -> TensorWrapper:
        if w.k_ndim == kmax:
            return w
        pad = kmax - w.k_ndim
        # If the underlying data is a 0-d scalar, torch broadcast handles the
        # alignment against any other K-padded operand "for free" -- we don't
        # need to physically prepend leading size-1 axes to the tensor.
        # Skipping the unsqueeze here avoids hitting an Inductor lowering
        # bug where ``aten.unsqueeze.default`` (and ``aten.reshape.default``)
        # crash on a ``Constant`` IR node (a scalar coefficient like
        # ``vf = 1.0 / (3.0 * K)`` that the constant-folder reduced after
        # all its inputs were baked parameters). The wrapper metadata still
        # claims ``k_ndim == kmax``; downstream ``op_fn(aa.data, bb.data)``
        # broadcasts the scalar against the K-padded side. The result of
        # the op has the full K-padded shape, so the post-op rewrap stays
        # consistent.
        if w.data.ndim == 0:
            new_data = w.data
        else:
            new_data = w.data
            for _ in range(pad):
                new_data = new_data.unsqueeze(0)
        new_state: tuple[KStateFlag, ...] = (cast("KStateFlag", "broadcast"),) * pad + tuple(
            w.k_state
        )
        new_pairing: tuple[int | None, ...] = (None,) * pad + tuple(w.k_pairing)
        return w._rewrap(
            new_data,
            sub_batch_ndim=w.sub_batch_ndim,
            k_ndim=kmax,
            k_state=new_state,
            k_pairing=new_pairing,
        )

    return tuple(_aligned(w) for w in wrappers), kmax


def combine_k_state(
    *wrappers: TensorWrapper,
) -> tuple[tuple[KStateFlag, ...], tuple[int | None, ...]]:
    """Combine per-K-axis state + pairing across n-ary op operands.

    Caller has already passed the inputs through :func:`align_k` so they
    share a common ``k_ndim``.

    Per-K-axis rules:

    - If any operand is ``"full"`` at axis i, output state[i] is ``"full"``.
    - Else if every operand is ``"broadcast"``, output state[i] is ``"broadcast"``.
    - Pairing combine:
        * If exactly one operand has a non-None pairing, output takes it.
        * If multiple operands have non-None pairings and they agree, output takes it.
        * Multiple non-None pairings that disagree raise.
        * Mixing a "full" lane with any K-paired-broadcast lane breaks the
          pairing (the full lane has no per-site interpretation) -- pairing
          collapses to ``None``.

    Returns ``((), ())`` when every operand has empty K state (the legacy
    primal lane).
    """
    if not wrappers:
        return (), ()
    kmax = wrappers[0].k_ndim
    if kmax == 0:
        return (), ()
    have_any = any(getattr(w, "k_state", ()) for w in wrappers)
    if not have_any:
        return (), ()
    state_out: list[KStateFlag] = []
    pairing_out: list[int | None] = []
    for i in range(kmax):
        all_broadcast = True
        any_kstate = False
        for w in wrappers:
            ws = getattr(w, "k_state", ())
            if not ws:
                continue
            any_kstate = True
            if ws[i] != "broadcast":
                all_broadcast = False
        flag: KStateFlag = "broadcast" if (any_kstate and all_broadcast) else "full"
        # Pairing resolution.
        seen: int | None = None
        seen_set = False
        for w in wrappers:
            wp = getattr(w, "k_pairing", ())
            ws = getattr(w, "k_state", ())
            if not wp or not ws:
                continue
            p = wp[i]
            if p is None:
                continue
            if not seen_set:
                seen = p
                seen_set = True
            elif seen != p:
                raise ValueError(
                    f"combine_k_state: K axis {i} has disagreeing pairings "
                    f"({seen!r} vs {p!r}) across operands; align upstream."
                )
        if flag == "full" and not all_broadcast:
            # Result is "full"; pairing only survives if every contributing
            # operand had the same pairing AND none of them was a primal/
            # naked-full lane that would dilute the per-site interpretation.
            # Conservative choice: drop pairing on any full result.
            pairing: int | None = None
        else:
            pairing = seen
        state_out.append(flag)
        pairing_out.append(pairing)
    return tuple(state_out), tuple(pairing_out)
