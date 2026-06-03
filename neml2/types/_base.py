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

`TensorWrapper` is **not** a `@dataclass` itself — it carries the shared
methods (`.dtype`, `.device`, `.batch`, `.dynamic_batch`, `.sub_batch`,
`.base`, `.to(...)`, `__repr__`) and declares the class-level `BASE_NDIM`
/ `BASE_SHAPE` invariants that each concrete wrapper (`Scalar`, `SR2`,
`SSR4`) supplies. The concrete classes are `@dataclass(frozen=True,
eq=False)` and each declares two fields: ``data: torch.Tensor`` (the
underlying tensor) and ``sub_batch_ndim: int = 0`` (number of trailing
batch axes that act as the structured "sub-batch" region — the
Python-native analogue of the C++ `intmd_dim`).

Shape decomposition
-------------------

A wrapper's tensor shape splits into three regions::

    data.shape == (*dynamic_batch_shape, *sub_batch_shape, *base_shape)
                  └── leading ──┘└── middle (static) ──┘└─ BASE_NDIM ─┘

- ``base_shape`` is fixed by the wrapper type (e.g. ``(6,)`` for SR2).
- ``sub_batch_shape`` is the next ``sub_batch_ndim`` trailing batch axes.
  These broadcast like batch dims in forward ops but encode per-site
  structure (size bins, FV cells, slip systems) that the chain-rule
  machinery treats specially. Default 0 — most models don't need it.
- ``dynamic_batch_shape`` is everything left over (the variable dimension
  count that ``torch.export`` traces as dynamic).

The combined ``batch_shape`` (``dynamic + sub_batch``) matches the C++
``batch_dim() = dynamic_dim + intmd_dim`` invariant.

Region-view API
---------------

Shape-manipulating ops are exposed through four lightweight view
properties — ``t.batch``, ``t.dynamic_batch``, ``t.sub_batch``,
``t.base`` — instead of region-prefixed top-level methods. Each view is
computed on access (no storage, no pytree registration) and its
methods return a fresh wrapper, so chaining works:

    t.dynamic_batch.expand(20).sub_batch.unsqueeze(0)

The shared surface across views is ``.shape``, ``.ndim``, ``.unsqueeze``,
``.squeeze``, ``.expand``, ``.cat``. Region-specific extras:

- ``sub_batch.diagonalize()`` — embed trailing sub-batch axis as a diagonal block
- ``sub_batch.expand_at(size, dim=0)`` — insert one new sub-batch axis
  (the C++ ``intmd_expand`` pattern)
- ``sub_batch.retag(n)`` / ``sub_batch.flatten()`` — re-tag ``sub_batch_ndim``
- ``base.transpose(dim0, dim1)`` — swap two base axes (requires BASE_NDIM >= 2)

The ``base`` region is otherwise read-only — ``BASE_SHAPE`` is fixed by
the wrapper type, so ``.expand`` / ``.unsqueeze`` / ``.squeeze`` on
``base`` raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, overload

import torch

if TYPE_CHECKING:
    from typing_extensions import Self

_WT = TypeVar("_WT", bound="TensorWrapper")


class TensorWrapper:
    """Shared methods for the typed tensor wrappers.

    Subclasses (`@dataclass(frozen=True, eq=False)`) declare a
    ``data: torch.Tensor`` field and a ``sub_batch_ndim: int = 0`` field;
    the trailing ``BASE_NDIM`` dimensions of ``data`` have shape
    ``BASE_SHAPE`` (in Mandel packing where applicable), the next
    ``sub_batch_ndim`` dims are the sub-batch region, and everything before
    that is the dynamic batch.
    """

    data: torch.Tensor
    sub_batch_ndim: int
    BASE_NDIM: ClassVar[int]
    BASE_SHAPE: ClassVar[tuple[int, ...]]

    def __init__(self, data: torch.Tensor, sub_batch_ndim: int = 0) -> None:
        # Subclasses are ``@dataclass(frozen=True, eq=False)``; the dataclass
        # decorator generates the real ``__init__`` on each one and that
        # override is what's actually called at runtime. This stub exists
        # purely so type-checkers see the signature when callers do
        # ``t_cls(tensor)`` with ``t_cls: type[TensorWrapper]``.
        raise NotImplementedError("TensorWrapper is abstract; instantiate a concrete subclass.")

    # Each concrete ``@dataclass`` subclass defines these operators with its own
    # return type. The base declarations exist purely so type-checkers accept
    # wrapper algebra on values typed as the abstract ``TensorWrapper`` — the
    # generic chain-rule accumulation loops (``apply_chain_rule``) and the
    # residual / time-integration models that take ``TensorWrapper`` inputs.
    # They never run (the subclass override always wins), so they raise like
    # ``__init__``.
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
        """Defend against double-wrapping at the wrapper boundary.

        ``ComposedModel`` re-wraps each child's inputs via ``type_cls(state[n])``
        before dispatch. When the child is itself a ``ComposedModel`` (or
        ``ImplicitUpdate``) whose own dispatch then re-wraps, the same value
        gets wrapped twice — yielding e.g. ``SR2(SR2(tensor))`` where
        ``self.data`` is an ``SR2`` instead of a ``torch.Tensor``. That
        silently breaks every downstream op that expects a raw tensor.

        This hook (auto-called by the dataclass-generated ``__init__``)
        unwraps a same-type wrapper input to its inner ``data`` so
        double-wrapping becomes a no-op. A different-type wrapper input is
        almost certainly a bug — we raise so the caller fixes it rather
        than silently smearing the wrong tensor type through the chain.

        The ``sub_batch_ndim`` of the outer call wins. That matches the
        ``sub_batch.retag`` idiom (a wrapper view with a different rank
        tag) and lets parent dispatchers re-tag without copying.
        """
        if isinstance(self.data, TensorWrapper):
            inner = self.data
            if not isinstance(inner, type(self)):
                raise TypeError(
                    f"Cannot wrap a {type(inner).__name__} as a {type(self).__name__}; "
                    "wrapper types must match. Pass `inner.data` instead of the wrapper."
                )
            object.__setattr__(self, "data", inner.data)

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    # ---- shape decomposition (read-only properties; mirror the view shapes) ----

    @property
    def base_shape(self) -> torch.Size:
        if self.BASE_NDIM == 0:
            return torch.Size(())
        return self.data.shape[-self.BASE_NDIM :]

    @property
    def batch_shape(self) -> torch.Size:
        """All non-base dims, i.e. dynamic + sub-batch (matches C++ ``batch_dim``)."""
        return self.data.shape[: self.data.ndim - self.BASE_NDIM]

    @property
    def sub_batch_shape(self) -> torch.Size:
        n = self.sub_batch_ndim
        if n == 0:
            return torch.Size(())
        bsh = self.batch_shape
        return bsh[len(bsh) - n :]

    @property
    def dynamic_batch_shape(self) -> torch.Size:
        n = self.sub_batch_ndim
        bsh = self.batch_shape
        if n == 0:
            return bsh
        return bsh[: len(bsh) - n]

    # ---- region views ----

    @property
    def batch(self: Self) -> BatchView[Self]:
        """View into the combined batch region (``dynamic + sub_batch``).

        Read-only surface (``.shape``, ``.ndim``, ``.cat``). For
        shape-changing ops, use ``.dynamic_batch`` or ``.sub_batch``
        explicitly so the intent — whether a new axis is dynamic or
        structured — is unambiguous.
        """
        return BatchView(self)

    @property
    def dynamic_batch(self: Self) -> DynamicBatchView[Self]:
        """View into the dynamic batch region. ``sub_batch_ndim`` is preserved by ops here."""
        return DynamicBatchView(self)

    @property
    def sub_batch(self: Self) -> SubBatchView[Self]:
        """View into the sub-batch region. Ops here adjust ``sub_batch_ndim``."""
        return SubBatchView(self)

    @property
    def base(self: Self) -> BaseView[Self]:
        """View into the base region. Shape is fixed by the wrapper type;
        read-only except for ``transpose``.
        """
        return BaseView(self)

    # ---- internal: clone-with-new-tensor-and-metadata ----

    def _rewrap(self, data: torch.Tensor, *, sub_batch_ndim: int) -> Self:
        """Build a structurally-identical wrapper around new ``data``."""
        new: Self = object.__new__(type(self))
        object.__setattr__(new, "data", data)
        object.__setattr__(new, "sub_batch_ndim", sub_batch_ndim)
        return new

    def to(self, *args, **kwargs) -> Self:
        moved = self.data.to(*args, **kwargs)
        if moved is self.data:
            return self  # type: ignore[return-value]
        return self._rewrap(moved, sub_batch_ndim=self.sub_batch_ndim)

    # Concrete subclasses use ``@dataclass(...)`` which auto-generates
    # ``__repr__``; we deliberately don't override it here. The auto-repr
    # surfaces ``data`` and ``sub_batch_ndim`` directly, which is the
    # closest thing to a round-trippable representation we can offer.


# ---------------------------------------------------------------------------
# Region views — lightweight, computed-on-access wrappers that expose the
# shape-manipulating ops for one of the four regions of a TensorWrapper.
#
# Each view holds a reference to its wrapper, knows where its region
# starts and how long it is, translates region-relative dim indices to
# absolute axes on ``data``, and returns fresh wrappers (so calls chain).
#
# Views are *not* pytree-registered — they're transient handles, not
# part of the data model.
# ---------------------------------------------------------------------------


class _RegionView(Generic[_WT]):
    """Common machinery for the four region views.

    Subclasses define ``_REGION`` (a human-readable name used in error
    messages) and ``_bounds`` (the ``(start, length)`` slice of
    ``data.shape`` that the view governs).

    Generic over the wrapper type so view methods preserve the concrete
    return type: ``Scalar.sub_batch.unsqueeze(0)`` is typed as ``Scalar``.
    """

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

    # ---- region-relative dim resolution ----

    def _resolve_dim(self, dim: int) -> int:
        """Region-relative ``dim`` -> absolute axis on ``data``. Range ``[0, ndim)``."""
        n = self.ndim
        if dim < 0:
            dim += n
        if dim < 0 or dim >= n:
            raise IndexError(f"{self._REGION} dim {dim} out of range for {self._REGION}_ndim={n}")
        start, _ = self._bounds()
        return start + dim

    def _resolve_insert_dim(self, dim: int) -> int:
        """Region-relative insertion point. Range ``[0, ndim]``."""
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
    """Common shape-changing ops shared by ``batch`` / ``dynamic_batch`` / ``sub_batch``.

    The base region is fixed and uses a separate ``BaseView`` that
    doesn't inherit these ops.
    """

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        """Subclasses override to declare how an op changes ``sub_batch_ndim``."""
        raise NotImplementedError

    def unsqueeze(self, dim: int = -1, n: int = 1) -> _WT:
        """Insert ``n`` size-1 axes at region-relative position ``dim``."""
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
        """Drop a size-1 axis at region-relative position ``dim``."""
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
        """Make the region's shape equal to ``shape``.

        Mirrors ``torch.Tensor.expand`` left-padding: if the region's
        current ``ndim`` is less than ``len(shape)``, size-1 axes are
        inserted at the *start* of the region first. The result is made
        contiguous so downstream ``.contiguous()`` calls are redundant.

        Raises ``ValueError`` when ``len(shape) < ndim`` (would lose
        existing axes — use ``squeeze`` to drop them explicitly).
        """
        cur = self.shape
        target = tuple(shape)
        if len(cur) > len(target):
            raise ValueError(
                f"{self._REGION}.expand: target {target} has fewer dims than "
                f"current {tuple(cur)}; squeeze first if dropping axes is intended"
            )
        pad = len(target) - len(cur)
        start, length = self._bounds()
        # Insert ``pad`` size-1 axes at the start of the region so the
        # subsequent broadcast has matching ndim.
        new_data = self._w.data
        for _ in range(pad):
            new_data = new_data.unsqueeze(start)
        # Now build the full target shape on data: everything before the
        # region stays, region is broadcast to ``target``, everything after
        # the region stays.
        before = self._w.data.shape[:start]
        after = self._w.data.shape[start + length :]
        full_target = (*before, *target, *after)
        new_data = new_data.expand(full_target).contiguous()
        new_sb = self._new_sub_batch_ndim(axes_added=pad)
        return self._w._rewrap(new_data, sub_batch_ndim=new_sb)

    def cat(self, others: Sequence[_WT], dim: int = -1) -> _WT:
        """Concatenate ``self``'s wrapper with ``others`` along region-relative ``dim``.

        All inputs must be the same concrete wrapper type and share
        ``sub_batch_ndim``.
        """
        wrappers = (self._w, *others)
        for w in others:
            if type(w) is not type(self._w):
                raise TypeError(
                    f"{self._REGION}.cat: mixed wrapper types "
                    f"{type(self._w).__name__} vs {type(w).__name__}"
                )
            if w.sub_batch_ndim != self._w.sub_batch_ndim:
                raise ValueError(
                    f"{self._REGION}.cat: mismatched sub_batch_ndim "
                    f"{self._w.sub_batch_ndim} vs {w.sub_batch_ndim}"
                )
        axis = self._resolve_dim(dim)
        new_data = torch.cat([w.data for w in wrappers], dim=axis)
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim)


class BatchView(_MutableRegionView[_WT]):
    """Read-mostly view into the combined batch region (``dynamic + sub_batch``).

    Use ``.dynamic_batch`` or ``.sub_batch`` for shape-changing ops so
    intent is unambiguous. The mutable ops are inherited but disallowed
    on ``batch`` because the dynamic-vs-sub-batch split would be
    ambiguous; ``cat`` is the one allowed mutator and treats the batch
    region as a single axis range, leaving ``sub_batch_ndim`` unchanged.
    """

    _REGION = "batch"

    def _bounds(self) -> tuple[int, int]:
        return (0, self._w.data.ndim - self._w.BASE_NDIM)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        return self._w.sub_batch_ndim

    def unsqueeze(self, dim: int = -1, n: int = 1) -> _WT:
        raise TypeError(
            "batch.unsqueeze is ambiguous (would the new axis be dynamic or "
            "sub-batch?); use t.dynamic_batch.unsqueeze(...) or t.sub_batch.unsqueeze(...)"
        )

    def squeeze(self, dim: int = -1) -> _WT:
        raise TypeError(
            "batch.squeeze is ambiguous; "
            "use t.dynamic_batch.squeeze(...) or t.sub_batch.squeeze(...)"
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
        return (0, w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        # New dynamic axes are dynamic; sub_batch_ndim never changes.
        return self._w.sub_batch_ndim


class SubBatchView(_MutableRegionView[_WT]):
    """View into the sub-batch region. Shape-changing ops adjust ``sub_batch_ndim``.

    Extra ops beyond the shared surface:

    - ``expand_at(size, dim=0)`` — insert one new sub-batch axis of given
      size at sub-batch position ``dim``. Mirrors the C++ ``intmd_expand``
      pattern and the v2 ``sub_batch_expand`` method.
    - ``diagonalize()`` — embed the trailing sub-batch axis as an
      ``(L, L)`` diagonal block. The KWN block-diagonal-derivative kernel.
    - ``retag(n)`` / ``flatten()`` — pure metadata moves; no data copy.
    """

    _REGION = "sub_batch"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        return (sb_start, w.sub_batch_ndim)

    def _new_sub_batch_ndim(self, *, axes_added: int = 0, axes_removed: int = 0) -> int:
        return self._w.sub_batch_ndim + axes_added - axes_removed

    def expand_at(self, size: int, dim: int = 0) -> _WT:
        """Insert one new sub-batch axis of ``size`` at sub-batch position ``dim``.

        ``dim=0`` puts the new axis at the leading sub-batch position;
        ``dim=-1`` puts it adjacent to ``base``. Mirrors C++
        ``intmd_expand(size, d)``.
        """
        if size <= 0:
            raise ValueError(f"sub_batch.expand_at: size must be positive, got {size}")
        insert_at = self._resolve_insert_dim(dim)
        new_data = self._w.data.unsqueeze(insert_at).expand(
            *self._w.data.shape[:insert_at], size, *self._w.data.shape[insert_at:]
        )
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim + 1)

    def diagonalize(self) -> _WT:
        """Embed the trailing sub-batch axis as an ``(L, L)`` diagonal block.

        Takes a wrapper with ``sub_batch_shape[-1] == L`` and returns
        one with the trailing axis duplicated as an ``(L, L)`` diagonal
        pair ($δ_{ij}$-style). Increases ``sub_batch_ndim`` by 1.
        Mirrors C++ ``intmd_diagonalize``.

        Used by per-site rate models (e.g. KWN's
        ``RateLimitedPrecipitateGrowthRate``) to express
        block-diagonal derivatives without materialising the full
        dense block.
        """
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
        """Return a view of the same ``data`` with ``sub_batch_ndim=ndim``.

        Pure metadata move — no copy. Validates that ``ndim`` is
        non-negative and does not exceed the number of non-base dims
        currently present.
        """
        if ndim < 0:
            raise ValueError(f"sub_batch.retag: ndim must be non-negative, got {ndim}")
        batch_ndim = self._w.data.ndim - self._w.BASE_NDIM
        if ndim > batch_ndim:
            raise ValueError(
                f"sub_batch.retag: ndim={ndim} exceeds available batch dims "
                f"({batch_ndim}); tensor shape={tuple(self._w.data.shape)}, "
                f"BASE_NDIM={self._w.BASE_NDIM}"
            )
        if ndim == self._w.sub_batch_ndim:
            return self._w
        return self._w._rewrap(self._w.data, sub_batch_ndim=ndim)

    def flatten(self) -> _WT:
        """Demote the sub-batch region to dynamic batch (``retag(0)``)."""
        return self.retag(0)


class BaseView(_RegionView[_WT]):
    """View into the base region.

    The base shape is fixed by the wrapper type — you cannot
    ``expand``, ``unsqueeze``, ``squeeze``, ``cat``, or ``reshape`` it
    at this layer (build a different-typed wrapper instead). The one
    safe mutator is ``transpose``, useful for the ``R2`` 3×3 case.
    """

    _REGION = "base"

    def _bounds(self) -> tuple[int, int]:
        w = self._w
        return (w.data.ndim - w.BASE_NDIM, w.BASE_NDIM)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> _WT:
        """Swap two base axes. Requires ``BASE_NDIM >= 2``.

        Default arguments swap the trailing two axes, which is the
        ``R2.T`` / matrix-transpose case.
        """
        if self._w.BASE_NDIM < 2:
            raise TypeError(
                f"base.transpose requires BASE_NDIM >= 2; "
                f"{type(self._w).__name__} has BASE_NDIM={self._w.BASE_NDIM}"
            )
        ax0 = self._resolve_dim(dim0)
        ax1 = self._resolve_dim(dim1)
        new_data = self._w.data.transpose(ax0, ax1)
        return self._w._rewrap(new_data, sub_batch_ndim=self._w.sub_batch_ndim)


# ---------------------------------------------------------------------------
# Sub-batch alignment — mirror of C++ ``neml2::utils::align_intmd_dim``
# (``include/neml2/tensors/functions/utils.h:36``).
# ---------------------------------------------------------------------------


_W1 = TypeVar("_W1", bound=TensorWrapper)
_W2 = TypeVar("_W2", bound=TensorWrapper)
_W3 = TypeVar("_W3", bound=TensorWrapper)


def align_scalar_base(scalar_data: torch.Tensor, base_ndim: int) -> torch.Tensor:
    """Pad a Scalar's data with ``base_ndim`` trailing length-1 axes so it
    broadcasts against a higher-base-rank wrapper's data (Vec/SR2/R2/...).

    The typical use is the ``Scalar * Vec`` / ``Scalar * SR2`` / ``Scalar /
    SSR4`` operators: ``scalar.data`` arrives as ``(*B,)`` (or ``(*B,
    *sub_batch)``) and needs to align with the wrapper's ``(*B, ..., *base)``
    shape. Inserting trailing 1s gives ``(*B, ..., 1, 1, ...)`` that
    broadcasts cleanly against the wrapper's base axes.

    Skips the unsqueezes entirely when ``scalar_data`` is 0-d. A 0-d tensor
    broadcasts against any shape natively, AND ``aten.unsqueeze`` on a 0-d
    tensor that PyTorch's Inductor lowered as a ``Constant`` IR node crashes
    with ``AttributeError: 'Constant' object has no attribute 'data'`` --
    a real PyTorch AOTI bug that affects any model whose Scalar parameter
    was initialised from a HIT literal (the common case for ``coefficients
    = '1'`` etc.). The guard sidesteps the bug without changing eager
    behavior; a 0-d Scalar multiplied by a wrapper still broadcasts to the
    wrapper's full shape, exactly as before.
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

    Direct Python analogue of C++ ``utils::align_intmd_dim``. For each input
    wrapper whose ``sub_batch_ndim < smax``, returns a view whose ``data`` has
    ``(smax - sub_batch_ndim)`` size-1 axes inserted at the start of its
    sub-batch region — i.e. between the dynamic batch and any existing
    sub-batch axes. This is the only padding pattern that lets a global
    tensor (``sub_batch_ndim=0``, shape ``(B, *base)``) broadcast cleanly
    against a per-sub-batch-site tensor (shape ``(B, *sub_batch, *base)``).

    The ``sub_batch.unsqueeze(0, n=k)`` view primitive on
    :class:`TensorWrapper` does the per-wrapper work — this helper just
    orchestrates the pairwise/n-ary case for binary ops and free functions.

    Returns
    -------
    aligned : tuple[TensorWrapper, ...]
        The input wrappers, each padded as needed. Wrappers whose
        ``sub_batch_ndim`` already equals ``smax`` are returned unchanged
        (same identity).
    smax : int
        ``max(w.sub_batch_ndim for w in wrappers)``. Use as the
        ``sub_batch_ndim`` of the result wrapper.

    Examples
    --------
    Used inside every typed-wrapper arithmetic operator:

    >>> def __sub__(self, other):
    ...     [aa, bb], sb = align_sub_batch(self, other)
    ...     return type(self)(aa.data - bb.data, sub_batch_ndim=sb)

    And inside free functions that take 2+ wrappers:

    >>> def inner(a, b):
    ...     [aa, bb], sb = align_sub_batch(a, b)
    ...     return Scalar((aa.data * bb.data).sum(dim=-1), sub_batch_ndim=sb)
    """
    if not wrappers:
        return (), 0
    smax = max(w.sub_batch_ndim for w in wrappers)
    if all(w.sub_batch_ndim == smax for w in wrappers):
        return tuple(wrappers), smax
    return tuple(
        w if w.sub_batch_ndim == smax else w.sub_batch.unsqueeze(0, smax - w.sub_batch_ndim)
        for w in wrappers
    ), smax
