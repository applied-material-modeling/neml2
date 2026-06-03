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
methods (`.dtype`, `.device`, `.batch_shape`, `.to(...)`, `__repr__`) and
declares the class-level `BASE_NDIM` / `BASE_SHAPE` invariants that each
concrete wrapper (`Scalar`, `SR2`, `SSR4`) supplies. The concrete classes
are `@dataclass(frozen=True, eq=False)` and each declares two fields:
``data: torch.Tensor`` (the underlying tensor) and
``sub_batch_ndim: int = 0`` (number of trailing batch axes that act as
the structured "sub-batch" region — the Python-native analogue of the
C++ `intmd_dim`). `TensorWrapper`'s methods read both via ``self.data``
and ``self.sub_batch_ndim``.

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

``batch_shape`` returns ``dynamic + sub_batch`` (matching the C++
``batch_dim() = dynamic_dim + intmd_dim`` invariant), so code that
broadcasts over "everything but base" keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, TypeVar, overload

import torch

if TYPE_CHECKING:
    from typing_extensions import Self


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
    # Declared so type-checkers accept wrapper algebra on values typed as the
    # abstract ``TensorWrapper`` (the generic ``apply_chain_rule`` accumulation
    # and the residual / time-integration models). Returns are ``TensorWrapper``
    # (not ``Self``) so the concrete ``-> SR2`` / ``-> Scalar`` overrides stay
    # covariantly compatible, and ``other`` is left unannotated to match the
    # subclasses' own permissive runtime isinstance dispatch. The reflected
    # ``__radd__`` / ``__rsub__`` / ``__rmul__`` are declared too so
    # ``float * wrapper`` (the natural order in leaf math) typechecks at the
    # abstract level. Subclasses must override each reflected operator with an
    # explicit ``def`` (not a ``__radd__ = __add__`` alias) — the alias picks
    # up the base's ``-> TensorWrapper`` declaration instead of the concrete
    # subclass return type pyright would otherwise infer from ``__add__``.
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
        ``with_sub_batch`` re-tagging idiom (a wrapper view with a different
        rank tag) and lets parent dispatchers re-tag without copying.
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

    # ---- shape decomposition ----

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

    # ---- sub-batch retagging ----

    def with_sub_batch(self, ndim: int) -> Self:
        """Return a wrapper viewing the same ``data`` with ``sub_batch_ndim=ndim``.

        Validates that ``ndim`` is non-negative and does not exceed the
        number of non-base dims currently present.
        """
        if ndim < 0:
            raise ValueError(f"sub_batch_ndim must be non-negative, got {ndim}")
        batch_ndim = self.data.ndim - self.BASE_NDIM
        if ndim > batch_ndim:
            raise ValueError(
                f"sub_batch_ndim={ndim} exceeds available batch dims "
                f"({batch_ndim}); tensor shape={tuple(self.data.shape)}, "
                f"BASE_NDIM={self.BASE_NDIM}"
            )
        if ndim == self.sub_batch_ndim:
            return self  # type: ignore[return-value]
        return self._rewrap(self.data, sub_batch_ndim=ndim)

    def flatten_sub_batch(self) -> Self:
        """Return a wrapper with the sub-batch region demoted to dynamic batch."""
        return self.with_sub_batch(0)

    # ---- sub-batch-aware tensor ops ----
    #
    # These mirror the C++ ``intmd_*`` helpers used by KWN, finite-volume,
    # and crystal-plasticity. Each lives on the base class so every
    # concrete wrapper picks it up. The "sub-batch axis index" is
    # interpreted relative to the sub-batch region: ``dim=0`` is the
    # leading sub-batch axis, ``dim=-1`` (or ``dim=sub_batch_ndim-1``)
    # is the trailing one (immediately before ``base``).

    def _sub_batch_axis(self, dim: int) -> int:
        """Resolve a sub-batch-relative dim to an absolute axis on ``data``."""
        if dim < 0:
            dim += self.sub_batch_ndim
        if dim < 0 or dim >= self.sub_batch_ndim:
            raise IndexError(
                f"sub-batch dim {dim} out of range for sub_batch_ndim={self.sub_batch_ndim}"
            )
        # Sub-batch starts right after dynamic batch and ends right before base.
        sb_start = self.data.ndim - self.BASE_NDIM - self.sub_batch_ndim
        return sb_start + dim

    def _sub_batch_insert_axis(self, dim: int) -> int:
        """Resolve a sub-batch-relative insertion point (range ``[0, sub_batch_ndim]``)."""
        if dim < 0:
            dim += self.sub_batch_ndim + 1
        if dim < 0 or dim > self.sub_batch_ndim:
            raise IndexError(
                f"sub-batch insert dim {dim} out of range for sub_batch_ndim={self.sub_batch_ndim}"
            )
        sb_start = self.data.ndim - self.BASE_NDIM - self.sub_batch_ndim
        return sb_start + dim

    def sub_batch_expand(self, size: int, dim: int = 0) -> Self:
        """Insert a new sub-batch axis of given ``size`` at sub-batch position ``dim``.

        Increases ``sub_batch_ndim`` by 1. Mirrors C++ ``intmd_expand(size, d)``.
        ``dim=0`` (default) puts the new axis at the leading sub-batch
        position; ``dim=sub_batch_ndim`` (or ``-1`` after the +1 increment)
        puts it adjacent to ``base``.
        """
        if size <= 0:
            raise ValueError(f"sub_batch_expand size must be positive, got {size}")
        insert_at = self._sub_batch_insert_axis(dim)
        new_data = self.data.unsqueeze(insert_at).expand(
            *self.data.shape[:insert_at], size, *self.data.shape[insert_at:]
        )
        return self._rewrap(new_data, sub_batch_ndim=self.sub_batch_ndim + 1)

    def sub_batch_unsqueeze(self, dim: int = 0, n: int = 1) -> Self:
        """Insert ``n`` size-1 sub-batch axes at sub-batch position ``dim``.

        Increases ``sub_batch_ndim`` by ``n``. Mirrors C++ ``intmd_unsqueeze``.
        """
        if n < 0:
            raise ValueError(f"sub_batch_unsqueeze n must be non-negative, got {n}")
        if n == 0:
            return self  # type: ignore[return-value]
        insert_at = self._sub_batch_insert_axis(dim)
        new_data = self.data
        for _ in range(n):
            new_data = new_data.unsqueeze(insert_at)
        return self._rewrap(new_data, sub_batch_ndim=self.sub_batch_ndim + n)

    def sub_batch_diagonalize(self) -> Self:
        """Embed the trailing sub-batch axis as a diagonal block.

        Takes a wrapper with ``sub_batch_shape[-1] == L`` and returns one
        with the trailing sub-batch axis duplicated as an ``(L, L)``
        diagonal pair ($δ_{ij}$-style). Increases ``sub_batch_ndim`` by
        1. Mirrors C++ ``intmd_diagonalize``.

        Used by per-site rate models (e.g. KWN's
        ``RateLimitedPrecipitateGrowthRate``) to express block-diagonal
        derivatives without ever materialising the full dense block.
        """
        if self.sub_batch_ndim == 0:
            raise ValueError(
                "sub_batch_diagonalize requires at least one sub-batch dim; got sub_batch_ndim=0"
            )
        if self.BASE_NDIM != 0:
            # KWN-style block-diagonal derivatives only need scalar bins
            # today. Adding the base-dim shuffle is straightforward but
            # not exercised by any in-scope caller; refuse rather than
            # ship untested code.
            raise NotImplementedError(
                "sub_batch_diagonalize for BASE_NDIM > 0 wrappers is not "
                "implemented yet; no in-scope caller needs it."
            )
        new_data = torch.diag_embed(self.data)
        return self._rewrap(new_data, sub_batch_ndim=self.sub_batch_ndim + 1)

    @classmethod
    def sub_batch_cat(cls, tensors: Sequence[Self], dim: int = 0) -> Self:
        """Concatenate wrappers along sub-batch axis ``dim``.

        All inputs must be the same concrete wrapper type and share
        ``sub_batch_ndim``. Mirrors C++ ``intmd_cat``.
        """
        if not tensors:
            raise ValueError("sub_batch_cat requires at least one tensor")
        first = tensors[0]
        for t in tensors[1:]:
            if type(t) is not type(first):
                raise TypeError(
                    f"sub_batch_cat got mixed wrapper types: "
                    f"{type(first).__name__} vs {type(t).__name__}"
                )
            if t.sub_batch_ndim != first.sub_batch_ndim:
                raise ValueError(
                    f"sub_batch_cat got mismatched sub_batch_ndim: "
                    f"{first.sub_batch_ndim} vs {t.sub_batch_ndim}"
                )
        axis = first._sub_batch_axis(dim)
        new_data = torch.cat([t.data for t in tensors], dim=axis)
        return first._rewrap(new_data, sub_batch_ndim=first.sub_batch_ndim)

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
    against a per-sub-batch-site tensor (shape ``(B, *sub_batch, *base)``):
    PyTorch's implicit broadcast pads missing leading dims with ``1`` on the
    left, which yields ``(1, B)`` vs ``(B, 5)`` and fails at any ``B != 1``;
    inserting in the middle yields ``(B, 1, *base)`` vs ``(B, 5, *base)``
    which broadcasts correctly for any $B$.

    The Python ``sub_batch_unsqueeze(dim=0, n=k)`` primitive on
    :class:`TensorWrapper` already does exactly the C++
    ``intmd_unsqueeze(0, k)`` operation — this helper just orchestrates the
    pairwise/n-ary case for binary ops and free functions.

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
        w if w.sub_batch_ndim == smax else w.sub_batch_unsqueeze(0, smax - w.sub_batch_ndim)
        for w in wrappers
    ), smax
