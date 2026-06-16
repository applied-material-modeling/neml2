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

"""Dynamic-base-shape `Tensor` -- the storage unit for variable-pair Jacobian
blocks at the equation-system / solver layer.

`Tensor` is the Python-native analogue of v2's ``neml2::Tensor``. Unlike the
fixed-base-shape typed wrappers (`Scalar`, `SR2`, ...) where ``BASE_SHAPE`` is
a class invariant, `Tensor` carries ``batch_ndim`` and ``sub_batch_ndim`` as
runtime instance fields; ``base_ndim`` is recovered as
``data.ndim - batch_ndim - sub_batch_ndim``. The same `Tensor` class can
represent a Scalar, an SR2-shaped block, or an arbitrary ``(R_row, R_col)``
Jacobian -- whatever shape the equation-system assembly needs.

Shape decomposition::

    data.shape == (*batch_shape, *sub_batch_shape, *base_shape)
                  └── leading ──┘└── per-site ────┘└── matrix ──┘

`AssembledVector` / `AssembledMatrix` (in `neml2.es`) wrap
`Tensor` instances and forward arithmetic / matmul / solve to its methods --
the dense-vs-block-diagonal solver branch is implicit in `sub_batch_ndim`
(both cases are the same ``torch.linalg.solve`` call; block-diagonal just
has extra leading sub-batch axes that batch naturally).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import ClassVar, Literal, TypeAlias

import torch
from torch.utils import _pytree as _torch_pytree

from neml2.types._base import TensorWrapper

#: Region selector for ops that need to disambiguate which group of axes
#: a ``dim`` index addresses. ``"batch"`` = leading dynamic-batch axes,
#: ``"sub_batch"`` = per-site axes, ``"base"`` = trailing matrix/vector axes.
AxisKind: TypeAlias = Literal["batch", "sub_batch", "base"]


@dataclass(frozen=True, eq=False)
class Tensor:
    """Dynamic-base-shape tensor with a ``(batch, sub_batch, base)`` runtime split."""

    data: torch.Tensor
    batch_ndim: int = 0
    sub_batch_ndim: int = 0
    sub_batch_state: tuple = ()
    sub_batch_meta: tuple = ()
    k_ndim: int = 0
    k_state: tuple = ()
    k_pairing: tuple = ()

    def __post_init__(self) -> None:
        if self.batch_ndim < 0 or self.sub_batch_ndim < 0:
            raise ValueError(
                f"Tensor batch_ndim and sub_batch_ndim must be non-negative, "
                f"got batch_ndim={self.batch_ndim}, sub_batch_ndim={self.sub_batch_ndim}"
            )
        if self.batch_ndim + self.sub_batch_ndim > self.data.ndim:
            raise ValueError(
                f"Tensor batch_ndim ({self.batch_ndim}) + sub_batch_ndim "
                f"({self.sub_batch_ndim}) exceeds data.ndim ({self.data.ndim})"
            )

    # ---- shape decomposition ----

    @property
    def base_ndim(self) -> int:
        # K axes (chain-rule seed directions) sit ENTIRELY LEFT of
        # ``batch``; exclude them so base counts only the trailing
        # region. Without subtracting ``k_ndim`` here, a Tensor that
        # carries K (e.g. inside ``fullify``) would silently count its
        # leading K axes as part of base, mis-locating the
        # ``sub_batch`` region in helpers like
        # :func:`~neml2.types.functions.fullify` that compute
        # ``sub_data_axis = data.ndim - base_ndim - sub_batch_ndim``.
        return self.data.ndim - self.k_ndim - self.batch_ndim - self.sub_batch_ndim

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def batch_shape(self) -> torch.Size:
        return self.data.shape[: self.batch_ndim]

    @property
    def sub_batch_shape(self) -> torch.Size:
        return self.data.shape[self.batch_ndim : self.batch_ndim + self.sub_batch_ndim]

    @property
    def base_shape(self) -> torch.Size:
        start = self.batch_ndim + self.sub_batch_ndim
        return self.data.shape[start:]

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    # ---- internal: clone-with-new-data ----

    def _rewrap(
        self,
        data: torch.Tensor,
        *,
        batch_ndim: int | None = None,
        sub_batch_ndim: int | None = None,
        sub_batch_state: tuple | None = None,
        sub_batch_meta: tuple | None = None,
        k_ndim: int | None = None,
        k_state: tuple | None = None,
        k_pairing: tuple | None = None,
    ) -> Tensor:
        # Mirror :meth:`TensorWrapper._rewrap` -- inherit each metadata
        # field independently from ``self`` when the caller doesn't
        # override. Without this, framework helpers like
        # :func:`~neml2.types.functions.fullify` that pass explicit
        # ``sub_batch_state`` / ``k_state`` / ``k_pairing`` updates
        # crash on the dynamic-base ``Tensor`` even though the same
        # call works fine on a static-base ``TensorWrapper``.
        new_sub_ndim = self.sub_batch_ndim if sub_batch_ndim is None else sub_batch_ndim
        same_sb = new_sub_ndim == self.sub_batch_ndim
        if sub_batch_state is None:
            sub_batch_state = self.sub_batch_state if same_sb else ()
        if sub_batch_meta is None:
            sub_batch_meta = self.sub_batch_meta if same_sb else ()
        new_k_ndim = self.k_ndim if k_ndim is None else k_ndim
        same_k = new_k_ndim == self.k_ndim
        if k_state is None:
            k_state = self.k_state if same_k else ()
        if k_pairing is None:
            k_pairing = self.k_pairing if same_k else ()
        return Tensor(
            data,
            self.batch_ndim if batch_ndim is None else batch_ndim,
            new_sub_ndim,
            sub_batch_state=sub_batch_state,
            sub_batch_meta=sub_batch_meta,
            k_ndim=new_k_ndim,
            k_state=k_state,
            k_pairing=k_pairing,
        )

    def to(self, *args, **kwargs) -> Tensor:
        moved = self.data.to(*args, **kwargs)
        if moved is self.data:
            return self
        return self._rewrap(moved)

    # ---- factories ----

    @classmethod
    def from_typed(cls, wrapper: TensorWrapper, *, batch_ndim: int | None = None) -> Tensor:
        """Construct a `Tensor` view of an existing `TensorWrapper`.

        The wrapper's ``sub_batch_ndim``, ``BASE_NDIM``, and
        ``sub_batch_labels`` are carried over verbatim; ``batch_ndim``
        defaults to ``data.ndim - sub_batch_ndim - BASE_NDIM`` (i.e.
        everything not claimed by sub-batch or base). Pass an explicit
        ``batch_ndim`` to override -- useful when the caller knows the
        dyn rank but the wrapper's leading axes encode a different
        layout (e.g. the JVP K-dim).
        """
        base = wrapper.BASE_NDIM
        sub = wrapper.sub_batch_ndim
        if batch_ndim is None:
            batch_ndim = wrapper.data.ndim - sub - base
        return cls(
            wrapper.data,
            batch_ndim,
            sub,
        )

    def as_typed(self, cls: type[TensorWrapper]) -> TensorWrapper:
        """View this `Tensor` as a `TensorWrapper` subclass.

        Requires the trailing dims of ``data`` to match ``cls.BASE_SHAPE``
        exactly. ``sub_batch_ndim`` and ``sub_batch_labels`` are
        preserved; any leading ``batch_ndim`` becomes the wrapper's
        dynamic batch.
        """
        if self.base_ndim != cls.BASE_NDIM:
            raise ValueError(
                f"Cannot view Tensor with base_ndim={self.base_ndim} as "
                f"{cls.__name__} (BASE_NDIM={cls.BASE_NDIM})"
            )
        if tuple(self.base_shape) != tuple(cls.BASE_SHAPE):
            raise ValueError(
                f"Cannot view Tensor with base_shape={tuple(self.base_shape)} as "
                f"{cls.__name__} (BASE_SHAPE={cls.BASE_SHAPE})"
            )
        return cls(
            self.data,
            sub_batch_ndim=self.sub_batch_ndim,
        )

    @classmethod
    def zeros(
        cls,
        *,
        batch_shape: Sequence[int] = (),
        sub_batch_shape: Sequence[int] = (),
        base_shape: Sequence[int] = (),
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Zero-filled `Tensor` with explicit region shapes.

        ``sub_batch_labels`` optionally attaches per-axis labels (length must
        match ``len(sub_batch_shape)``). Used by the BLOCK-aware
        ``_build_group_block`` to build zero blocks at the cell's canonical
        sub_batch shape with the cell's preserved labels.
        """
        shape = (*batch_shape, *sub_batch_shape, *base_shape)
        data = (
            torch.zeros(*shape, dtype=dtype, device=device)
            if shape
            else torch.zeros((), dtype=dtype, device=device)
        )
        return cls(
            data,
            len(batch_shape),
            len(sub_batch_shape),
        )

    @classmethod
    def zeros_like(cls, other: Tensor) -> Tensor:
        return cls(torch.zeros_like(other.data), other.batch_ndim, other.sub_batch_ndim)

    def new_zeros(
        self,
        *,
        batch_shape: Sequence[int] | None = None,
        sub_batch_shape: Sequence[int] | None = None,
        base_shape: Sequence[int] | None = None,
    ) -> Tensor:
        """Allocate a same-dtype, same-device zero `Tensor` with the given region shapes.

        Any region whose shape is omitted defaults to ``self``'s shape for
        that region -- handy for building a zero block that mirrors an
        existing block's dyn / sub-batch layout.
        """
        b = tuple(self.batch_shape) if batch_shape is None else tuple(batch_shape)
        s = tuple(self.sub_batch_shape) if sub_batch_shape is None else tuple(sub_batch_shape)
        ba = tuple(self.base_shape) if base_shape is None else tuple(base_shape)
        shape = (*b, *s, *ba)
        data = self.data.new_zeros(shape) if shape else self.data.new_zeros(())
        return Tensor(data, len(b), len(s))

    # ---- arithmetic ----

    def _binary(self, other: Tensor | float | int, op_fn) -> Tensor:
        if isinstance(other, (float, int)):
            return self._rewrap(op_fn(self.data, other))
        if not isinstance(other, Tensor):
            return NotImplemented  # type: ignore[return-value]
        aa, bb, sb, ba = _align_pair(self, other)
        # After alignment both operands share (batch_ndim, sub_batch_ndim).
        return Tensor(op_fn(aa.data, bb.data), ba, sb)

    def __neg__(self) -> Tensor:
        return self._rewrap(-self.data)

    def __add__(self, other) -> Tensor:
        return self._binary(other, torch.add)

    def __radd__(self, other) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other) -> Tensor:
        return self._binary(other, torch.sub)

    def __rsub__(self, other) -> Tensor:
        if isinstance(other, (float, int)):
            return self._rewrap(other - self.data)
        return NotImplemented

    def __mul__(self, other) -> Tensor:
        return self._binary(other, torch.mul)

    def __rmul__(self, other) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other) -> Tensor:
        return self._binary(other, torch.div)

    def __rtruediv__(self, other) -> Tensor:
        if isinstance(other, (float, int)):
            return self._rewrap(other / self.data)
        return NotImplemented

    # ---- matmul / solve ----

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix product on the trailing base axes.

        ``A @ B`` where ``A.base_ndim == 2`` and ``B.base_ndim in (1, 2)``.
        The leading ``(batch, sub_batch)`` axes broadcast right-aligned;
        the result has ``base_ndim`` equal to ``B.base_ndim``.

        Sub-batch handling:

        * Both operands have the same set of BLOCK axes: standard
          per-grain batched matmul via ``torch.matmul``.
        * One operand has BLOCK sub_batch axes the other lacks AND
          the contraction shapes mismatch by exactly the product of
          those extra axes: the matmul auto-unfolds the BLOCK side
          into its first base axis to align (the mxpc-style
          ``A_sp @ Y`` case where A_sp's per-grain side was already
          flattened to base by an upstream dense-edge reduction).
        * Any DENSE sub_batch axis at this point is a bug: assembly
          should have folded it before storage. We refuse and tell
          the caller to call ``flatten_sub_batch`` (or assembly's
          equivalent) first.
        """
        if not isinstance(other, Tensor):
            return NotImplemented  # type: ignore[return-value]
        if self.base_ndim != 2:
            raise ValueError(f"Tensor.__matmul__: LHS must have base_ndim=2, got {self.base_ndim}")
        if other.base_ndim not in (1, 2):
            raise ValueError(
                f"Tensor.__matmul__: RHS must have base_ndim in (1, 2), got {other.base_ndim}"
            )
        # Auto-unfold: when the contraction shape would mismatch and the
        # mismatch can be resolved by folding one operand's BLOCK
        # sub_batch axes into its first base axis, do it. Try RHS first
        # (the common mxpc case: LHS A_sp has no sub_batch, RHS Y is
        # BLOCK and needs unfolding); fall back to LHS otherwise.
        self_c = int(self.base_shape[-1])
        other_c = int(other.base_shape[0])
        if self_c != other_c:
            if other.sub_batch_ndim > self.sub_batch_ndim:
                # Fold RHS's extra sub_batch axes into RHS's first base.
                rhs_extra = prod(
                    other.sub_batch_shape[: other.sub_batch_ndim - self.sub_batch_ndim]
                )
                if self_c == rhs_extra * other_c:
                    other = _fold_extra_sub_batch_into_first_base(
                        other, other.sub_batch_ndim - self.sub_batch_ndim
                    )
            elif self.sub_batch_ndim > other.sub_batch_ndim:
                lhs_extra = prod(self.sub_batch_shape[: self.sub_batch_ndim - other.sub_batch_ndim])
                # Folding LHS's extra into LHS's first base wouldn't help
                # the contraction shape (which is LHS's LAST base); we
                # could fold into LHS's last base instead, but no current
                # caller needs that path. Defer and error below if still
                # mismatched after the simpler RHS rule.
                del lhs_extra
        aa, bb, sb, ba = _align_pair(self, other)
        if bb.base_ndim == 1:
            # Vector RHS: unsqueeze last for matmul, squeeze back.
            out_data = torch.matmul(aa.data, bb.data.unsqueeze(-1)).squeeze(-1)
        else:
            out_data = torch.matmul(aa.data, bb.data)
        return Tensor(out_data, ba, sb)

    def solve(self, other: Tensor) -> Tensor:
        """Solve ``self @ x = other`` for ``x`` via ``torch.linalg.solve``.

        ``self.base_ndim == 2`` (square matrix); ``other.base_ndim in
        (1, 2)``. ``(batch, sub_batch)`` axes are leading-broadcast
        through torch's batched solve -- so a block-diagonal system
        ``(*B, L, n, n) \\ (*B, L, n)`` solves L independent dense LUs
        per batch element without any explicit dispatch. That is the
        whole point of the dynamic-base design.

        Sub-batch dispatch mirrors :meth:`__matmul__`:

        * Same number of BLOCK axes on both sides: standard batched
          solve.
        * RHS has BLOCK axes the LHS lacks AND folding them into RHS's
          first base axis would align ``A``'s last base with the
          resulting column dim: do the unfold (the case where ``A`` was
          a dense per-grain coupling already folded by assembly and
          ``b`` is per-grain BLOCK).
        * DENSE sub_batch on either operand at this point is a bug:
          assembly should have folded it.
        """
        if self.base_ndim != 2:
            raise ValueError(f"Tensor.solve: A must have base_ndim=2, got {self.base_ndim}")
        if other.base_ndim not in (1, 2):
            raise ValueError(
                f"Tensor.solve: b must have base_ndim in (1, 2), got {other.base_ndim}"
            )
        # Auto-unfold RHS's extra BLOCK sub_batch axes when the column
        # dim of A wouldn't otherwise align with the row dim of b. The
        # KWN case: A is (B, N, N) (dense per-grain folded by assembly),
        # b is (B, N, 1) BLOCK sub_batch on N -- unfold b's BLOCK into
        # its first base axis to give (B, N, 1) base / sub=0, then
        # standard solve.
        self_c = int(self.base_shape[-1])
        other_first_base = int(other.base_shape[0]) if other.base_ndim >= 1 else 1
        if self_c != other_first_base and other.sub_batch_ndim > self.sub_batch_ndim:
            extra = other.sub_batch_ndim - self.sub_batch_ndim
            rhs_extra_total = prod(other.sub_batch_shape[:extra])
            if self_c == rhs_extra_total * other_first_base:
                other = _fold_extra_sub_batch_into_first_base(other, extra)
        aa, bb, sb, ba = _align_pair(self, other)
        if bb.base_ndim == 1:
            out_data = torch.linalg.solve(aa.data, bb.data.unsqueeze(-1)).squeeze(-1)
        else:
            out_data = torch.linalg.solve(aa.data, bb.data)
        return Tensor(out_data, ba, sb)

    # ---- reductions over base ----
    #
    # ``sum``, ``dot``, ``norm``, ``norm_sq`` live in
    # :mod:`neml2.types.functions` as free functions on ``t.base`` views.
    # Per NEML2 convention, mathematical ops that don't modify type
    # invariants are free functions; only shape / invariant manipulators
    # stay as methods on the type.

    # ---- reshape ----

    def flatten_base(self) -> Tensor:
        """Collapse all base axes into one trailing axis; result has ``base_ndim=1``."""
        if self.base_ndim == 0:
            return Tensor(self.data.unsqueeze(-1), self.batch_ndim, self.sub_batch_ndim)
        if self.base_ndim == 1:
            return self
        new_data = self.data.reshape(*self.batch_shape, *self.sub_batch_shape, -1)
        return Tensor(new_data, self.batch_ndim, self.sub_batch_ndim)

    def unflatten_base(self, *shape: int) -> Tensor:
        """Reshape the base region to ``shape``. Total base storage must match."""
        target = (*self.batch_shape, *self.sub_batch_shape, *shape)
        new_data = self.data.reshape(target)
        return Tensor(new_data, self.batch_ndim, self.sub_batch_ndim)

    def flatten_sub_batch(self) -> Tensor:
        """Absorb sub-batch axes into a NEW leading base axis.

        Result has ``sub_batch_ndim=0`` and ``base_ndim`` increased by 1
        (the new leading base axis equals ``prod(sub_batch_shape)``).
        Use to materialise a block-diagonal-storage Tensor as one flat
        matrix when the downstream consumer demands base-only storage.
        """
        if self.sub_batch_ndim == 0:
            return self
        sub_total = 1
        for s in self.sub_batch_shape:
            sub_total *= int(s)
        new_data = self.data.reshape(*self.batch_shape, sub_total, *self.base_shape)
        # Sub-batch axes were folded into a new leading base axis. The
        # ``"dense"`` distinction disappears at this point: the resulting
        # base-leading axis carries the per-grain stride explicitly.
        return Tensor(new_data, self.batch_ndim, 0)

    def fold_preserving(
        self,
        preserved_idx: list[int],
        canonical_order: tuple[str, ...],
        other_idx: list[int],
        fold_size: int,
    ) -> Tensor:
        """Permute preserved sub_batch axes to leading position (in
        ``canonical_order``), then fold (remaining_sub + base) into a
        single trailing axis.

        Result has ``batch_ndim`` unchanged, ``sub_batch_ndim =
        len(canonical_order)``, ``sub_batch_labels = canonical_order``,
        ``base_ndim=1``, ``base[0] = fold_size``.

        Used by :meth:`~neml2.es.AssembledVector.from_dict`'s
        preserved-label storage path to keep declared per-axis sub_batch
        dims as real sub_batch on the assembled slab while folding the
        rest.
        """
        dyn_ndim = self.batch_ndim
        sb_ndim = self.sub_batch_ndim
        base_ndim = self.base_ndim
        # Defensive: callers (assembly path) compute preserved_idx /
        # other_idx from the LAYOUT's sub_batch_shape, but this Tensor's
        # sub_batch_ndim is what actually drives the permutation. A mismatch
        # means the wrapper was built with a different sub_batch_ndim than
        # the layout expects -- usually because some upstream rewrap forgot
        # to thread sub_batch_ndim through. Raise informatively so the
        # offending construction site is easy to find.
        max_idx = max(preserved_idx) if preserved_idx else -1
        if other_idx:
            max_idx = max(max_idx, max(other_idx))
        if max_idx >= sb_ndim:
            raise ValueError(
                f"Tensor.fold_preserving: wrapper sub_batch_ndim={sb_ndim} "
                f"too low for requested preserved_idx={preserved_idx} "
                f"other_idx={other_idx}. Caller layout expects "
                f"sub_batch_ndim >= {max_idx + 1}. Check that the wrapper "
                f"feeding from_dict was built with the right sub_batch_ndim."
            )
        perm = list(range(dyn_ndim))
        perm += [dyn_ndim + i for i in preserved_idx]
        perm += [dyn_ndim + i for i in other_idx]
        perm += [dyn_ndim + sb_ndim + i for i in range(base_ndim)]
        permuted = self.data.permute(perm).contiguous()
        preserved_extents = tuple(int(self.sub_batch_shape[i]) for i in preserved_idx)
        folded = permuted.reshape(*permuted.shape[:dyn_ndim], *preserved_extents, fold_size)
        return Tensor(
            folded,
            batch_ndim=dyn_ndim,
            sub_batch_ndim=len(canonical_order),
        )

    def flatten_sub_batch_into_first_base_axis(self) -> Tensor:
        """Fold sub_batch axes INTO (not beside) the first base axis.

        Result has ``sub_batch_ndim=0`` and ``base_ndim`` unchanged; the
        first base axis grows from ``base_shape[0]`` to ``prod(sub_batch)
        * base_shape[0]``. Distinct from :meth:`flatten_sub_batch` which
        inserts the collapsed sub_batch as a NEW leading base axis.

        This is the inverse of the assembly's "BLOCK-compact preserves
        per-grain structure" decision: when downstream code needs the
        fully-unfolded form (e.g. a global-row matmul against per-grain
        col blocks), call this. Requires ``base_ndim >= 1``.
        """
        if self.sub_batch_ndim == 0:
            return self
        if self.base_ndim == 0:
            raise ValueError(
                "flatten_sub_batch_into_first_base_axis: base_ndim=0; use "
                "``flatten_sub_batch`` to add a new base axis instead."
            )
        sub_total = 1
        for s in self.sub_batch_shape:
            sub_total *= int(s)
        first_base = int(self.base_shape[0])
        rest_base = tuple(int(s) for s in self.base_shape[1:])
        new_data = self.data.reshape(*self.batch_shape, sub_total * first_base, *rest_base)
        return Tensor(new_data, self.batch_ndim, 0)

    # ---- region views ----

    @property
    def batch(self) -> _BatchView:
        """View into the leading (dynamic) batch region. Ops preserve
        ``sub_batch_ndim``. Supports ``unsqueeze`` / ``squeeze`` / ``expand``
        / ``broadcast_to`` / ``cat``."""
        return _BatchView(self)

    @property
    def sub_batch(self) -> _SubBatchView:
        """View into the per-site sub-batch region. Ops adjust
        ``sub_batch_ndim``."""
        return _SubBatchView(self)

    @property
    def base(self) -> _BaseView:
        """View into the trailing base region. Unlike :class:`TensorWrapper`,
        the base region is mutable on a dynamic-base `Tensor` -- ``expand``
        / ``unsqueeze`` / ``squeeze`` / ``cat`` all work."""
        return _BaseView(self)

    # ---- repr ----
    def __repr__(self) -> str:
        return (
            f"Tensor(batch_shape={tuple(self.batch_shape)}, "
            f"sub_batch_shape={tuple(self.sub_batch_shape)}, "
            f"base_shape={tuple(self.base_shape)}, dtype={self.dtype}, device={self.device})"
        )


def _align_pair(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, int, int]:
    """Pad two `Tensor`s to a common ``(batch_ndim, sub_batch_ndim)`` for binary ops.

    Right-aligned broadcast: the operand with smaller ``batch_ndim`` gets
    size-1 axes inserted at the *front* of its batch region; the operand
    with smaller ``sub_batch_ndim`` gets size-1 axes inserted at the
    *front* of its sub-batch region (i.e. right after the padded batch).
    Newly-inserted sub_batch axes default to ``"block"`` kind.

    Returns ``(a_aligned, b_aligned, sub_batch_ndim, batch_ndim)`` of
    the common shape.
    """
    sb = max(a.sub_batch_ndim, b.sub_batch_ndim)
    ba = max(a.batch_ndim, b.batch_ndim)
    return _pad_to(a, ba, sb), _pad_to(b, ba, sb), sb, ba


def _fold_extra_sub_batch_into_first_base(t: Tensor, n_axes: int) -> Tensor:
    """Fold ``n_axes`` leading sub_batch axes into the first base axis.

    Used by matmul's auto-unfold path: when one operand has more
    sub_batch axes than the other and the contraction shape mismatches
    by exactly the product of the extra axes, fold them into the
    operand's first base axis so the matmul aligns.

    The remaining sub_batch axes (and their labels) shift down by
    ``n_axes``.
    """
    if n_axes <= 0:
        return t
    if n_axes > t.sub_batch_ndim:
        raise ValueError(
            f"_fold_extra_sub_batch_into_first_base: requested {n_axes} fold but "
            f"only {t.sub_batch_ndim} sub_batch axes available"
        )
    if t.base_ndim == 0:
        raise ValueError("_fold_extra_sub_batch_into_first_base: base_ndim=0; nowhere to fold")
    folded_extents = tuple(int(s) for s in t.sub_batch_shape[:n_axes])
    fold_total = prod(folded_extents) if folded_extents else 1
    first_base = int(t.base_shape[0])
    rest_base = tuple(int(s) for s in t.base_shape[1:])
    keep_extents = tuple(int(s) for s in t.sub_batch_shape[n_axes:])
    new_shape = (*t.batch_shape, *keep_extents, fold_total * first_base, *rest_base)
    new_data = t.data.reshape(*new_shape)
    return Tensor(
        new_data,
        t.batch_ndim,
        t.sub_batch_ndim - n_axes,
    )


def _pad_to(t: Tensor, batch_ndim: int, sub_batch_ndim: int) -> Tensor:
    """Insert size-1 axes to bring ``t`` up to the target region ndims.

    New sub_batch axes are inserted at the FRONT (right-aligned
    broadcast) and inherit anonymous labels; the original axes' labels
    shift to the end of the sub_batch region.
    """
    if t.batch_ndim == batch_ndim and t.sub_batch_ndim == sub_batch_ndim:
        return t
    cur = t.data
    sb_pad = sub_batch_ndim - t.sub_batch_ndim
    if sb_pad < 0:
        raise ValueError(
            f"_pad_to: cannot shrink sub_batch_ndim from {t.sub_batch_ndim} to {sub_batch_ndim}"
        )
    for _ in range(sb_pad):
        cur = cur.unsqueeze(t.batch_ndim)
    ba_pad = batch_ndim - t.batch_ndim
    if ba_pad < 0:
        raise ValueError(f"_pad_to: cannot shrink batch_ndim from {t.batch_ndim} to {batch_ndim}")
    for _ in range(ba_pad):
        cur = cur.unsqueeze(0)
    return Tensor(cur, batch_ndim, sub_batch_ndim)


# ---------------------------------------------------------------------------
# Region views -- transient handles exposing shape-manipulating ops over
# one of the three regions (batch, sub_batch, base) of a :class:`Tensor`.
# Mirror :mod:`neml2.types._base`'s view hierarchy, adapted for the
# dynamic-base layout where ``base_ndim`` is also runtime-mutable.
# ---------------------------------------------------------------------------


class _RegionView:
    """Common machinery for the three :class:`Tensor` region views.

    Subclasses define ``_REGION`` (human-readable name for error messages)
    and ``_bounds()`` (``(start, length)`` slice into ``data.shape``) and
    ``_new_ndims(axes_added, axes_removed)`` (how an op updates the parent
    ``(batch_ndim, sub_batch_ndim)`` pair).
    """

    _REGION: ClassVar[str]
    __slots__ = ("_t",)

    def __init__(self, t: Tensor) -> None:
        self._t = t

    # ---- subclass hooks ----

    def _bounds(self) -> tuple[int, int]:
        raise NotImplementedError

    def _new_ndims(self, *, axes_added: int = 0, axes_removed: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    # ---- read-only properties ----

    @property
    def shape(self) -> torch.Size:
        start, length = self._bounds()
        if length == 0:
            return torch.Size(())
        return self._t.data.shape[start : start + length]

    @property
    def ndim(self) -> int:
        return self._bounds()[1]

    # ---- region-relative dim resolution ----

    def _resolve_dim(self, dim: int) -> int:
        """Region-relative ``dim`` -> absolute axis on ``data.shape``."""
        n = self.ndim
        if n == 0:
            raise ValueError(f"Tensor: {self._REGION} region is empty (no axis to address)")
        if dim < 0:
            dim += n
        if dim < 0 or dim >= n:
            raise IndexError(
                f"Tensor: {self._REGION} dim {dim} out of range for {self._REGION}_ndim={n}"
            )
        start, _ = self._bounds()
        return start + dim

    def _resolve_insert_dim(self, dim: int) -> int:
        """Region-relative insertion point in ``[0, ndim]``."""
        n = self.ndim
        if dim < 0:
            dim += n + 1
        if dim < 0 or dim > n:
            raise IndexError(
                f"Tensor: {self._REGION} insert dim {dim} out of range for {self._REGION}_ndim={n}"
            )
        start, _ = self._bounds()
        return start + dim

    # ---- shared mutators ----
    #
    # All shape-changing region ops route through ``self._t._rewrap`` so
    # ``sub_batch_labels`` inheritance follows the single rule defined on
    # ``Tensor._rewrap`` (preserve when ``sub_batch_ndim`` is unchanged,
    # drop otherwise). No per-op label threading needed.

    def unsqueeze(self, dim: int = -1, n: int = 1) -> Tensor:
        """Insert ``n`` size-1 axes at region-relative position ``dim``."""
        if n < 0:
            raise ValueError(f"Tensor.{self._REGION}.unsqueeze: n must be >= 0, got {n}")
        if n == 0:
            return self._t
        insert_at = self._resolve_insert_dim(dim)
        new_data = self._t.data
        for _ in range(n):
            new_data = new_data.unsqueeze(insert_at)
        ba, sb = self._new_ndims(axes_added=n)
        return self._t._rewrap(new_data, batch_ndim=ba, sub_batch_ndim=sb)

    def squeeze(self, dim: int = -1) -> Tensor:
        """Drop a size-1 axis at region-relative position ``dim``."""
        axis = self._resolve_dim(dim)
        if self._t.data.shape[axis] != 1:
            raise ValueError(
                f"Tensor.{self._REGION}.squeeze: axis {dim} has size "
                f"{self._t.data.shape[axis]}, expected 1"
            )
        new_data = self._t.data.squeeze(axis)
        ba, sb = self._new_ndims(axes_removed=1)
        return self._t._rewrap(new_data, batch_ndim=ba, sub_batch_ndim=sb)

    def expand(self, *shape: int) -> Tensor:
        """Make this region's shape equal to ``shape``.

        Left-pads with size-1 axes if the current region rank is less than
        ``len(shape)``, then broadcasts via ``torch.Tensor.expand``. Result
        is made ``contiguous`` so downstream reshapes don't trip on strides.
        ``len(shape) < current ndim`` is an error (use ``squeeze`` to drop
        axes explicitly).
        """
        cur = self.shape
        target = tuple(shape)
        if len(target) < len(cur):
            raise ValueError(
                f"Tensor.{self._REGION}.expand: target {target} has fewer dims "
                f"than current {tuple(cur)}; squeeze first if dropping axes is intended"
            )
        pad = len(target) - len(cur)
        start, length = self._bounds()
        new_data = self._t.data
        for _ in range(pad):
            new_data = new_data.unsqueeze(start)
        before = self._t.data.shape[:start]
        after = self._t.data.shape[start + length :]
        full_target = (*before, *target, *after)
        new_data = new_data.expand(full_target).contiguous()
        ba, sb = self._new_ndims(axes_added=pad)
        return self._t._rewrap(new_data, batch_ndim=ba, sub_batch_ndim=sb)

    def broadcast_to(self, *shape: int) -> Tensor:
        """Like ``expand`` but also trims leading size-1 axes when
        ``len(shape) < self.ndim``.

        Useful when the parent ``Tensor`` carries placeholder size-1 axes
        (e.g. an identity-seed scaffold) that should be removed to match
        a shorter target. Non-singleton excess axes raise.
        """
        cur = self.shape
        target = tuple(shape)
        if tuple(cur) == target:
            return self._t
        if len(target) >= len(cur):
            return self.expand(*target)
        # Need to trim leading axes; each must be size 1.
        excess = len(cur) - len(target)
        for s in cur[:excess]:
            if int(s) != 1:
                raise ValueError(
                    f"Tensor.{self._REGION}.broadcast_to: extra leading dim of "
                    f"size {int(s)} cannot be trimmed (target has fewer dims)"
                )
        start, _ = self._bounds()
        new_data = self._t.data
        for _ in range(excess):
            new_data = new_data.squeeze(start)
        ba, sb = self._new_ndims(axes_removed=excess)
        trimmed = self._t._rewrap(new_data, batch_ndim=ba, sub_batch_ndim=sb)
        # After trimming, the view's bounds are different (region is shorter);
        # rebuild the view on the new Tensor to expand to the final target.
        view_cls = type(self)
        return view_cls(trimmed).expand(*target)

    def __getitem__(self, key) -> Tensor:
        """Index / slice within this region only.

        The ``key`` addresses the region's axes; pre-region and post-region
        axes pass through as ``:`` (all). Supports ``Ellipsis`` (expands to
        fill the region) and ``None`` (inserts a new region axis).

        Returns a :class:`Tensor` with the region's ndim updated to reflect
        the operation (integer indices remove an axis, ``None`` adds one).
        Batch / sub-batch / base counts on the other regions stay put.

        Example::

            tensor.base[..., -1]   # last element of the last base axis
            tensor.base[..., 1:3]  # slice the last base axis
            tensor.sub_batch[0]    # pick the first sub-batch site
        """
        if not isinstance(key, tuple):
            key = (key,)
        start, length = self._bounds()
        # Expand any Ellipsis inside the user's key to slice(None) fills.
        if Ellipsis in key:
            n_ellipsis = sum(1 for k in key if k is Ellipsis)
            if n_ellipsis > 1:
                raise IndexError(f"Tensor.{self._REGION}.__getitem__: at most one Ellipsis allowed")
            explicit_axes = sum(1 for k in key if k is not Ellipsis and k is not None)
            n_fill = length - explicit_axes
            if n_fill < 0:
                raise IndexError(
                    f"Tensor.{self._REGION}.__getitem__: key has more indices "
                    f"({explicit_axes}) than region ndim ({length})"
                )
            idx = key.index(Ellipsis)
            key = key[:idx] + (slice(None),) * n_fill + key[idx + 1 :]
        # Build the full data-side key: pre-region : , user's key, post-region :
        pre = (slice(None),) * start
        post = (slice(None),) * (self._t.data.ndim - start - length)
        full_key = pre + key + post
        new_data = self._t.data[full_key]
        # Count how the region's ndim changed: int indices remove an axis,
        # `None` inserts one; slices keep it.
        added = sum(1 for k in key if k is None)
        removed = sum(1 for k in key if isinstance(k, int))
        ba, sb = self._new_ndims(axes_added=added, axes_removed=removed)
        return self._t._rewrap(new_data, batch_ndim=ba, sub_batch_ndim=sb)


class _BatchView(_RegionView):
    """View into the leading (dynamic) batch region. Ops preserve
    ``sub_batch_ndim``."""

    _REGION = "batch"

    def _bounds(self) -> tuple[int, int]:
        return (0, self._t.batch_ndim)

    def _new_ndims(self, *, axes_added: int = 0, axes_removed: int = 0) -> tuple[int, int]:
        return (self._t.batch_ndim + axes_added - axes_removed, self._t.sub_batch_ndim)


class _SubBatchView(_RegionView):
    """View into the per-site sub-batch region. Ops adjust ``sub_batch_ndim``."""

    _REGION = "sub_batch"

    def _bounds(self) -> tuple[int, int]:
        return (self._t.batch_ndim, self._t.sub_batch_ndim)

    def _new_ndims(self, *, axes_added: int = 0, axes_removed: int = 0) -> tuple[int, int]:
        return (self._t.batch_ndim, self._t.sub_batch_ndim + axes_added - axes_removed)


class _BaseView(_RegionView):
    """View into the trailing base region. Unlike :class:`TensorWrapper`'s
    fixed-base view, the base region on a dynamic-base `Tensor` is mutable
    via ``expand`` / ``unsqueeze`` / ``squeeze`` / ``cat`` / ``transpose``."""

    _REGION = "base"

    def _bounds(self) -> tuple[int, int]:
        return (self._t.batch_ndim + self._t.sub_batch_ndim, self._t.base_ndim)

    def _new_ndims(self, *, axes_added: int = 0, axes_removed: int = 0) -> tuple[int, int]:
        # New base axes don't shift batch / sub-batch counts.
        del axes_added, axes_removed
        return (self._t.batch_ndim, self._t.sub_batch_ndim)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> Tensor:
        """Swap two base axes -- the matrix-transpose case for a
        ``base_ndim >= 2`` Tensor. Defaults to the trailing two axes."""
        if self._t.base_ndim < 2:
            raise ValueError(
                f"Tensor.base.transpose requires base_ndim >= 2; got base_ndim={self._t.base_ndim}"
            )
        ax0 = self._resolve_dim(dim0)
        ax1 = self._resolve_dim(dim1)
        new_data = self._t.data.transpose(ax0, ax1)
        return Tensor(new_data, self._t.batch_ndim, self._t.sub_batch_ndim)


# ---------------------------------------------------------------------------
# Free functions: cat / stack over region views.
# Match the existing convention in :mod:`neml2.types.functions` -- the view
# carries the axis-region information, so the call site reads
# ``cat([a.base, b.base, c.base], dim=-1)``.
# ---------------------------------------------------------------------------


def cat(views: Sequence[_RegionView], dim: int = -1) -> Tensor:
    """Concatenate :class:`Tensor`s along a region-relative axis.

    All views must be the same region kind (all ``.batch``, all ``.sub_batch``,
    or all ``.base``) over Tensors that share ``batch_ndim`` /
    ``sub_batch_ndim`` AND per-axis ``sub_batch_labels``. The cat-axis
    size is the only thing allowed to vary.
    """
    if not views:
        raise ValueError("cat: views must be non-empty")
    first = views[0]
    region_cls = type(first)
    head = first._t
    for v in views[1:]:
        if type(v) is not region_cls:
            raise TypeError(
                f"cat: heterogeneous region views {region_cls.__name__} vs {type(v).__name__}"
            )
        t = v._t
        if t.batch_ndim != head.batch_ndim or t.sub_batch_ndim != head.sub_batch_ndim:
            raise ValueError(
                f"cat: mismatched region ndims "
                f"({head.batch_ndim, head.sub_batch_ndim}) vs "
                f"({t.batch_ndim, t.sub_batch_ndim})"
            )
    axis = first._resolve_dim(dim)
    new_data = torch.cat([v._t.data for v in views], dim=axis)
    return Tensor(new_data, head.batch_ndim, head.sub_batch_ndim)


def stack(views: Sequence[_RegionView], dim: int = 0) -> Tensor:
    """Stack :class:`Tensor`s along a NEW region-relative axis.

    All views must be the same region kind over Tensors that share concrete
    region ndims and data shape. The new axis is inserted at region-relative
    position ``dim``; the parent region's ndim grows by one (so stacking on
    a ``.sub_batch`` view bumps ``sub_batch_ndim``). A new sub_batch
    axis from a sub_batch stack inherits an anonymous label.
    """
    if not views:
        raise ValueError("stack: views must be non-empty")
    first = views[0]
    region_cls = type(first)
    head = first._t
    for v in views[1:]:
        if type(v) is not region_cls:
            raise TypeError(
                f"stack: heterogeneous region views {region_cls.__name__} vs {type(v).__name__}"
            )
        t = v._t
        if t.batch_ndim != head.batch_ndim or t.sub_batch_ndim != head.sub_batch_ndim:
            raise ValueError(
                f"stack: mismatched region ndims "
                f"({head.batch_ndim, head.sub_batch_ndim}) vs "
                f"({t.batch_ndim, t.sub_batch_ndim})"
            )
        if t.data.shape != head.data.shape:
            raise ValueError(
                f"stack: mismatched data shapes {tuple(head.data.shape)} vs {tuple(t.data.shape)}"
            )
    axis = first._resolve_insert_dim(dim)
    new_data = torch.stack([v._t.data for v in views], dim=axis)
    ba, sb = first._new_ndims(axes_added=1)
    # When stacking onto sub_batch, insert an anonymous label entry at
    # the new axis position; other region stacks leave labels unchanged.
    return Tensor(new_data, ba, sb)


# Register Tensor as a pytree dataclass leaf. Following the same convention
# as the typed wrappers (`neml2/types/_pytree.py`), the integer shape-metadata
# fields are dropped from the pytree -- they reset to the dataclass defaults
# on unflatten. Callers that need to round-trip the metadata across a pytree
# boundary reconstruct it from a known source (the AxisLayout, or the
# ``_meta.json`` AOTI manifest); the exported graph operates on raw tensors
# whose shapes already encode the batch / sub-batch / base split.
_torch_pytree.register_dataclass(
    Tensor,
    field_names=["data"],
    drop_field_names=[
        "batch_ndim",
        "sub_batch_ndim",
        "sub_batch_state",
        "sub_batch_meta",
        "k_ndim",
        "k_state",
        "k_pairing",
    ],
    serialized_type_name="neml2.types.Tensor",
)
