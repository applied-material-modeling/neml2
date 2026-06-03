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

"""Free functions on the typed tensor wrappers.

Mirrors the C++ ``include/neml2/tensors/functions/`` surface, collapsed into a
single Python module (the one-file-per-op layout on the C++ side exists for
compilation parallelization, not for organization).

Operator-like behavior (``+``, ``-``, ``*``, ``@``, ``-x``, ``abs(x)``,
``x ** n``) lives on the wrapper classes themselves; everything else
(invariants, decompositions, transcendentals, products that produce a
different output type) lives here.
"""

from __future__ import annotations

import math
from typing import TypeVar, overload

import torch

from neml2.types._base import (
    DynamicBatchView,
    SubBatchView,
    TensorWrapper,
    align_scalar_base,
    align_sub_batch,
)
from neml2.types.miller_index import MillerIndex
from neml2.types.r2 import R2
from neml2.types.rot import Rot
from neml2.types.scalar import Scalar
from neml2.types.sr2 import SR2
from neml2.types.ssr4 import SSR4
from neml2.types.vec import Vec
from neml2.types.wr2 import WR2

_TW = TypeVar("_TW", bound=TensorWrapper)

# ---- SR2 invariants / decompositions ----


def tr(A: SR2) -> Scalar:
    """Trace of a symmetric rank-2 tensor.

    Matches ``neml2::tr(const SR2&)`` in ``src/neml2/tensors/functions/tr.cxx``.
    """
    return Scalar(
        A.data[..., 0] + A.data[..., 1] + A.data[..., 2],
        sub_batch_ndim=A.sub_batch_ndim,
    )


def vol(A: SR2) -> SR2:
    """Volumetric part of a symmetric rank-2 tensor: ``tr(A)/3 * I``.

    Matches ``neml2::vol(const SR2&)`` in ``src/neml2/tensors/functions/vol.cxx``.
    """
    I = SR2.identity(dtype=A.dtype, device=A.device).data
    return SR2(align_scalar_base(tr(A).data / 3.0, 1) * I, sub_batch_ndim=A.sub_batch_ndim)


def dev(A: SR2) -> SR2:
    """Deviatoric part of a symmetric rank-2 tensor: ``A - vol(A)``.

    Matches ``neml2::dev(const SR2&)`` in ``src/neml2/tensors/functions/dev.cxx``.
    """
    return A - vol(A)


def norm(A: SR2, eps: float = 0.0) -> Scalar:
    """Frobenius norm of a symmetric rank-2 tensor in Mandel packing.

    The ``eps`` regularizer keeps the result differentiable at ``A == 0``;
    matches ``neml2::norm(const Tensor&, std::optional<CScalar>)`` in
    ``src/neml2/tensors/functions/norm.cxx``.
    """
    sq = (A.data * A.data).sum(dim=-1)
    return Scalar(torch.sqrt(sq + eps * eps), sub_batch_ndim=A.sub_batch_ndim)


def unit(A: SR2, eps: float = 0.0) -> SR2:
    """Normalize $A$ by its Frobenius norm. ``eps`` regularizes at ``A == 0``."""
    n = align_scalar_base(norm(A, eps).data, 1)
    return SR2(A.data / n, sub_batch_ndim=A.sub_batch_ndim)


# ---- Scalar transcendentals ----


def _normalize_dim(dim: int, start: int, end: int) -> int:
    delta = end - start
    if dim < -delta or dim >= delta:
        raise IndexError(f"dim {dim} out of range [{-delta}, {delta})")
    return start + dim if dim >= 0 else end + dim


def _normalize_dims(dims: int | list[int] | tuple[int, ...], start: int, end: int) -> list[int]:
    if isinstance(dims, int):
        dims = (dims,)
    return sorted((_normalize_dim(d, start, end) for d in dims), reverse=True)


# ---- region-reduction free functions ----
#
# Each takes a region view (``t.dynamic_batch`` or ``t.sub_batch``) and
# reduces over axes in that region. Dispatch is via isinstance; the view
# carries everything needed to translate the region-relative ``dim`` to
# an absolute axis and to compute the right ``sub_batch_ndim`` on the
# result. The view's generic wrapper-type parameter ``_WT`` is threaded
# through so ``sum(t.sub_batch, ...)`` returns ``type(t)`` precisely —
# no ``cast`` needed at call sites.
#
# ``t.batch`` is rejected because it would silently straddle the
# dynamic/sub-batch split; ``t.base`` is rejected because reducing base
# axes would change the wrapper type.


def _reduce_view_bounds(
    view: DynamicBatchView[_TW] | SubBatchView[_TW], op_name: str
) -> tuple[_TW, int, int]:
    """Validate ``view`` and return ``(wrapper, region_start, region_end)`` on success."""
    if isinstance(view, DynamicBatchView):
        w = view._w
        return w, 0, len(w.dynamic_batch_shape)
    if isinstance(view, SubBatchView):
        w = view._w
        start = len(w.dynamic_batch_shape)
        return w, start, start + w.sub_batch_ndim
    raise TypeError(
        f"{op_name} expects t.dynamic_batch or t.sub_batch view; got {type(view).__name__}. "
        "(base reductions change wrapper type; batch reductions would straddle the "
        "dynamic/sub-batch split.)"
    )


def sum(  # noqa: A001 — intentionally shadows builtin, callers import explicitly
    view: DynamicBatchView[_TW] | SubBatchView[_TW],
    dims: int | list[int] | tuple[int, ...] = 0,
    keepdim: bool = False,
) -> _TW:
    """Sum over axes of a region view.

    ``view`` must be ``t.dynamic_batch`` or ``t.sub_batch``. When
    summing over a sub-batch axis with ``keepdim=False``, the result's
    ``sub_batch_ndim`` drops by the number of reduced axes. Returns
    the same wrapper type as the view's underlying wrapper.
    """
    w, start, end = _reduce_view_bounds(view, "sum")
    dn = _normalize_dims(dims, start, end)
    data = torch.sum(w.data, dim=dn, keepdim=keepdim)
    new_sb = w.sub_batch_ndim
    if isinstance(view, SubBatchView) and not keepdim:
        new_sb -= len(dn)
    return w._rewrap(data, sub_batch_ndim=new_sb)


def mean(view: DynamicBatchView[_TW] | SubBatchView[_TW], dim: int = 0) -> _TW:
    """Mean over one axis of a region view.

    ``view`` must be ``t.dynamic_batch`` or ``t.sub_batch``. Always
    collapses the axis (no ``keepdim``); reducing a sub-batch axis drops
    ``sub_batch_ndim`` by 1. Returns the same wrapper type as the view's
    underlying wrapper.
    """
    w, start, end = _reduce_view_bounds(view, "mean")
    d = _normalize_dim(dim, start, end)
    new_sb = w.sub_batch_ndim - (1 if isinstance(view, SubBatchView) else 0)
    return w._rewrap(torch.mean(w.data, dim=d), sub_batch_ndim=new_sb)


def diff(view: DynamicBatchView[_TW] | SubBatchView[_TW], n: int = 1, dim: int = 0) -> _TW:
    """``n``-th order finite difference along an axis of a region view.

    The selected axis length shrinks by ``n``; the wrapper type and
    ``sub_batch_ndim`` are preserved (``torch.diff`` does not collapse
    the axis, so this is *not* a reduction). ``view`` must be
    ``t.dynamic_batch`` or ``t.sub_batch``.

    Matches ``neml2::intmd_diff`` in ``src/neml2/tensors/functions/diff.cxx``.
    """
    if n < 0:
        raise ValueError(f"diff order n must be non-negative, got {n}")
    w, start, end = _reduce_view_bounds(view, "diff")
    d = _normalize_dim(dim, start, end)
    return w._rewrap(torch.diff(w.data, n=n, dim=d), sub_batch_ndim=w.sub_batch_ndim)


def sub_batch_zeros_like(
    template: _TW, *, size: int, dim: int = 0, sub_batch_ndim: int | None = None
) -> _TW:
    """Build a zero wrapper with a new sub-batch axis of the given ``size``.

    Returned wrapper has the same dtype/device/base-shape as ``template`` and
    ``template.dynamic_batch_shape`` as its dynamic batch; the new sub-batch
    axis is inserted at sub-batch position ``dim`` (default leading) of
    ``template``'s sub-batch region, giving $sub_batch_ndim = template.sub_batch_ndim + 1$.
    Pass ``sub_batch_ndim`` explicitly to override that default (e.g. to
    drop the template's own sub-batch axes when only the dynamic batch is
    desired).

    This is the typed counterpart of the C++ ``Scalar::zeros_like(tail)``
    pattern: build a zero tail of a given cell-axis length without dropping
    out of wrapper algebra (no raw ``torch.zeros`` against ``template.data``
    inside a leaf).
    """
    if size <= 0:
        raise ValueError(f"sub_batch_zeros_like size must be positive, got {size}")
    if sub_batch_ndim is None:
        sub_batch_ndim = template.sub_batch_ndim + 1
    # Build shape: dynamic_batch + (size at dim within sub-batch) + base
    dyn = template.dynamic_batch_shape
    base = template.BASE_SHAPE
    # Insert ``size`` at sub-batch position ``dim``; other sub-batch dims of
    # ``template`` are dropped (they're typically what's being replaced).
    shape = (*dyn, size, *base)
    data = torch.zeros(shape, dtype=template.dtype, device=template.device)
    return type(template)(data, sub_batch_ndim=sub_batch_ndim)


def abs(a: _TW) -> _TW:  # noqa: A001 - mirrors neml2::abs
    """Element-wise absolute value."""
    return a._rewrap(torch.abs(a.data), sub_batch_ndim=a.sub_batch_ndim)


def sign(a: _TW) -> _TW:
    """Element-wise sign."""
    return a._rewrap(torch.sign(a.data), sub_batch_ndim=a.sub_batch_ndim)


def heaviside(a: _TW) -> _TW:
    """Element-wise Heaviside step function $H(a) = (sign(a) + 1) / 2$.

    Matches ``neml2::heaviside`` in
    ``src/neml2/tensors/functions/heaviside.cxx`` (which uses the same
    ``(sign(a) + 1) / 2`` form), preserving wrapper type and ``sub_batch_ndim``.
    """
    return a._rewrap((torch.sign(a.data) + 1.0) / 2.0, sub_batch_ndim=a.sub_batch_ndim)


def macaulay(a: _TW) -> _TW:
    """Element-wise Macaulay bracket $<a>_+ = a * H(a) = a * (sign(a) + 1) / 2$.

    Matches ``neml2::macaulay`` in
    ``src/neml2/tensors/functions/macaulay.cxx``; preserves wrapper type and
    ``sub_batch_ndim``.
    """
    return a._rewrap(a.data * (torch.sign(a.data) + 1.0) / 2.0, sub_batch_ndim=a.sub_batch_ndim)


def clamp(
    a: _TW,
    lo: float | int | None = None,
    hi: float | int | None = None,
) -> _TW:
    """Element-wise clamp, matching ``neml2::clamp``.

    Either bound may be ``None`` to leave that side unbounded. Bounds are
    plain scalars (the C++ ``clamp`` overload used by leaves takes scalar
    endpoints); preserves wrapper type and ``sub_batch_ndim``.
    """
    return a._rewrap(torch.clamp(a.data, min=lo, max=hi), sub_batch_ndim=a.sub_batch_ndim)


def _as_scalar(value: float | int, like: TensorWrapper) -> Scalar:
    return Scalar(torch.as_tensor(value, dtype=like.dtype, device=like.device))


def _logical_binary(a: _TW, b: TensorWrapper | float | int, op) -> _TW:
    if isinstance(b, (float, int)):
        b = _as_scalar(b, a)
    if type(a) is not type(b) and not isinstance(b, Scalar):
        raise TypeError(
            f"logical operation requires matching wrappers or Scalar, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    [aa, bb], sb = align_sub_batch(a, b)
    b_data = bb.data
    if isinstance(bb, Scalar) and not isinstance(aa, Scalar):
        for _ in range(a.BASE_NDIM):
            b_data = b_data.unsqueeze(-1)
    return a._rewrap(op(aa.data, b_data), sub_batch_ndim=sb)


def gt(a: _TW, b: TensorWrapper | float | int) -> _TW:
    return _logical_binary(a, b, torch.gt)


def lt(a: _TW, b: TensorWrapper | float | int) -> _TW:
    return _logical_binary(a, b, torch.lt)


def where(c: TensorWrapper, a: _TW, b: _TW) -> _TW:
    """Element-wise select, matching ``neml2::where``."""
    if type(a) is not type(b):
        raise TypeError(
            f"where requires matching input wrapper types, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    [cc, aa, bb], sb = align_sub_batch(c, a, b)
    c_data = cc.data
    if isinstance(cc, Scalar) and not isinstance(aa, Scalar):
        for _ in range(a.BASE_NDIM):
            c_data = c_data.unsqueeze(-1)
    return a._rewrap(torch.where(c_data, aa.data, bb.data), sub_batch_ndim=sb)


def linear_interpolation(argument: Scalar, abscissa: Scalar, ordinate: Scalar) -> Scalar:
    """Piecewise-linear interpolation of a Scalar table."""
    x = argument.data
    X = abscissa.data
    Y = ordinate.data
    n = X.shape[-1]
    idx = torch.searchsorted(X, x, right=True).clamp(1, n - 1)
    x1 = X[..., idx - 1]
    x2 = X[..., idx]
    y1 = Y[..., idx - 1]
    y2 = Y[..., idx]
    slope = (y2 - y1) / (x2 - x1)
    return Scalar(y1 + slope * (x - x1), sub_batch_ndim=argument.sub_batch_ndim)


def jvp_linear_interpolation(
    argument: Scalar, abscissa: Scalar, ordinate: Scalar, dargument: Scalar
) -> Scalar:
    """Differential pushforward of :func:`linear_interpolation` along ``dargument``.

    Returns ``slope · dargument`` where ``slope`` is the piecewise-constant
    ``dy/dx`` at ``argument``. The leading-K seed axis of a chain-rule
    tangent rides through naturally as a broadcast batch dim on
    ``dargument``. Hides the ``searchsorted`` + gather behind a typed-
    function boundary so leaves stay in pure typed-wrapper algebra — same
    pattern as :func:`jvp_compose`, :func:`jvp_exp_map`, etc.
    """
    x = argument.data
    X = abscissa.data
    Y = ordinate.data
    n = X.shape[-1]
    idx = torch.searchsorted(X, x, right=True).clamp(1, n - 1)
    x1 = X[..., idx - 1]
    x2 = X[..., idx]
    y1 = Y[..., idx - 1]
    y2 = Y[..., idx]
    slope = (y2 - y1) / (x2 - x1)
    sb = max(argument.sub_batch_ndim, dargument.sub_batch_ndim)
    return Scalar(slope * dargument.data, sub_batch_ndim=sb)


def _bilinear_corners(
    arg1: Scalar,
    arg2: Scalar,
    abscissa1: Scalar,
    abscissa2: Scalar,
    ordinate: _TW,
) -> tuple[_TW, _TW, _TW, _TW, Scalar, Scalar, Scalar, Scalar]:
    """Shared corner / parametric-coordinate setup for bilinear primitives.

    Returns the four corner values ``(Y00, Y01, Y10, Y11)`` of the cell that
    contains ``(arg1, arg2)`` (each shaped like ``arg1.data + ordinate.base``),
    plus the parametric coordinates ``(xi, eta)`` and per-cell axis widths
    ``(dX1, dX2)``. The corners broadcast against any combination of batched
    abscissa / ordinate / argument shapes (mirroring the C++ ``apply_mask``
    machinery with a plain ``torch.gather``-based extraction).

    ``abscissa1.data`` has shape ``(*dyn_X, N1)``;
    ``abscissa2.data`` has shape ``(*dyn_X, N2)``;
    ``ordinate.data`` has shape ``(*dyn_Y, N1, N2, *base)``;
    ``arg1.data`` and ``arg2.data`` have shape ``(*dyn_arg)``.
    All ``*dyn_*`` regions must be torch-broadcastable.
    """
    base_ndim = ordinate.BASE_NDIM
    Y = ordinate.data
    X1 = abscissa1.data
    X2 = abscissa2.data
    x1 = arg1.data
    x2 = arg2.data
    N1 = X1.shape[-1]
    N2 = X2.shape[-1]

    # Broadcast the abscissa / ordinate / argument dynamic shapes together so a
    # single per-row gather works uniformly. The ``-1`` / ``-2-base_ndim`` axes
    # are the table axes — we strip them off, broadcast the remainder, then put
    # them back via ``expand``.
    dyn_X1 = X1.shape[:-1]
    dyn_X2 = X2.shape[:-1]
    dyn_Y = Y.shape[: Y.ndim - 2 - base_ndim]
    base_shape = Y.shape[Y.ndim - base_ndim :] if base_ndim > 0 else torch.Size(())
    dyn_arg = torch.broadcast_shapes(x1.shape, x2.shape)
    dyn = torch.broadcast_shapes(dyn_X1, dyn_X2, dyn_Y, dyn_arg)

    X1b = X1.expand(*dyn, N1)
    X2b = X2.expand(*dyn, N2)
    Yb = Y.expand(*dyn, N1, N2, *base_shape)
    x1b = x1.expand(dyn)
    x2b = x2.expand(dyn)

    # Locate cell index along each axis (clamped to [1, Nk-1] so a query at the
    # extreme upper edge extrapolates from the last segment instead of falling
    # off — matches the C++ mask convention ``x1 > X10 && x1 <= X11``).
    idx1 = torch.searchsorted(X1b, x1b.unsqueeze(-1), right=True).squeeze(-1).clamp(1, N1 - 1)
    idx2 = torch.searchsorted(X2b, x2b.unsqueeze(-1), right=True).squeeze(-1).clamp(1, N2 - 1)

    def _gather_X(X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(X, -1, idx.unsqueeze(-1)).squeeze(-1)

    X10 = _gather_X(X1b, idx1 - 1)
    X11 = _gather_X(X1b, idx1)
    X20 = _gather_X(X2b, idx2 - 1)
    X21 = _gather_X(X2b, idx2)

    # Gather the four corner ordinate slabs (..., *base).
    def _gather_Y(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        # Insert N1, N2, base singleton dims so gather operates per-row.
        # i1, i2 share ``dyn``; we expand them across N2 / base / N1 as needed.
        i1_idx = i1.reshape(*dyn, *([1] * (2 + base_ndim))).expand(*dyn, 1, N2, *base_shape)
        Y_row = torch.gather(Yb, -2 - base_ndim, i1_idx).squeeze(-2 - base_ndim)
        i2_idx = i2.reshape(*dyn, *([1] * (1 + base_ndim))).expand(*dyn, 1, *base_shape)
        return torch.gather(Y_row, -1 - base_ndim, i2_idx).squeeze(-1 - base_ndim)

    Y00_data = _gather_Y(idx1 - 1, idx2 - 1)
    Y01_data = _gather_Y(idx1 - 1, idx2)
    Y10_data = _gather_Y(idx1, idx2 - 1)
    Y11_data = _gather_Y(idx1, idx2)

    Y00 = ordinate._rewrap(Y00_data, sub_batch_ndim=0)
    Y01 = ordinate._rewrap(Y01_data, sub_batch_ndim=0)
    Y10 = ordinate._rewrap(Y10_data, sub_batch_ndim=0)
    Y11 = ordinate._rewrap(Y11_data, sub_batch_ndim=0)

    dX1 = X11 - X10
    dX2 = X21 - X20
    xi_data = (x1b - X10) / dX1
    eta_data = (x2b - X20) / dX2
    return (
        Y00,
        Y01,
        Y10,
        Y11,
        Scalar(xi_data),
        Scalar(eta_data),
        Scalar(dX1),
        Scalar(dX2),
    )


def bilinear_interpolation(
    arg1: Scalar,
    arg2: Scalar,
    abscissa1: Scalar,
    abscissa2: Scalar,
    ordinate: _TW,
) -> _TW:
    """Bilinear interpolation of ``ordinate`` over a 2-D rectilinear grid.

    Mirrors the C++ ``BilinearInterpolation::set_value`` evaluator with the
    fixed ``_dim=0`` convention used by every in-tree test fixture. Accepts
    any ``ordinate`` typed wrapper (Scalar, Vec, SR2) shaped
    ``(..., N1, N2, *base)`` with ``sub_batch_ndim=2``; the abscissae carry
    ``sub_batch_ndim=1`` and the arguments carry ``sub_batch_ndim=0``. The
    returned wrapper has ``sub_batch_ndim=0`` and shape ``(..., *base)``.
    """
    Y00, Y01, Y10, Y11, xi, eta, _dX1, _dX2 = _bilinear_corners(
        arg1, arg2, abscissa1, abscissa2, ordinate
    )
    c1 = Y10 - Y00
    c2 = Y01 - Y00
    c3 = Y11 - Y10 - Y01 + Y00
    return Y00 + c1 * xi + c2 * eta + c3 * (xi * eta)


def bilinear_interpolation_slopes(
    arg1: Scalar,
    arg2: Scalar,
    abscissa1: Scalar,
    abscissa2: Scalar,
    ordinate: _TW,
) -> tuple[_TW, _TW]:
    """Return ``(dP/dx1, dP/dx2)`` at the query point, one wrapper per axis.

    The bilinear cell evaluates as
    $P = Y00 + (Y10-Y00) xi + (Y01-Y00) eta + (Y11-Y10-Y01+Y00) xi eta$ where
    $xi = (x1-X10)/(X11-X10)$, $eta = (x2-X20)/(X21-X20)$. Differentiating
    once in each argument gives the two slope wrappers returned here, sized
    like the ordinate's base.
    """
    Y00, Y01, Y10, Y11, xi, eta, dX1, dX2 = _bilinear_corners(
        arg1, arg2, abscissa1, abscissa2, ordinate
    )
    c1 = Y10 - Y00
    c2 = Y01 - Y00
    c3 = Y11 - Y10 - Y01 + Y00
    dP_dx1 = (c1 + c3 * eta) * (1.0 / dX1)
    dP_dx2 = (c2 + c3 * xi) * (1.0 / dX2)
    return dP_dx1, dP_dx2


def sqrt(s: Scalar) -> Scalar:
    return Scalar(torch.sqrt(s.data), sub_batch_ndim=s.sub_batch_ndim)


def exp(s: Scalar) -> Scalar:
    return Scalar(torch.exp(s.data), sub_batch_ndim=s.sub_batch_ndim)


def tanh(s: Scalar) -> Scalar:
    """Hyperbolic tangent. Matches ``neml2::tanh(const Scalar&)``."""
    return Scalar(torch.tanh(s.data), sub_batch_ndim=s.sub_batch_ndim)


def cosh(s: Scalar) -> Scalar:
    """Hyperbolic cosine. Matches ``neml2::cosh(const Scalar&)``."""
    return Scalar(torch.cosh(s.data), sub_batch_ndim=s.sub_batch_ndim)


def sinh(s: Scalar) -> Scalar:
    """Hyperbolic sine. Matches ``neml2::sinh(const Scalar&)``."""
    return Scalar(torch.sinh(s.data), sub_batch_ndim=s.sub_batch_ndim)


def log(s: Scalar) -> Scalar:
    """Natural logarithm. Matches ``neml2::log(const Scalar&)``."""
    return Scalar(torch.log(s.data), sub_batch_ndim=s.sub_batch_ndim)


def log10(s: Scalar) -> Scalar:
    """Base-10 logarithm. Matches ``neml2::log10(const Scalar&)``."""
    return Scalar(torch.log10(s.data), sub_batch_ndim=s.sub_batch_ndim)


def pow(a: _TW, n: float | int | Scalar) -> _TW:  # noqa: A001
    """Element-wise power."""
    if isinstance(n, Scalar):
        [aa, nn], sb = align_sub_batch(a, n)
        n_data = nn.data
        for _ in range(a.BASE_NDIM):
            n_data = n_data.unsqueeze(-1)
        return a._rewrap(torch.pow(aa.data, n_data), sub_batch_ndim=sb)
    return a._rewrap(torch.pow(a.data, n), sub_batch_ndim=a.sub_batch_ndim)


# ---- Cross-type products on SR2 ----


def outer(a: SR2, b: SR2) -> SSR4:
    """Tensor product ``a ⊗ b`` of two SR2s, producing an SSR4 in Mandel packing."""
    [aa, bb], sb = align_sub_batch(a, b)
    return SSR4(aa.data.unsqueeze(-1) * bb.data.unsqueeze(-2), sub_batch_ndim=sb)


def inner(A: TensorWrapper, B: TensorWrapper) -> Scalar:
    """Frobenius inner product ``A : B`` over the wrappers' base axes.

    Mirrors ``neml2::inner(const Tensor&, const Tensor&)`` in
    ``src/neml2/tensors/functions/inner.cxx``: contract every base component
    of *A* against the matching base component of *B*, leaving a Scalar over
    the shared batch + sub-batch axes.

    For ``SR2`` the Mandel packing's per-component weights are already absorbed
    into the storage (off-diagonal entries carry the ``sqrt(2)`` factor), so
    the contraction collapses to a plain dot product over the 6-component
    vector and naturally agrees with the full-tensor Frobenius product. For
    ``R2`` (or any non-symmetric wrapper) the contraction sums over all base
    axes directly.
    """
    if type(A) is not type(B):
        raise TypeError(
            f"inner requires matching wrapper types; got {type(A).__name__} and {type(B).__name__}"
        )
    if A.BASE_SHAPE != B.BASE_SHAPE:
        raise ValueError(
            f"inner requires matching base shape; got {A.BASE_SHAPE} and {B.BASE_SHAPE}"
        )
    [aa, bb], sb = align_sub_batch(A, B)
    base_dims = tuple(range(-A.BASE_NDIM, 0)) if A.BASE_NDIM > 0 else ()
    prod = aa.data * bb.data
    if base_dims:
        out = prod.sum(dim=base_dims)
    else:
        out = prod
    return Scalar(out, sub_batch_ndim=sb)


# ---- Second-order tensor determinant / inverse ----
#
# Mirror ``neml2::det`` / ``neml2::inv`` (see
# ``src/neml2/tensors/functions/det.cxx`` and ``inv.cxx``). The wrapper-aware
# overloads accept ``R2`` or ``SR2`` and return ``Scalar`` / the same wrapper
# type respectively. The C++ side uses the explicit 3x3 cofactor expansion;
# ``torch.linalg.det`` / ``torch.linalg.inv`` give the same result and broadcast
# cleanly over arbitrary leading batch + sub-batch axes, which is what we want.


def det(A: TensorWrapper) -> Scalar:
    """Determinant of a (..., 3, 3) second-order tensor wrapper.

    Accepts ``R2`` (full 3x3) or ``SR2`` (symmetric, Mandel-packed); the SR2
    overload converts to its full 3x3 form first. Returns a ``Scalar`` over
    the wrapper's batch + sub-batch axes. Mirrors ``neml2::det``.
    """
    if isinstance(A, R2):
        full = A.data
    elif isinstance(A, SR2):
        full = r2_from_sr2(A).data
    else:
        raise TypeError(f"det requires R2 or SR2; got {type(A).__name__}")
    return Scalar(torch.linalg.det(full), sub_batch_ndim=A.sub_batch_ndim)


def inv(A: _TW) -> _TW:
    """Matrix inverse of a (..., 3, 3) second-order tensor wrapper.

    For ``R2`` returns the full inverse as an ``R2``; for ``SR2`` returns the
    inverse repacked into Mandel form (the inverse of a symmetric tensor is
    symmetric). Mirrors ``neml2::inv``.
    """
    if isinstance(A, R2):
        return R2(torch.linalg.inv(A.data), sub_batch_ndim=A.sub_batch_ndim)  # type: ignore[return-value]
    if isinstance(A, SR2):
        full = r2_from_sr2(A).data
        inv_full = torch.linalg.inv(full)
        # The inverse of a symmetric tensor is symmetric; ``sym(R2(...))``
        # both repacks to Mandel and absorbs any tiny numerical asymmetry.
        return sym(R2(inv_full, sub_batch_ndim=A.sub_batch_ndim))  # type: ignore[return-value]
    raise TypeError(f"inv requires R2 or SR2; got {type(A).__name__}")


# ---- R2 ↔ SR2 / WR2 (Mandel / skew representation changes) ----
#
# These are mathematical mappings between distinct typed tensors — not pure
# shape reshapes — so they live as free functions per the project philosophy.
# Packing follows the C++ side (Mandel: off-diagonals scaled by sqrt(2);
# skew axial: ``W = [[0,-w2,w1],[w2,0,-w0],[-w1,w0,0]]``).

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_SQRT2 = math.sqrt(2.0)


# NOTE on Mandel / skew pack/unpack vectorization: a "matmul against fixed
# (9, 6) projection" form was tried for ``mandel_pack_sym3`` /
# ``skew_pack_axial3`` / ``r2_from_sr2`` / ``r2_from_wr2`` / ``sym`` /
# ``skew``. It improved CPU wall-time by ~10 % but regressed CUDA wall-time
# by ~9 % at small batch (B ≤ 1,024): each matmul launches a separate
# kernel that Inductor cannot fuse with surrounding pointwise ops, while
# the explicit select+stack chains do fuse into a single Triton kernel
# via Inductor's pointwise scheduler. Sticking with the pointwise form
# is the CUDA-favourable choice. A hand-rolled ``unrolled_matmul`` that
# emits fusible pointwise ops (instead of dispatching to BLAS / Triton's
# matmul kernel) is a candidate future optimisation that could unify the
# CPU and CUDA wins — see the note on bench results in
# ``python/neml2/native/README.md``.


def r2_from_sr2(s: SR2) -> R2:
    """Unpack an SR2 (Mandel) into a full ``R2`` ``(..., 3, 3)``.

    Matches the C++ ``R2(const SR2&)`` constructor / ``mandel_to_full``.

    NOTE: an earlier attempt rewrote this as ``s.data @ P_unpack`` (a
    ``(6, 9)`` projection matmul). The op count dropped from ~10 to 2,
    but the change **regressed CUDA wall-time by ~9 %** at small batch
    because Inductor doesn't fuse a separate matmul kernel with the
    surrounding pointwise ops, while it does fuse the explicit
    select+stack chain below into a single Triton kernel. The pointwise
    form is the deliberate CUDA-favourable choice — see the note in the
    README under the CP single-crystal bench.
    """
    d = s.data
    a, b, c = d[..., 0], d[..., 1], d[..., 2]
    yz = d[..., 3] * _INV_SQRT2
    xz = d[..., 4] * _INV_SQRT2
    xy = d[..., 5] * _INV_SQRT2
    row0 = torch.stack([a, xy, xz], dim=-1)
    row1 = torch.stack([xy, b, yz], dim=-1)
    row2 = torch.stack([xz, yz, c], dim=-1)
    return R2(torch.stack([row0, row1, row2], dim=-2), sub_batch_ndim=s.sub_batch_ndim)


def r2_from_wr2(w: WR2) -> R2:
    """Unpack a ``WR2`` axial vector into a full skew-symmetric ``R2``.

    Matches ``R2(const WR2&)`` / ``skew_to_full`` — the convention is the
    same as ``R2::skew(Vec)``: ``W = [[0,-w2,w1],[w2,0,-w0],[-w1,w0,0]]``.
    See :func:`r2_from_sr2` for why the explicit select+stack chain is
    preferred over a matmul-against-fixed-projection form (CUDA fusion).
    """
    d = w.data
    w0, w1, w2 = d[..., 0], d[..., 1], d[..., 2]
    z = torch.zeros_like(w0)
    row0 = torch.stack([z, -w2, w1], dim=-1)
    row1 = torch.stack([w2, z, -w0], dim=-1)
    row2 = torch.stack([-w1, w0, z], dim=-1)
    return R2(torch.stack([row0, row1, row2], dim=-2), sub_batch_ndim=w.sub_batch_ndim)


def sym(t: R2) -> SR2:
    """Symmetric part of an ``R2`` packed in Mandel form.

    Equivalent to $full_to_mandel((T + T^T) / 2)$: diagonals are kept as-is,
    off-diagonals carry the symmetric average times ``sqrt(2)``.
    """
    m = t.data
    a = m[..., 0, 0]
    b = m[..., 1, 1]
    c = m[..., 2, 2]
    yz = (m[..., 1, 2] + m[..., 2, 1]) * (_SQRT2 / 2.0)
    xz = (m[..., 0, 2] + m[..., 2, 0]) * (_SQRT2 / 2.0)
    xy = (m[..., 0, 1] + m[..., 1, 0]) * (_SQRT2 / 2.0)
    return SR2(torch.stack([a, b, c, yz, xz, xy], dim=-1), sub_batch_ndim=t.sub_batch_ndim)


def skew(t: R2) -> WR2:
    """Skew part of an ``R2`` packed as an axial vector.

    $w0 = (m[2,1] - m[1,2]) / 2$, $w1 = (m[0,2] - m[2,0]) / 2$,
    $w2 = (m[1,0] - m[0,1]) / 2$ (matches the ``W = [[0,-w2,w1],
    [w2,0,-w0],[-w1,w0,0]]`` convention).
    """
    m = t.data
    w0 = (m[..., 2, 1] - m[..., 1, 2]) / 2.0
    w1 = (m[..., 0, 2] - m[..., 2, 0]) / 2.0
    w2 = (m[..., 1, 0] - m[..., 0, 1]) / 2.0
    return WR2(torch.stack([w0, w1, w2], dim=-1), sub_batch_ndim=t.sub_batch_ndim)


# ---- Rot ↔ R2 (Euler-Rodrigues mapping) ----


def euler_rodrigues(r: Rot) -> R2:
    """Convert an MRP rotation to its 3x3 rotation matrix.

    Mirrors ``Rot::euler_rodrigues`` in ``src/neml2/tensors/Rot.cxx``::

        R = (1+rr)^-2 * ( (1+rr)^2 * I + 4(1-rr) W + 8 W^2 )

    where $W$ is the skew-symmetric matrix of $r$ (via ``R2::skew``)
    and $rr = ||r||^2$.
    """
    rr = (r.data * r.data).sum(dim=-1)  # (...,)
    W = r2_from_wr2(WR2(r.data, sub_batch_ndim=r.sub_batch_ndim)).data  # (...,3,3)
    I = torch.eye(3, dtype=r.dtype, device=r.device)
    one_plus_rr = (1.0 + rr).unsqueeze(-1).unsqueeze(-1)
    one_minus_rr = (1.0 - rr).unsqueeze(-1).unsqueeze(-1)
    W2 = W @ W
    R_mat = (one_plus_rr * one_plus_rr * I + 4.0 * one_minus_rr * W + 8.0 * W2) / (
        one_plus_rr * one_plus_rr
    )
    return R2(R_mat, sub_batch_ndim=r.sub_batch_ndim)


def deuler_rodrigues(r: Rot) -> torch.Tensor:
    """Derivative ``∂R/∂r`` of the Euler-Rodrigues map; shape ``(..., 3, 3, 3)``.

    Returned as a raw tensor (no R3 wrapper exists in the Python-native stack
    yet — this derivative only flows through chain-rule actions, never
    surfaces as a Model input/output). Mirrors ``Rot::deuler_rodrigues`` in
    ``src/neml2/tensors/Rot.cxx``.
    """
    rr = (r.data * r.data).sum(dim=-1)  # (...,)
    W = r2_from_wr2(WR2(r.data, sub_batch_ndim=r.sub_batch_ndim)).data  # (...,3,3)
    W2 = W @ W
    # Levi-Civita symbol (constant 3x3x3).
    E = torch.zeros(3, 3, 3, dtype=r.dtype, device=r.device)
    E[0, 1, 2] = E[1, 2, 0] = E[2, 0, 1] = 1.0
    E[0, 2, 1] = E[2, 1, 0] = E[1, 0, 2] = -1.0

    one_plus_rr = 1.0 + rr  # (...,)
    inv2 = (1.0 / (one_plus_rr * one_plus_rr)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    inv3 = (
        (1.0 / (one_plus_rr * one_plus_rr * one_plus_rr)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    rm3 = (rr - 3.0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    om = (1.0 - rr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Each term: shape (..., 3, 3, 3) over last three axes (i, j, k).
    rk = r.data  # (..., 3) — broadcast over (3,3) front by unsqueezing -2,-3.
    W_ij = W  # (..., 3, 3)
    W2_ij = W2
    # ...ij,...k -> ...ijk
    term1 = 8.0 * rm3 * inv3 * (W_ij.unsqueeze(-1) * rk.unsqueeze(-2).unsqueeze(-2))
    term2 = -32.0 * inv3 * (W2_ij.unsqueeze(-1) * rk.unsqueeze(-2).unsqueeze(-2))
    # E has shape (3,3,3) with index order (k,i,j); we want (i,j,k) per the C++
    # ``R3::einsum("...kij->...ijk", {E})``.
    E_ijk = E.permute(1, 2, 0)
    term3 = -4.0 * om * inv2 * E_ijk
    # E_kim · W_mj -> (...,i,j,k) and W_im · E_kmj -> (...,i,j,k); both contract
    # over ``m``. Done as broadcast-multiply-sum over ``m`` (same idiom as
    # term1/term2 above; no einsum). The (...,3,3,3,3) intermediate is tiny and
    # fuses as a single pointwise kernel in Inductor — the CUDA-preferred form.
    #   EW[...,i,j,k] = sum_m E[k,i,m] * W[...,m,j]
    E_EW = E.permute(1, 0, 2).unsqueeze(1)  # (i, j=1, k, m)
    W_EW = W.transpose(-1, -2).unsqueeze(-3).unsqueeze(-2)  # (..., i=1, j, k=1, m)
    EW = (E_EW * W_EW).sum(dim=-1)  # (..., i, j, k)
    #   WE[...,i,j,k] = sum_m W[...,i,m] * E[k,m,j]
    W_WE = W.unsqueeze(-2).unsqueeze(-2)  # (..., i, j=1, k=1, m)
    E_WE = E.permute(2, 0, 1).unsqueeze(0)  # (i=1, j, k, m)
    WE = (W_WE * E_WE).sum(dim=-1)  # (..., i, j, k)
    term4 = -8.0 * inv2 * (EW + WE)

    return term1 + term2 + term3 + term4


# ---- Rot composition ----


def compose(r1: Rot, r2: Rot) -> Rot:
    """Compose two MRP rotations: ``r1 ∘ r2`` (apply r2 first, then r1).

    Matches ``operator*(const Rot&, const Rot&)`` in ``src/neml2/tensors/Rot.cxx``.
    The result is again an MRP; the formula handles the standard MRP
    composition with denominator $1 + ||r1||^2 ||r2||^2 - 2 r1·r2$.
    """
    [aa, bb], sb = align_sub_batch(r1, r2)
    a = aa.data
    b = bb.data
    rr1 = (a * a).sum(dim=-1, keepdim=True)
    rr2 = (b * b).sum(dim=-1, keepdim=True)
    dot = (a * b).sum(dim=-1, keepdim=True)
    cross_ba = _cross_raw(b, a)
    num = (1.0 - rr2) * a + (1.0 - rr1) * b - 2.0 * cross_ba
    den = 1.0 + rr1 * rr2 - 2.0 * dot
    return Rot(num / den, sub_batch_ndim=sb)


def _cross_raw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector cross product over the last axis of two ``(*, 3)`` raw tensors.

    Broadcasts the leading batch dims before dispatching to ``torch.cross``,
    which itself requires matching ndim. The leading-dim broadcast is the same
    rule typed-wrapper algebra uses elsewhere (``+`` / ``-`` / ``*`` go through
    PyTorch's right-aligned broadcast), so this keeps ``compose`` and similar
    callers honest when one input carries an extra dynamic-batch axis from a
    cross-group state coupling (e.g. global ``dt`` flowing into per-crystal
    orientation rate in a Taylor polycrystal).
    """
    if a.shape != b.shape:
        target = torch.broadcast_shapes(a.shape, b.shape)
        a = a.expand(target)
        b = b.expand(target)
    return torch.cross(a, b, dim=-1)


def drotate_self(r1: Rot, r2: Rot) -> R2:
    """``d(r2 ∘ r1) / d(r1)`` where the composition is ``r2 * r1``.

    Mirrors ``Rot::drotate_self`` in ``src/neml2/tensors/Rot.cxx``. The
    naming follows the C++ convention: $r1.rotate(r2) == r2 * r1$ (apply
    ``r1`` first then ``r2``), and ``r1.drotate_self(r2)`` is the derivative
    of that composed rotation w.r.t. the receiver ``r1``. Returns ``R2``.
    """
    [a1, a2], sb = align_sub_batch(r1, r2)
    rr1 = (a2.data * a2.data).sum(dim=-1, keepdim=False)  # note: matches C++ swap
    rr2 = (a1.data * a1.data).sum(dim=-1, keepdim=False)
    dot = (a2.data * a1.data).sum(dim=-1, keepdim=False)
    d = 1.0 + rr1 * rr2 - 2.0 * dot
    r3 = compose(a2, a1)  # rotated MRP — aligned inputs
    I = torch.eye(3, dtype=r1.dtype, device=r1.device)
    skew_r = r2_from_wr2(WR2(a2.data, sub_batch_ndim=sb)).data
    # outer(a, b) with a, b shape (...,3) -> (...,3,3) via unsqueeze broadcast
    a_vec = r3.data
    b_vec = 2.0 * rr1.unsqueeze(-1) * a1.data - 2.0 * a2.data
    term1 = -(a_vec.unsqueeze(-1) * b_vec.unsqueeze(-2))
    term2 = -2.0 * (a2.data.unsqueeze(-1) * a1.data.unsqueeze(-2))
    term3 = (1.0 - rr1).unsqueeze(-1).unsqueeze(-1) * I
    term4 = 2.0 * skew_r
    res = (term1 + term2 + term3 + term4) / d.unsqueeze(-1).unsqueeze(-1)
    return R2(res, sub_batch_ndim=sb)


def drotate(r1: Rot, r2: Rot) -> R2:
    """``d(r2 ∘ r1) / d(r2)`` where the composition is ``r2 * r1``.

    Mirrors ``Rot::drotate`` in ``src/neml2/tensors/Rot.cxx``. Returns ``R2``.
    """
    [a1, a2], sb = align_sub_batch(r1, r2)
    rr1 = (a1.data * a1.data).sum(dim=-1, keepdim=False)
    rr2 = (a2.data * a2.data).sum(dim=-1, keepdim=False)
    dot = (a1.data * a2.data).sum(dim=-1, keepdim=False)
    d = 1.0 + rr1 * rr2 - 2.0 * dot
    r3 = compose(a2, a1)
    I = torch.eye(3, dtype=r1.dtype, device=r1.device)
    skew_r1 = r2_from_wr2(WR2(a1.data, sub_batch_ndim=sb)).data
    a_vec = r3.data
    b_vec = 2.0 * rr1.unsqueeze(-1) * a2.data - 2.0 * a1.data
    term1 = -(a_vec.unsqueeze(-1) * b_vec.unsqueeze(-2))
    term2 = -2.0 * (a1.data.unsqueeze(-1) * a2.data.unsqueeze(-2))
    term3 = (1.0 - rr1).unsqueeze(-1).unsqueeze(-1) * I
    term4 = -2.0 * skew_r1
    res = (term1 + term2 + term3 + term4) / d.unsqueeze(-1).unsqueeze(-1)
    return R2(res, sub_batch_ndim=sb)


# ---- WR2 exponential map ----


def exp_map(w: WR2) -> Rot:
    """Exponential of a skew axial vector — yields an MRP rotation.

    Mirrors ``WR2::exp_map`` in ``src/neml2/tensors/WR2.cxx``. Uses a Taylor
    series near $||w||^2 ≈ 0$ to avoid the singularity at the origin; the
    other singularity at $||w||^2 = 2π$ is unavoidable and shared with the
    C++ implementation.
    """
    eps = torch.finfo(w.dtype).eps
    thresh = eps ** (1.0 / 3.0)
    norm2 = (w.data * w.data).sum(dim=-1)  # (...,)
    # Taylor near zero: r ≈ w * (1/4 + 5 ||w||^4 / 96)
    taylor_scale = 0.25 + 5.0 * norm2 * norm2 / 96.0
    res_taylor = w.data * taylor_scale.unsqueeze(-1)
    # Actual definition: r = w * tan(||w||^2/2) / (2 ||w||^2 cos(||w||^2/2))
    safe_norm2 = torch.where(norm2 > thresh, norm2, torch.ones_like(norm2))
    actual_scale = torch.tan(safe_norm2 / 2.0) / (2.0 * safe_norm2 * torch.cos(safe_norm2 / 2.0))
    res_actual = w.data * actual_scale.unsqueeze(-1)
    out = torch.where((norm2 > thresh).unsqueeze(-1), res_actual, res_taylor)
    return Rot(out, sub_batch_ndim=w.sub_batch_ndim)


def dexp_map(w: WR2) -> R2:
    """Derivative ``∂(exp_map(w))/∂w``, returned as a 3x3 ``R2``.

    Mirrors ``WR2::dexp_map`` in ``src/neml2/tensors/WR2.cxx``.
    """
    eps = torch.finfo(w.dtype).eps
    thresh = eps ** (1.0 / 3.0)
    norm2 = (w.data * w.data).sum(dim=-1)  # (...,)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    outer_ww = w.data.unsqueeze(-1) * w.data.unsqueeze(-2)  # ``"...i,...j->...ij"``

    # Taylor near zero.
    res_taylor = (5.0 * norm2 / 24.0).unsqueeze(-1).unsqueeze(-1) * outer_ww + (
        0.25 + 5.0 * norm2 * norm2 / 96.0
    ).unsqueeze(-1).unsqueeze(-1) * I

    # Actual formula.
    safe_norm2 = torch.where(norm2 > thresh, norm2, torch.ones_like(norm2))
    half = safe_norm2 / 2.0
    sec = 1.0 / torch.cos(half)
    f1 = torch.tan(half) / (2.0 * safe_norm2 * torch.cos(half))
    f2 = (safe_norm2 * sec**3 + torch.tan(half) * (safe_norm2 * torch.tan(half) - 2.0) * sec) / (
        2.0 * safe_norm2 * safe_norm2
    )
    res_actual = f1.unsqueeze(-1).unsqueeze(-1) * I + f2.unsqueeze(-1).unsqueeze(-1) * outer_ww

    sel = (norm2 > thresh).unsqueeze(-1).unsqueeze(-1)
    out = torch.where(sel, res_actual, res_taylor)
    return R2(out, sub_batch_ndim=w.sub_batch_ndim)


# ---- Rotation of SR2 / WR2 by an R2 (R * X * R^T projected back) ----
#
# Used in CP: every per-slip Schmid tensor (M : SR2, W : WR2) is rotated by
# the per-crystal orientation matrix R before being summed.


def _rotate_sym(s: SR2, R: R2) -> SR2:
    """$sym(R S R^T)$ packed back to Mandel; the symmetric tensor rotation."""
    [ss, rr], sb = align_sub_batch(s, R)
    S_full = r2_from_sr2(ss).data  # (...,3,3) — sub_batch already aligned with rr
    rotated = rr.data @ S_full @ rr.data.transpose(-2, -1)
    return sym(R2(rotated, sub_batch_ndim=sb))


def _rotate_skew(w: WR2, R: R2) -> WR2:
    """$skew(R W R^T)$ packed back to an axial vector."""
    [ww, rr], sb = align_sub_batch(w, R)
    W_full = r2_from_wr2(ww).data
    rotated = rr.data @ W_full @ rr.data.transpose(-2, -1)
    return skew(R2(rotated, sub_batch_ndim=sb))


def _jvp_rotate_sym(s: SR2, R: R2, dR: R2) -> SR2:
    """Pushforward of :func:`rotate` (SR2 overload) w.r.t. $R$ along ``dR``.

    ``rotate(s, R) = sym(R S Rᵀ)`` (linear in $s$, so the $s$-direction is
    just ``rotate(ds, R)`` and needs no primitive). The $R$-direction is the
    product rule ``sym(dR S Rᵀ + R S dRᵀ)``. ``dR`` is a leading-K ``R2``
    tangent; the 3×3 ``@`` / transpose broadcast $K$. All sub-batch alignment
    (e.g. per-crystal $R$ against a per-slip $s$) is handled by
    :func:`align_sub_batch`, exactly as in the forward.
    """
    [ss, RR, dRR], sb = align_sub_batch(s, R, dR)
    S = r2_from_sr2(ss).data
    Rm = RR.data
    dRm = dRR.data
    rotated = dRm @ S @ Rm.transpose(-2, -1) + Rm @ S @ dRm.transpose(-2, -1)
    return sym(R2(rotated, sub_batch_ndim=sb))


def _jvp_rotate_skew(w: WR2, R: R2, dR: R2) -> WR2:
    """Pushforward of :func:`rotate` (WR2 overload) w.r.t. $R$ along ``dR``.

    ``rotate(w, R) = skew(R W Rᵀ)`` (linear in $w$); the $R$-direction is
    ``skew(dR W Rᵀ + R W dRᵀ)``.
    """
    [ww, RR, dRR], sb = align_sub_batch(w, R, dR)
    W = r2_from_wr2(ww).data
    Rm = RR.data
    dRm = dRR.data
    rotated = dRm @ W @ Rm.transpose(-2, -1) + Rm @ W @ dRm.transpose(-2, -1)
    return skew(R2(rotated, sub_batch_ndim=sb))


# ---- Rotation of a general (asymmetric) R2 by an R2 ----
#
# Used in CP for the plastic spatial velocity gradient
# ``l^p = Q (sum_i gamma_i d_i (x) n_i) Q^T`` where the per-slip Schmid tensor
# is the full asymmetric outer product (no sym/skew projection).


def _rotate_r2(a: R2, R: R2) -> R2:
    """``R A Rᵀ`` — the full (asymmetric) 3x3 rotation, no projection."""
    [aa, rr], sb = align_sub_batch(a, R)
    rotated = rr.data @ aa.data @ rr.data.transpose(-2, -1)
    return R2(rotated, sub_batch_ndim=sb)


def _jvp_rotate_r2(a: R2, R: R2, dR: R2) -> R2:
    """Pushforward of :func:`rotate` (R2 overload) w.r.t. $R$ along ``dR``.

    ``rotate(a, R) = R A Rᵀ`` is linear in $a$ (so the $a$-direction is
    just ``rotate(da, R)`` and needs no primitive). The $R$-direction is
    the product rule ``dR A Rᵀ + R A dRᵀ``. ``dR`` is a leading-K ``R2``
    tangent; the 3x3 ``@`` / transpose broadcast $K$. Sub-batch alignment
    (e.g. per-crystal $R$ against a per-slip $a$) is handled by
    :func:`align_sub_batch`, exactly as in the forward.
    """
    [aa, RR, dRR], sb = align_sub_batch(a, R, dR)
    A = aa.data
    Rm = RR.data
    dRm = dRR.data
    rotated = dRm @ A @ Rm.transpose(-2, -1) + Rm @ A @ dRm.transpose(-2, -1)
    return R2(rotated, sub_batch_ndim=sb)


# ---- SSR4 rotation by R2 ----


def _mandel_basis_matrix(R: torch.Tensor) -> torch.Tensor:
    """6x6 Mandel basis rotation matrix ``Q(R)`` such that $T'_M = Q T_M Q^T$.

    The Mandel packing in NEML2 is ``[xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz,
    sqrt(2)*xy]``. The rotation matrix on this basis encodes the standard
    fourth-order rotation ``T'_ijkl = R_im R_jn R_kp R_lq T_mnpq`` for any
    SR2 $T_mn$ with the off-diagonal Mandel-weight factor absorbed.
    $R$ has shape ``(*, 3, 3)``; output has shape ``(*, 6, 6)``.
    """
    s2 = math.sqrt(2.0)
    r = R  # (..., 3, 3)
    # Each entry written explicitly to be torch.export-friendly (no
    # constant-tensor indexing acrobatics).
    Q = torch.stack(
        [
            torch.stack(
                [
                    r[..., 0, 0] * r[..., 0, 0],
                    r[..., 0, 1] * r[..., 0, 1],
                    r[..., 0, 2] * r[..., 0, 2],
                    s2 * r[..., 0, 1] * r[..., 0, 2],
                    s2 * r[..., 0, 0] * r[..., 0, 2],
                    s2 * r[..., 0, 0] * r[..., 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    r[..., 1, 0] * r[..., 1, 0],
                    r[..., 1, 1] * r[..., 1, 1],
                    r[..., 1, 2] * r[..., 1, 2],
                    s2 * r[..., 1, 1] * r[..., 1, 2],
                    s2 * r[..., 1, 0] * r[..., 1, 2],
                    s2 * r[..., 1, 0] * r[..., 1, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    r[..., 2, 0] * r[..., 2, 0],
                    r[..., 2, 1] * r[..., 2, 1],
                    r[..., 2, 2] * r[..., 2, 2],
                    s2 * r[..., 2, 1] * r[..., 2, 2],
                    s2 * r[..., 2, 0] * r[..., 2, 2],
                    s2 * r[..., 2, 0] * r[..., 2, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * r[..., 1, 0] * r[..., 2, 0],
                    s2 * r[..., 1, 1] * r[..., 2, 1],
                    s2 * r[..., 1, 2] * r[..., 2, 2],
                    r[..., 1, 1] * r[..., 2, 2] + r[..., 1, 2] * r[..., 2, 1],
                    r[..., 1, 0] * r[..., 2, 2] + r[..., 1, 2] * r[..., 2, 0],
                    r[..., 1, 0] * r[..., 2, 1] + r[..., 1, 1] * r[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * r[..., 0, 0] * r[..., 2, 0],
                    s2 * r[..., 0, 1] * r[..., 2, 1],
                    s2 * r[..., 0, 2] * r[..., 2, 2],
                    r[..., 0, 1] * r[..., 2, 2] + r[..., 0, 2] * r[..., 2, 1],
                    r[..., 0, 0] * r[..., 2, 2] + r[..., 0, 2] * r[..., 2, 0],
                    r[..., 0, 0] * r[..., 2, 1] + r[..., 0, 1] * r[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * r[..., 0, 0] * r[..., 1, 0],
                    s2 * r[..., 0, 1] * r[..., 1, 1],
                    s2 * r[..., 0, 2] * r[..., 1, 2],
                    r[..., 0, 1] * r[..., 1, 2] + r[..., 0, 2] * r[..., 1, 1],
                    r[..., 0, 0] * r[..., 1, 2] + r[..., 0, 2] * r[..., 1, 0],
                    r[..., 0, 0] * r[..., 1, 1] + r[..., 0, 1] * r[..., 1, 0],
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    return Q


def _mandel_basis_bilinear(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """6×6 bilinear form of the Mandel basis rotation matrix.

    Identical structure to :func:`_mandel_basis_matrix` but with two distinct
    input rotation tensors $A$, $B$, so each ``R_ij · R_kl`` product in
    the formula becomes ``A_ij · B_kl``. Used by :func:`d_rotate_dR` to
    compute the directional derivative of ``Q(R)`` via the product rule

        $dQ(R)[dR] = _mandel_basis_bilinear(R, dR) + _mandel_basis_bilinear(dR, R)$.
    """
    s2 = math.sqrt(2.0)
    a = A
    b = B
    Q = torch.stack(
        [
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 0],
                    a[..., 0, 1] * b[..., 0, 1],
                    a[..., 0, 2] * b[..., 0, 2],
                    s2 * a[..., 0, 1] * b[..., 0, 2],
                    s2 * a[..., 0, 0] * b[..., 0, 2],
                    s2 * a[..., 0, 0] * b[..., 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    a[..., 1, 0] * b[..., 1, 0],
                    a[..., 1, 1] * b[..., 1, 1],
                    a[..., 1, 2] * b[..., 1, 2],
                    s2 * a[..., 1, 1] * b[..., 1, 2],
                    s2 * a[..., 1, 0] * b[..., 1, 2],
                    s2 * a[..., 1, 0] * b[..., 1, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    a[..., 2, 0] * b[..., 2, 0],
                    a[..., 2, 1] * b[..., 2, 1],
                    a[..., 2, 2] * b[..., 2, 2],
                    s2 * a[..., 2, 1] * b[..., 2, 2],
                    s2 * a[..., 2, 0] * b[..., 2, 2],
                    s2 * a[..., 2, 0] * b[..., 2, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * a[..., 1, 0] * b[..., 2, 0],
                    s2 * a[..., 1, 1] * b[..., 2, 1],
                    s2 * a[..., 1, 2] * b[..., 2, 2],
                    a[..., 1, 1] * b[..., 2, 2] + a[..., 1, 2] * b[..., 2, 1],
                    a[..., 1, 0] * b[..., 2, 2] + a[..., 1, 2] * b[..., 2, 0],
                    a[..., 1, 0] * b[..., 2, 1] + a[..., 1, 1] * b[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * a[..., 0, 0] * b[..., 2, 0],
                    s2 * a[..., 0, 1] * b[..., 2, 1],
                    s2 * a[..., 0, 2] * b[..., 2, 2],
                    a[..., 0, 1] * b[..., 2, 2] + a[..., 0, 2] * b[..., 2, 1],
                    a[..., 0, 0] * b[..., 2, 2] + a[..., 0, 2] * b[..., 2, 0],
                    a[..., 0, 0] * b[..., 2, 1] + a[..., 0, 1] * b[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    s2 * a[..., 0, 0] * b[..., 1, 0],
                    s2 * a[..., 0, 1] * b[..., 1, 1],
                    s2 * a[..., 0, 2] * b[..., 1, 2],
                    a[..., 0, 1] * b[..., 1, 2] + a[..., 0, 2] * b[..., 1, 1],
                    a[..., 0, 0] * b[..., 1, 2] + a[..., 0, 2] * b[..., 1, 0],
                    a[..., 0, 0] * b[..., 1, 1] + a[..., 0, 1] * b[..., 1, 0],
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    return Q


def _d_rotate_ssr4_dR(T: SSR4, R: R2) -> torch.Tensor:
    """``d(rotate(T, R)) / dR`` (SSR4 overload) as a ``(..., 6, 6, 9)`` Jacobian.

    ``Q(R)`` is quadratic in $R$, so the product rule gives
    $dQ_dir[A, B] = _mandel_basis_bilinear(R, dR)[A, B] + _mandel_basis_bilinear(dR, R)[A, B]$
    for any direction ``dR``. The full Jacobian iterates the 9 unit-direction
    perturbations of $R$; each yields one ``(*B, 6, 6)`` slab assembled into
    the trailing 9 axis. No autograd.
    """
    Q = _mandel_basis_matrix(R.data)  # (..., 6, 6)
    T_d = T.data  # (..., 6, 6)
    slabs: list[torch.Tensor] = []
    for c in range(3):
        for d in range(3):
            dR = R.data.new_zeros(*R.data.shape[:-2], 3, 3)
            dR[..., c, d] = 1.0
            # dQ shape: (..., 6, 6)
            dQ = _mandel_basis_bilinear(R.data, dR) + _mandel_basis_bilinear(dR, R.data)
            dTrot = dQ @ T_d @ Q.transpose(-2, -1) + Q @ T_d @ dQ.transpose(-2, -1)
            slabs.append(dTrot)
    return torch.stack(slabs, dim=-1)  # (..., 6, 6, 9)


def _rotate_ssr4(T: SSR4, R: R2) -> SSR4:
    """``T'_ijkl = R_im R_jn R_kp R_lq T_mnpq`` performed in Mandel packing.

    Builds the 6x6 Mandel basis rotation ``Q(R)`` and forms $Q T Q^T$;
    matches the C++ ``SSR4::rotate(R2)`` semantics.
    """
    [TT, RR], sb = align_sub_batch(T, R)
    Q = _mandel_basis_matrix(RR.data)
    rotated = Q @ TT.data @ Q.transpose(-2, -1)
    return SSR4(rotated, sub_batch_ndim=sb)


# ---- Typed JVP (pushforward) primitives for the irreducible geometric maps ----
#
# D-062: leaf chain-rule actions are differential pushforwards in strongly-typed
# wrapper algebra. The only irreducible local-derivative contractions — the
# nonlinear geometric maps whose JVP is intrinsically ``d(map)(x)·dx`` with no
# product-rule decomposition — live here, hidden behind a typed
# ``(x, dx_tangent) -> dy_tangent`` interface (the same way the forward primitive
# hides its own math). A tangent of type ``T`` is an ordinary ``T`` with the
# seed-direction axis ``K`` as the **leftmost** batch dim
# (``data.shape == (K, *batch, *base)``); the ``d*`` local-derivative tensor
# carries no ``K``, so it broadcasts over ``K`` under right-aligned broadcasting.


def jvp_euler_rodrigues(r: Rot, dr: Rot) -> R2:
    """Pushforward of :func:`euler_rodrigues` (Rot→R2) along the tangent ``dr``.

    Closed-form via the body-frame angular rate. ``Rot`` is the Modified
    Rodrigues Parameter (MRP) form (``Rot.cxx``), for which the MRP-rate /
    body-rate kinematic relation (Schaub & Junkins, *Analytical Mechanics of
    Space Systems*) inverts to::

        ω_b = (4 / s²) · [ (1 − r·r) v − 2 (r × v) + 2 (r·v) r ]
        D R[r]{v} = R(r) · [ω_b]_×

    with $s = 1 + r·r$ and ``v = dr``. The action collapses to one 3×3
    skew matrix multiply — no ``(..., 3, 3, 3)`` derivative kernel is ever
    formed. (The simpler $ω_b = (2/s)(v − r × v)$ you may have seen applies
    to the *classical* Rodrigues form $R = I + (2/s)([r]_× + [r]_×²)$, not
    the MRP form NEML2 uses.)
    """
    r_d, v_d = r.data, dr.data  # r_d: (*batch, 3); v_d: (K, *batch, 3)
    rr = (r_d * r_d).sum(dim=-1, keepdim=True)  # (*batch, 1)
    s = 1.0 + rr  # (*batch, 1)
    rTv = (r_d * v_d).sum(dim=-1, keepdim=True)  # (K, *batch, 1) — r broadcasts
    # r × v as explicit per-component stack — broadcasts the K-less r against
    # the leading-K v naturally, which torch.linalg.cross's ndim-strict
    # validator rejects.
    r0, r1, r2 = r_d[..., 0], r_d[..., 1], r_d[..., 2]
    v0, v1, v2 = v_d[..., 0], v_d[..., 1], v_d[..., 2]
    rxv = torch.stack([r1 * v2 - r2 * v1, r2 * v0 - r0 * v2, r0 * v1 - r1 * v0], dim=-1)
    omega_b = (4.0 / (s * s)) * ((1.0 - rr) * v_d - 2.0 * rxv + 2.0 * rTv * r_d)
    skew = r2_from_wr2(WR2(omega_b, sub_batch_ndim=dr.sub_batch_ndim)).data
    R_mat = euler_rodrigues(r).data  # (*batch, 3, 3) — K-less, broadcasts left
    dR = R_mat @ skew  # (K, *batch, 3, 3)
    return R2(dR, sub_batch_ndim=dr.sub_batch_ndim)


def jvp_exp_map(w: WR2, dw: WR2) -> Rot:
    """Pushforward of :func:`exp_map` (WR2→Rot) along the tangent ``dw``.

    Closed-form rank-1-plus-identity: $dexp_map(w) = a(|w|²) I + b(|w|²) w wᵀ$,
    so the action is $dr = a·dw + b·(w·dw)·w$ — two vector ops, no 3×3
    matrix materialised. $a$ and $b$ are the same scalar coefficients
    :func:`dexp_map` builds the matrix from, with the same Taylor branch
    near ``||w||² ≈ 0`` to avoid the origin singularity.
    """
    w_d, dw_d = w.data, dw.data  # w_d: (*batch, 3); dw_d: (K, *batch, 3)
    eps = torch.finfo(w.dtype).eps
    thresh = eps ** (1.0 / 3.0)
    norm2 = (w_d * w_d).sum(dim=-1, keepdim=True)  # (*batch, 1)

    # Taylor near zero: a = 1/4 + 5||w||⁴/96, b = 5||w||²/24.
    a_taylor = 0.25 + 5.0 * norm2 * norm2 / 96.0
    b_taylor = 5.0 * norm2 / 24.0
    # Actual formula (kinematic-identity coefficients).
    safe_norm2 = torch.where(norm2 > thresh, norm2, torch.ones_like(norm2))
    half = safe_norm2 / 2.0
    cos_h = torch.cos(half)
    tan_h = torch.tan(half)
    sec = 1.0 / cos_h
    a_actual = tan_h / (2.0 * safe_norm2 * cos_h)
    b_actual = (safe_norm2 * sec**3 + tan_h * (safe_norm2 * tan_h - 2.0) * sec) / (
        2.0 * safe_norm2 * safe_norm2
    )
    a = torch.where(norm2 > thresh, a_actual, a_taylor)
    b = torch.where(norm2 > thresh, b_actual, b_taylor)

    w_dot_dw = (w_d * dw_d).sum(dim=-1, keepdim=True)  # (K, *batch, 1) — w broadcasts
    dr = a * dw_d + (b * w_dot_dw) * w_d  # (K, *batch, 3)
    return Rot(dr, sub_batch_ndim=dw.sub_batch_ndim)


def jvp_compose(r1: Rot, r2: Rot, *, dr1: Rot | None = None, dr2: Rot | None = None) -> Rot:
    """Pushforward of :func:`compose` (``compose(r1, r2)``) along its operands.

    $d(compose(r1, r2)) = (∂/∂r1)·dr1 + (∂/∂r2)·dr2$ with the operand
    derivatives given by the established :func:`drotate` / :func:`drotate_self`
    convention ($∂compose(r1, r2)/∂r1 = drotate(r2, r1)$,
    $∂compose(r1, r2)/∂r2 = drotate_self(r2, r1)$ — both 3×3 ``R2`` linear maps
    from a 3-vector tangent to a 3-vector tangent). Pass only the operands that
    vary; a ``None`` tangent means that operand is held fixed.
    """
    acc: torch.Tensor | None = None
    if dr1 is not None:
        term = (drotate(r2, r1).data @ dr1.data.unsqueeze(-1)).squeeze(-1)
        acc = term if acc is None else acc + term
    if dr2 is not None:
        term = (drotate_self(r2, r1).data @ dr2.data.unsqueeze(-1)).squeeze(-1)
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("jvp_compose requires at least one of dr1, dr2")
    sb = dr1.sub_batch_ndim if dr1 is not None else dr2.sub_batch_ndim  # type: ignore[union-attr]
    return Rot(acc, sub_batch_ndim=sb)


def _jvp_rotate_ssr4(T: SSR4, R: R2, dR: R2) -> SSR4:
    """Pushforward of :func:`rotate` (SSR4 overload) w.r.t. $R$ along ``dR``.

    ``rotate(T, R)`` is ``Q(R) T Q(R)ᵀ`` with the 6×6 Mandel rotation $Q$
    quadratic in $R$; the directional derivative is
    ``dQ T Qᵀ + Q T dQᵀ`` with
    $dQ = bilinear(R, dR) + bilinear(dR, R)$. $T$ is held fixed (the
    parameter), so only the $R$-dependence is pushed forward. ``dR`` is a
    leading-K ``R2`` tangent; ``_mandel_basis_bilinear`` broadcasts $K$.
    """
    Q = _mandel_basis_matrix(R.data)  # (*batch, 6, 6)
    dQ = _mandel_basis_bilinear(R.data, dR.data) + _mandel_basis_bilinear(dR.data, R.data)
    T_d = T.data  # (*batch, 6, 6)
    dTrot = dQ @ T_d @ Q.transpose(-2, -1) + Q @ T_d @ dQ.transpose(-2, -1)
    return SSR4(dTrot, sub_batch_ndim=dR.sub_batch_ndim)


# ---- Unified rotate / jvp_rotate / d_rotate_dR entry points ----
#
# The public surface is three names, each overloaded on the operand type. The
# underlying ``_rotate_*`` / ``_jvp_rotate_*`` / ``_d_rotate_ssr4_dR`` kernels
# above hold the actual implementations; this section threads them through
# ``@typing.overload`` so static type-checkers infer ``rotate(SR2, R2) -> SR2``,
# ``rotate(R2, R2) -> R2``, etc. The runtime dispatch is a single isinstance
# chain — more specific types (SR2, WR2, SSR4) checked before the catch-all
# (R2) since they are not subclasses of each other.


@overload
def rotate(x: SR2, R: R2) -> SR2: ...
@overload
def rotate(x: WR2, R: R2) -> WR2: ...
@overload
def rotate(x: SSR4, R: R2) -> SSR4: ...
@overload
def rotate(x: R2, R: R2) -> R2: ...
def rotate(x, R):
    """Rotate a typed tensor by an ``R2`` rotation matrix.

    Overloaded on the operand type:

    - ``SR2 -> SR2`` — ``sym(R S Rᵀ)`` packed back to Mandel.
    - ``WR2 -> WR2`` — ``skew(R W Rᵀ)`` packed back to an axial vector.
    - ``R2 -> R2`` — the full asymmetric ``R A Rᵀ`` (no projection).
    - ``SSR4 -> SSR4`` — the 6×6 Mandel basis rotation ``Q(R) T Q(R)ᵀ``.
    """
    if isinstance(x, SR2):
        return _rotate_sym(x, R)
    if isinstance(x, WR2):
        return _rotate_skew(x, R)
    if isinstance(x, SSR4):
        return _rotate_ssr4(x, R)
    if isinstance(x, R2):
        return _rotate_r2(x, R)
    raise TypeError(f"rotate: unsupported operand type {type(x).__name__}")


@overload
def jvp_rotate(x: SR2, R: R2, dR: R2) -> SR2: ...
@overload
def jvp_rotate(x: WR2, R: R2, dR: R2) -> WR2: ...
@overload
def jvp_rotate(x: SSR4, R: R2, dR: R2) -> SSR4: ...
@overload
def jvp_rotate(x: R2, R: R2, dR: R2) -> R2: ...
def jvp_rotate(x, R, dR):
    """Pushforward of :func:`rotate` along the tangent ``dR``.

    Overloaded on ``x``'s type. The forward is always linear in ``x``, so only
    the ``R``-direction needs an explicit primitive; the ``x``-direction is
    just ``rotate(dx, R)`` and can be expressed directly.
    """
    if isinstance(x, SR2):
        return _jvp_rotate_sym(x, R, dR)
    if isinstance(x, WR2):
        return _jvp_rotate_skew(x, R, dR)
    if isinstance(x, SSR4):
        return _jvp_rotate_ssr4(x, R, dR)
    if isinstance(x, R2):
        return _jvp_rotate_r2(x, R, dR)
    raise TypeError(f"jvp_rotate: unsupported operand type {type(x).__name__}")


def d_rotate_dR(x: SSR4, R: R2) -> torch.Tensor:
    """``d(rotate(x, R)) / dR`` as a raw-tensor Jacobian.

    SSR4-only today — returns the ``(..., 6, 6, 9)`` Jacobian over the 9
    free components of ``R``. The R2/SR2/WR2 cases have closed-form
    pushforwards via :func:`jvp_rotate` and don't need the full Jacobian,
    so they aren't overloaded here. Add ``@overload`` annotations if and
    when another operand type grows a Jacobian-style derivative.
    """
    if not isinstance(x, SSR4):
        raise TypeError(f"d_rotate_dR: unsupported operand type {type(x).__name__}")
    return _d_rotate_ssr4_dR(x, R)


# ---- Mandel/axial packing helpers (no autograd) ----


def mandel_pack_sym3(X_full: torch.Tensor) -> torch.Tensor:
    """Pack a (...,3,3) symmetric tensor to Mandel (...,6) without going through SR2.

    Useful in chain-rule actions that need the Mandel form of an intermediate
    full 3×3 result. Equivalent to ``sym(R2(X_full)).data`` but bypasses
    wrapper construction.

    Kept as the explicit select+add+mul+stack chain. The matmul-against-
    fixed-projection alternative (``flat @ _MANDEL_PACK``) was measurably
    slower on CUDA at small batch — see the design note on
    :func:`r2_from_sr2`.
    """
    a = X_full[..., 0, 0]
    b = X_full[..., 1, 1]
    c = X_full[..., 2, 2]
    yz = (X_full[..., 1, 2] + X_full[..., 2, 1]) * (_SQRT2 / 2.0)
    xz = (X_full[..., 0, 2] + X_full[..., 2, 0]) * (_SQRT2 / 2.0)
    xy = (X_full[..., 0, 1] + X_full[..., 1, 0]) * (_SQRT2 / 2.0)
    return torch.stack([a, b, c, yz, xz, xy], dim=-1)


def skew_pack_axial3(X_full: torch.Tensor) -> torch.Tensor:
    """Pack a (...,3,3) tensor to its skew-axial (...,3) form (matches `skew(R2)`).

    Kept as the explicit select+sub+div+stack chain for the same CUDA
    fusion reason as :func:`mandel_pack_sym3`.
    """
    w0 = (X_full[..., 2, 1] - X_full[..., 1, 2]) / 2.0
    w1 = (X_full[..., 0, 2] - X_full[..., 2, 0]) / 2.0
    w2 = (X_full[..., 1, 0] - X_full[..., 0, 1]) / 2.0
    return torch.stack([w0, w1, w2], dim=-1)


# ---- Vec helpers ----


def vec_component(v: Vec, i: int) -> Scalar:
    """Extract the ``i``-th Scalar component of a ``Vec`` (i in 0, 1, 2).

    Mirrors the C++ ``Vec::operator()(int)`` slot access used by leaves like
    ``VecComponents`` that decompose a Vec into per-axis Scalars. Preserves
    sub-batch metadata; works inside a leaf's forward without dropping out
    of wrapper algebra.
    """
    if i < 0 or i > 2:
        raise IndexError(f"vec_component index {i} out of range [0, 3)")
    return Scalar(v.data[..., i], sub_batch_ndim=v.sub_batch_ndim)


def vec_from_scalars(s0: Scalar, s1: Scalar, s2: Scalar) -> Vec:
    """Assemble a ``Vec`` from three ``Scalar`` components.

    Mirrors the C++ ``Vec::fill(Scalar, Scalar, Scalar)`` factory: stacks the
    three Scalar values along a fresh trailing axis to produce a ``(..., 3)``
    ``Vec``. All three inputs must share dtype/device; sub-batch alignment
    flows through :func:`align_sub_batch` so per-sub-batch and global Scalars
    combine cleanly.
    """
    [aa, bb, cc], sb = align_sub_batch(s0, s1, s2)
    return Vec(torch.stack([aa.data, bb.data, cc.data], dim=-1), sub_batch_ndim=sb)


# ---- MillerIndex helpers ----


def to_cartesian(mi: MillerIndex, lattice: R2) -> torch.Tensor:
    """Convert Miller indices to Cartesian coordinates using a lattice matrix.

    ``lattice`` is the matrix whose columns are the three lattice vectors
    (``a1, a2, a3``); for a cubic crystal with parameter ``a`` this is
    ``a * I``. Returns a raw ``(..., 3)`` tensor of Cartesian components.
    """
    # mi.data: (..., 3); lattice.data: (3, 3) or broadcastable.
    # ``"...j,ij->...i"`` is the matvec ``lattice @ mi``.
    return (lattice.data @ mi.data.unsqueeze(-1)).squeeze(-1)


__all__ = [
    "abs",
    "bilinear_interpolation",
    "bilinear_interpolation_slopes",
    "compose",
    "cosh",
    "d_rotate_dR",
    "det",
    "deuler_rodrigues",
    "dev",
    "dexp_map",
    "diff",
    "drotate",
    "drotate_self",
    "euler_rodrigues",
    "exp",
    "exp_map",
    "gt",
    "inner",
    "inv",
    "jvp_compose",
    "jvp_euler_rodrigues",
    "jvp_exp_map",
    "jvp_rotate",
    "linear_interpolation",
    "log",
    "log10",
    "lt",
    "macaulay",
    "mandel_pack_sym3",
    "mean",
    "norm",
    "outer",
    "pow",
    "r2_from_sr2",
    "r2_from_wr2",
    "rotate",
    "skew",
    "skew_pack_axial3",
    "sign",
    "sinh",
    "sqrt",
    "sub_batch_zeros_like",
    "sum",
    "sym",
    "tanh",
    "to_cartesian",
    "tr",
    "unit",
    "vec_component",
    "vec_from_scalars",
    "vol",
]
