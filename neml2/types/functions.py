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
from collections.abc import Sequence
from typing import TypeVar, cast, overload

import torch

from neml2.types._base import (
    BaseView,
    BatchView,
    DynamicBatchView,
    KStateFlag,
    SubBatchStateFlag,
    SubBatchView,
    TensorWrapper,
    align_scalar_base,
    align_sub_batch,
    combine_sub_batch_state,
    wrap_like,
)
from neml2.types.r2 import R2
from neml2.types.rot import Rot
from neml2.types.scalar import Scalar
from neml2.types.sr2 import SR2
from neml2.types.ssr4 import SSR4
from neml2.types.tensor import Tensor
from neml2.types.tensor import _BaseView as _TensorBaseView
from neml2.types.tensor import _RegionView as _TensorRegionView
from neml2.types.tensor import cat as _tensor_cat
from neml2.types.tensor import stack as _tensor_stack
from neml2.types.vec import Vec
from neml2.types.wr2 import WR2

_TW = TypeVar("_TW", bound=TensorWrapper)


def _combine_k_from_operands(
    *wrappers: TensorWrapper,
) -> tuple[int, tuple[KStateFlag, ...], tuple[int | None, ...]]:
    """Combine K_ndim / K_state / K_pairing across wrapper operands.

    For mixed k_ndim, takes the maximum -- the OP is assumed to use torch
    broadcasting on the underlying data so the result's data naturally
    has the max-K shape. Returns ``(k_ndim, k_state, k_pairing)``.

    The K threading at binary-op call sites in this module follows the
    "broadcast-friendly" pattern: operands with smaller k_ndim broadcast
    against operands with larger k_ndim via torch's right-aligned rules
    (the leading K axes of the larger operand stay leading after the op).
    """
    # Explicit loop rather than ``max((w.k_ndim for w in wrappers), default=0)``:
    # ``k_ndim`` is always >= 0 so starting at 0 is equivalent to ``default=0``,
    # and the loop form is traceable by ``torch.compile`` (Dynamo has no handler
    # for the builtin ``max``'s ``default=`` kwarg, which otherwise graph-breaks
    # this on every typed-tensor binary op in the chain-rule JVP path).
    kmax = 0
    for w in wrappers:
        if w.k_ndim > kmax:
            kmax = w.k_ndim
    if kmax == 0:
        return 0, (), ()
    # Take the K_state / K_pairing of the operand with the max k_ndim.
    # If multiple operands tie at kmax, prefer the first one. The op's
    # broadcast preserves K shape semantically.
    for w in wrappers:
        if w.k_ndim == kmax:
            return kmax, w.k_state, w.k_pairing
    return kmax, cast("tuple[KStateFlag, ...]", ("full",) * kmax), (None,) * kmax


# ---- Opaque ``pow`` op (Inductor fusion barrier) ------------------------------
#
# Register ``neml2::opaque_pow`` so Inductor's fusion pass treats the pow
# call as a black box rather than inlining it into downstream fused
# kernels. This is an **opt-in** barrier surfaced via the :func:`opaque_pow`
# free function -- the default :func:`pow` still routes through
# ``torch.pow`` directly so the call fuses with surrounding pointwise ops
# on the common path.
#
# When the barrier matters: a leaf whose pow output flows into a reduction
# whose output references each pow value many times. The benchmarked case
# is ``PowerLawSlipRule``'s ``|τ/τ̂|^(n-1)`` -> ``SumSlipRates`` per-slip sum
# -> K-batched JVP -- Triton inlines the pow into the (K, B, n_slip)
# reduction kernel and recomputes it K × n_slip times because the pow
# inputs don't depend on K or n_slip. Measured on scpcoup CUDA B=8192:
# 4.95 s without the barrier vs 2.14 s with it (2.3x); across the CP suite
# the barrier delivers 2-3x. For everything else the barrier costs a real
# fusion opportunity, so ``opaque_pow`` is leaf-specific opt-in.
#
# We use the low-level ``torch.library.Library`` API rather than the
# ``@torch.library.custom_op`` decorator because the decorator
# transitively imports ``torch._dynamo`` → ``torch.distributed.fsdp``
# whose ``FlatParamHandle`` class decorator calls ``torch.enable_grad()``
# at import time -- triggering NEML2's autograd guard (see
# ``neml2/_guard.py``). The ``Library.define`` / ``impl`` path stays
# dynamo-free.
_NEML2_LIB = torch.library.Library("neml2", "FRAGMENT")
_NEML2_LIB.define("opaque_pow(Tensor base, Tensor exponent) -> Tensor")


def _opaque_pow_impl(base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    return torch.pow(base, exponent)


def _opaque_pow_meta(base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(base)


def _opaque_pow_setup_context(ctx, inputs, output) -> None:
    base, exponent = inputs
    ctx.save_for_backward(base, exponent)


def _opaque_pow_backward(ctx, grad):
    # Explicit backward for the eager-autograd paths (pyzag adjoint training,
    # in-process ``torch.compile`` Jacobian). Standard power-rule gradients
    # ``d(b^e)/db = e*b^(e-1)`` and ``d(b^e)/de = b^e*ln(b)``, computed only
    # for the inputs that require grad.
    base, exponent = ctx.saved_tensors
    grad_base = grad_exponent = None
    if ctx.needs_input_grad[0]:
        grad_base = grad * exponent * torch.pow(base, exponent - 1.0)
    if ctx.needs_input_grad[1]:
        grad_exponent = grad * torch.pow(base, exponent) * torch.log(base)
    return grad_base, grad_exponent


_NEML2_LIB.impl("opaque_pow", _opaque_pow_impl, "CompositeExplicitAutograd")
# Mark ``neml2::opaque_pow`` as a proper opaque custom op via ``register_fake``
# (abstract impl) + ``register_autograd`` (explicit backward). Two subtleties,
# each load-bearing for keeping the op OPAQUE through ``torch.export`` /
# ``run_decompositions()`` (the AOTI lowering path):
#
#   * a bare ``lib.impl(..., "Meta")`` does NOT mark the op as preservable, so
#     export decomposes it back into its ``CompositeExplicitAutograd`` body
#     (``torch.pow``); ``register_fake`` does mark it.
#   * registering the *forward* on the ``Autograd`` key (the old approach) is
#     itself a decomposition -- export inlines the Autograd-key body to
#     ``aten.pow``; ``register_autograd`` supplies a backward while leaving the
#     forward opaque.
#
# When the op decomposes, Inductor inlines the pow into the downstream
# K-batched per-slip reduction and recomputes it K x n_slip times -- the exact
# regression this op exists to prevent (~2x on the crystal-plasticity suite).
torch.library.register_fake("neml2::opaque_pow", _opaque_pow_meta)
torch.library.register_autograd(
    "neml2::opaque_pow", _opaque_pow_backward, setup_context=_opaque_pow_setup_context
)


# ---- AOTI-safe saved-output element-wise ops ---------------------------------
#
# AOTInductor cannot lower a reverse-mode-autograd graph through an element-wise
# op whose backward SAVES ITS OUTPUT (sqrt / exp / tanh / reciprocal -- and
# ``const / x`` division, which lowers to reciprocal). Under strict export +
# dynamic batch + ``torch._dynamo.config.trace_autograd_ops``, AOTAutograd lifts
# that saved-output activation as a CONSTANT with a symbolic batch shape, which
# Inductor can neither inline nor serialise (pytorch/pytorch#187907). Ops that
# save their INPUT (log, x*x, pow, mul, x/const) reference the input placeholder
# instead and lower fine.
#
# Fix: a transparent ``autograd.Function`` whose backward recomputes the
# derivative FROM THE SAVED INPUT, so the backward references the input
# placeholder, never a lifted saved-output constant. Empirically validated to:
# lower through AOTI, survive ``ep.run_decompositions()``, pass
# ``torch.compile(fullgraph=True)`` with no graph break (safe for pyzag), keep
# the forward a fusible ``aten.<op>`` (no fusion barrier), and produce exact
# numerics. Dispatch is conditional on ``requires_grad``: the value / no-AD path
# uses the plain fast ``torch.<op>`` with zero Function overhead; only the AD
# path uses the recompute Function.
#
# Defining ``autograd.Function`` subclasses at import is dynamo-free (unlike the
# ``@torch.library.custom_op`` decorator -- see the opaque_pow note above): a
# plain subclass is pure Python with no ``torch._dynamo`` import, so it does not
# trip the autograd guard at import. The guard (``neml2/models/_guard.py``) does
# NOT patch ``Function.apply``, so these are guard-compatible inside a forward.
# Remove this machinery once #187907 is fixed (the xfail test
# ``tests/aoti/test_upstream_pytorch_187907.py`` fires when it is).


def _recompute_unary(fwd, dfwd):
    """Build a transparent ``autograd.Function.apply`` for a unary element-wise
    op whose backward recomputes ``dfwd`` FROM THE SAVED INPUT.

    ``fwd(x)`` is the forward (e.g. ``torch.exp``); ``dfwd(x)`` is the local
    derivative recomputed from the input (e.g. ``torch.exp(x)`` for exp,
    ``0.5 * x**-0.5`` for sqrt). The forward stays a bare ``aten.<op>`` so
    Inductor fuses it; the backward saves only ``x`` (an existing graph
    placeholder), so AOTI lowers it instead of lifting a saved-output constant.

    Both ``fwd`` and ``dfwd`` are built from differentiable torch ops, so the
    Function supports double-backward. ``dfwd`` must itself use only saved-INPUT
    ops (pow / mul) to stay AOTI-safe at second order.
    """

    class _Recompute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            ctx.save_for_backward(x)
            return fwd(x)

        @staticmethod
        def backward(ctx, *grad_outputs):  # type: ignore[override]
            (g,) = grad_outputs
            (x,) = ctx.saved_tensors
            return g * dfwd(x)

    return _Recompute.apply


def _ad_safe(data: torch.Tensor, fast, recompute) -> torch.Tensor:
    """Raw-tensor dispatcher: plain ``fast(data)`` off the AD path, the
    input-recompute ``recompute(data)`` on it (keyed on ``data.requires_grad``).

    Raw ``torch.Tensor`` in/out -- internal to ``neml2/types/`` where ``.data``
    access is permitted. dtype / device agnostic (both branches operate on the
    passed tensor).
    """
    if data.requires_grad:
        return recompute(data)
    return fast(data)


# Saved-output forwards paired with their input-recompute derivatives. Each
# ``dfwd`` recomputes the local slope from the INPUT x using only saved-input
# ops (pow / mul), so it is AOTI-safe and safe at second order:
#   sqrt:       d/dx sqrt(x)  = 0.5 * x**-0.5
#   exp:        d/dx exp(x)   = exp(x)            (recomputed, not the saved output)
#   tanh:       d/dx tanh(x)  = 1 - tanh(x)**2    (recomputed from input)
#   reciprocal: d/dx (1/x)    = -x**-2            (pow, not a nested reciprocal)
_sqrt_ad = _recompute_unary(torch.sqrt, lambda x: 0.5 * x**-0.5)
_exp_ad = _recompute_unary(torch.exp, torch.exp)
_tanh_ad = _recompute_unary(torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2)
_reciprocal_ad = _recompute_unary(torch.reciprocal, lambda x: -(x**-2))


def sqrt_ad(data: torch.Tensor) -> torch.Tensor:
    """``torch.sqrt`` as a raw tensor, AOTI-safe under reverse-mode AD (see
    :func:`_ad_safe`). For direct-call sites that bypass the typed wrappers."""
    return _ad_safe(data, torch.sqrt, _sqrt_ad)


def exp_ad(data: torch.Tensor) -> torch.Tensor:
    """``torch.exp`` as a raw tensor, AOTI-safe under reverse-mode AD."""
    return _ad_safe(data, torch.exp, _exp_ad)


def tanh_ad(data: torch.Tensor) -> torch.Tensor:
    """``torch.tanh`` as a raw tensor, AOTI-safe under reverse-mode AD."""
    return _ad_safe(data, torch.tanh, _tanh_ad)


def reciprocal_ad(data: torch.Tensor) -> torch.Tensor:
    """Reciprocal ``1/data`` as a raw tensor, AOTI-safe under reverse-mode AD.

    Off the AD path returns plain ``torch.reciprocal``; on it returns the
    input-recompute Function (backward ``-x**-2``). Used by the division dunders
    to route a denominator-requires-grad ``num / den`` through
    ``num * reciprocal_ad(den)`` without lifting a saved-output constant.
    """
    return _ad_safe(data, torch.reciprocal, _reciprocal_ad)


# ---- SR2 invariants / decompositions ----


def tr(A: SR2) -> Scalar:
    """Trace of a symmetric rank-2 tensor.

    Matches ``neml2::tr(const SR2&)`` in ``src/neml2/tensors/functions/tr.cxx``.
    """
    return wrap_like(
        Scalar,
        A.data[..., 0] + A.data[..., 1] + A.data[..., 2],
        A,
    )


def vol(A: SR2) -> SR2:
    """Volumetric part of a symmetric rank-2 tensor: ``tr(A)/3 * I``.

    Matches ``neml2::vol(const SR2&)`` in ``src/neml2/tensors/functions/vol.cxx``.
    """
    I = SR2.identity(dtype=A.dtype, device=A.device).data
    return A._rewrap(align_scalar_base(tr(A).data / 3.0, 1) * I, sub_batch_ndim=A.sub_batch_ndim)


def dev(A: SR2) -> SR2:
    """Deviatoric part of a symmetric rank-2 tensor: ``A - vol(A)``.

    Matches ``neml2::dev(const SR2&)`` in ``src/neml2/tensors/functions/dev.cxx``.
    """
    return A - vol(A)


@overload
def norm(A: SR2, eps: float = 0.0) -> Scalar: ...


@overload
def norm(A: _TensorBaseView, eps: float = 0.0) -> Tensor: ...


def norm(A, eps: float = 0.0):
    """Euclidean / Frobenius norm depending on the input type.

    * ``norm(A: SR2, eps=0.0)`` -- Frobenius norm of a symmetric rank-2
      tensor in Mandel packing, computed as ``sqrt(norm_sq(A) + eps)``.
      The ``eps`` regularizer (added unsquared under the sqrt) keeps the
      result differentiable at ``A == 0``; matches ``neml2::norm`` on
      the v2 C++ side.
    * ``norm(t.base)`` -- ``sqrt(sum_over_base(t * t))`` on a
      :class:`~neml2.types.Tensor`. Returns a ``Tensor`` with
      ``base_ndim=0``; ``batch`` / ``sub_batch`` axes are preserved.
    """
    # ``sqrt_ad`` (not raw ``torch.sqrt``): this Frobenius norm is the J2 / von
    # Mises norm on the parameter-derivative path; raw sqrt's saved-output
    # backward would not lower through AOTI (pytorch/pytorch#187907).
    if isinstance(A, _TensorBaseView):
        ns = norm_sq(A)
        return Tensor(sqrt_ad(ns.data), ns.batch_ndim, ns.sub_batch_ndim)
    sr2 = cast(SR2, A)
    sq = (sr2.data * sr2.data).sum(dim=-1)
    return wrap_like(Scalar, sqrt_ad(sq + eps), sr2)


def dot(a: _TensorBaseView, b: _TensorBaseView) -> Tensor:
    """Element-wise product summed over the base axes.

    Two :class:`~neml2.types.Tensor` base views are multiplied
    element-wise (Tensor's right-aligned broadcasting on batch + sub-
    batch applies), then summed over all base axes. The result has
    ``base_ndim=0``; ``batch`` / ``sub_batch`` axes survive.
    """
    return sum((a._t * b._t).base, dims=None)


def norm_sq(view: _TensorBaseView) -> Tensor:
    """``sum_over_base(t * t)`` for a :class:`~neml2.types.Tensor`."""
    return dot(view, view)


def unit(A: SR2, eps: float = 0.0) -> SR2:
    """Normalize $A$ by its Frobenius norm. ``eps`` regularizes at ``A == 0``."""
    n = align_scalar_base(norm(A, eps).data, 1)
    # On the AD path divide via the input-recompute reciprocal so the norm in the
    # denominator doesn't trip AOTI's saved-output lowering (pytorch/pytorch#187907);
    # off it keep the plain divide (value path byte-identical).
    if n.requires_grad:
        return A._rewrap(A.data * reciprocal_ad(n), sub_batch_ndim=A.sub_batch_ndim)
    return A._rewrap(A.data / n, sub_batch_ndim=A.sub_batch_ndim)


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
        return w, w.k_ndim, w.k_ndim + len(w.dynamic_batch_shape)
    if isinstance(view, SubBatchView):
        w = view._w
        start = w.k_ndim + len(w.dynamic_batch_shape)
        return w, start, start + w.sub_batch_ndim
    raise TypeError(
        f"{op_name} expects t.dynamic_batch or t.sub_batch view; got {type(view).__name__}. "
        "(base reductions change wrapper type; batch reductions would straddle the "
        "dynamic/sub-batch split.)"
    )


@overload
def sum(  # noqa: A001
    view: DynamicBatchView[_TW] | SubBatchView[_TW],
    dims: int | list[int] | tuple[int, ...] = 0,
    keepdim: bool = False,
) -> _TW: ...


@overload
def sum(  # noqa: A001
    view: _TensorBaseView,
    dims: int | list[int] | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor: ...


def sum(  # noqa: A001 — intentionally shadows builtin, callers import explicitly
    view,
    dims: int | list[int] | tuple[int, ...] | None = 0,
    keepdim: bool = False,
):
    """Sum over axes of a region view.

    Two view families are supported:

    * **TensorWrapper** (``t.dynamic_batch`` or ``t.sub_batch``). When
      summing over a sub-batch axis with ``keepdim=False``, the
      result's ``sub_batch_ndim`` drops by the number of reduced axes.
      Returns the same wrapper type as the view's underlying wrapper.
      When reducing over a sub-batch axis that's currently stored
      ``"broadcast"`` (size 1 with logical extent in
      ``sub_batch_meta``), the wrapper is materialised first so the
      sum sees every per-site copy.

    * **Tensor base** (``t.base`` on a :class:`~neml2.types.Tensor`).
      ``dims=None`` means "sum over every base axis", collapsing to
      ``base_ndim=0``; an explicit ``dims`` reduces the named region-
      relative axes. ``keepdim`` follows torch semantics.

    ``t.batch`` and TensorWrapper ``t.base`` are rejected: the former
    would straddle dynamic/sub-batch; the latter would change the
    wrapper type (use ``Tensor.base`` for dynamic-base reductions).
    """
    if isinstance(view, _TensorBaseView):
        t = view._t
        if t.base_ndim == 0:
            return t  # nothing to reduce
        if dims is None:
            absdims = list(range(t.batch_ndim + t.sub_batch_ndim, t.data.ndim))
        else:
            absdims = [view._resolve_dim(d) for d in ((dims,) if isinstance(dims, int) else dims)]
        new_data = torch.sum(t.data, dim=absdims, keepdim=keepdim)
        return Tensor(new_data, t.batch_ndim, t.sub_batch_ndim)
    assert dims is not None
    w, start, end = _reduce_view_bounds(view, "sum")
    dn = _normalize_dims(dims, start, end)
    # Expose path: if reducing exactly one sub axis and that axis is
    # K-paired-broadcast, route through sum_sub_batch to expose K.
    if isinstance(view, SubBatchView) and not keepdim and len(dn) == 1:
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        axis = dn[0] - sb_start
        if w.k_pairing and any(
            p == axis and w.k_state[i] == "broadcast" for i, p in enumerate(w.k_pairing)
        ):
            return sum_sub_batch(w, axis)
    if isinstance(view, SubBatchView) and w.sub_batch_state:
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        reduced_sb_rel = [d - sb_start for d in dn]
        if any(w.sub_batch_state[i] == "broadcast" for i in reduced_sb_rel):
            w = w.materialize()
    data = torch.sum(w.data, dim=dn, keepdim=keepdim)
    new_sb = w.sub_batch_ndim
    new_state, new_meta = w.sub_batch_state, w.sub_batch_meta
    new_k_pairing = w.k_pairing
    if isinstance(view, SubBatchView) and not keepdim:
        new_sb -= len(dn)
        if w.sub_batch_state:
            sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
            from neml2.types._base import drop_sub_batch_state_axes  # noqa: PLC0415

            new_state, new_meta = drop_sub_batch_state_axes(
                w.sub_batch_state, w.sub_batch_meta, [d - sb_start for d in dn]
            )
        # Renumber K_pairing to drop pairings to reduced axes + shift down
        # any pairing to a surviving axis above a dropped one.
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        dropped = sorted({d - sb_start for d in dn})

        def _renumber(p: int | None) -> int | None:
            if p is None or p in dropped:
                return None if (p is None or p in dropped) else p
            # Use builtins.sum (this module shadows sum with the typed version).
            import builtins  # noqa: PLC0415

            shift = builtins.sum(1 for d in dropped if d < p)
            return p - shift

        new_k_pairing = tuple(_renumber(p) for p in w.k_pairing)
    return w._rewrap(
        data,
        sub_batch_ndim=new_sb,
        sub_batch_state=new_state,
        sub_batch_meta=new_meta,
        k_ndim=w.k_ndim,
        k_state=w.k_state,
        k_pairing=new_k_pairing,
    )


# ---- V2P-4 primitives: fullify + positional sum_sub_batch ----


def _base_ndim_of(w: TensorWrapper) -> int:
    """Read base_ndim from either a fixed-base ``TensorWrapper`` subclass
    (``ClassVar BASE_NDIM``) or the dynamic-base :class:`~neml2.types.Tensor`
    (instance ``base_ndim``). Avoids leaking the dynamic-vs-fixed distinction
    into the primitives that consume both kinds of wrapper."""
    base_ndim = getattr(type(w), "BASE_NDIM", None)
    if base_ndim is not None:
        return int(base_ndim)
    # Dynamic-base ``Tensor`` carries ``base_ndim`` as an instance attribute.
    return int(cast(int, getattr(w, "base_ndim")))  # noqa: B009


def fullify(w: _TW) -> _TW:
    """Materialize every K-paired broadcast axis to its enumerated form.

    For each K axis ``i`` where ``k_state[i] == "broadcast"`` and
    ``k_pairing[i] is not None``:

    - Expand K axis ``i`` from size 1 to size ``N`` (the paired sub axis's
      extent).
    - Set values to the eye diagonal on (K_i, sub_pair): value at
      ``(K_i=k, ..., sub_pair=g)`` becomes ``original[K_i=0, ..., sub_pair=g] *
      delta(k==g)``.
    - Demote ``k_state[i]`` from ``"broadcast"`` to ``"full"``.
    - Clear ``k_pairing[i]`` to ``None``.

    No-op when no K axis is paired-broadcast — returns ``w`` unchanged.

    Called by cross-mixing leaves that contract a paired sub axis, so the
    contraction sees per-site rows instead of a size-1 placeholder.
    Living in ``neml2.types.functions`` per CLAUDE.md rule 2 (typed
    primitives belong in ``types/``).
    """
    if not w.k_state:
        return w
    targets = [
        i
        for i, (s, p) in enumerate(zip(w.k_state, w.k_pairing, strict=True))
        if s == "broadcast" and p is not None
    ]
    if not targets:
        return w
    # Build the eye-diagonal expansion for each (K_i, sub_pair) pair.
    # Work axis by axis, materialising one paired-broadcast K at a time.
    out = w
    for i in targets:
        sub_axis = out.k_pairing[i]
        assert sub_axis is not None
        # Logical extent N comes from the paired sub axis.
        sub_start = out.data.ndim - _base_ndim_of(out) - out.sub_batch_ndim
        sub_data_axis = sub_start + sub_axis
        # sub_batch_shape may be 1 in storage if the paired sub axis is
        # itself broadcast — recover logical via sub_batch_meta.
        if out.sub_batch_state and out.sub_batch_state[sub_axis] == "broadcast":
            N = int(out.sub_batch_meta[sub_axis])
        else:
            N = int(out.data.shape[sub_data_axis])
        eye_pos = torch.arange(N, device=out.device)
        # Build the indicator: shape (N, 1, ..., N, 1, ...) where the first
        # axis is K_i and another axis at sub_data_axis is sub_pair.
        # We multiply onto a broadcasted copy of `out.data`.
        # 1) Expand K axis i from 1 -> N.
        target_shape = list(out.data.shape)
        target_shape[i] = N
        # Materialize the sub axis too if it's currently size-1 broadcast.
        if target_shape[sub_data_axis] == 1:
            target_shape[sub_data_axis] = N
        data = out.data.expand(target_shape).contiguous()
        # Multiply by the eye mask on (K_i, sub_pair).
        mask_shape = [1] * data.ndim
        mask_shape[i] = N
        mask_shape[sub_data_axis] = N
        # eye_pos[k] == eye_pos[g] -> identity matrix when reshaped.
        mask = (
            (eye_pos.reshape([N] + [1] * (data.ndim - 1 - i) + [1] * 0))
            == eye_pos.reshape([1] * sub_data_axis + [N] + [1] * (data.ndim - sub_data_axis - 1))
        ).to(out.dtype)
        # Build the proper mask via broadcasting.
        ax0 = [1] * data.ndim
        ax0[i] = N
        eye_k = eye_pos.reshape(ax0)
        ax1 = [1] * data.ndim
        ax1[sub_data_axis] = N
        eye_s = eye_pos.reshape(ax1)
        mask = (eye_k == eye_s).to(out.dtype)
        new_data = data * mask
        # Update sub_batch_state for the materialised sub axis.
        new_sb_state: tuple[SubBatchStateFlag, ...] = out.sub_batch_state
        new_sb_meta = out.sub_batch_meta
        if new_sb_state and new_sb_state[sub_axis] == "broadcast":
            new_sb_state = tuple(
                cast(SubBatchStateFlag, "full" if j == sub_axis else s)
                for j, s in enumerate(new_sb_state)
            )
        # Update K state/pairing for the materialised K axis.
        new_k_state: tuple[KStateFlag, ...] = tuple(
            cast(KStateFlag, "full" if j == i else s) for j, s in enumerate(out.k_state)
        )
        new_k_pairing = tuple(None if j == i else p for j, p in enumerate(out.k_pairing))
        out = out._rewrap(
            new_data,
            sub_batch_ndim=out.sub_batch_ndim,
            sub_batch_state=new_sb_state,
            sub_batch_meta=new_sb_meta,
            k_ndim=out.k_ndim,
            k_state=new_k_state,
            k_pairing=new_k_pairing,
        )
    return out


def sum_sub_batch(t, axis: int):
    """Sum over the sub_batch axis at positional index ``axis``.

    If a K axis is paired with the reduced sub axis, EXPOSE: promote that
    K axis from broadcast to full (size = paired sub extent), drop the K
    pairing, drop the sub axis. Otherwise: regular ``torch.sum`` over the
    sub axis.

    Replaces the v2-style ``sum_sub_batch_by_label`` -- callers now
    address the axis by position. Works on both :class:`TensorWrapper`
    instances and dynamic-base :class:`Tensor` instances.
    """
    sb_ndim = t.sub_batch_ndim
    if axis < 0:
        axis += sb_ndim
    if axis < 0 or axis >= sb_ndim:
        raise IndexError(f"sum_sub_batch: axis {axis} out of range for sub_batch_ndim={sb_ndim}")
    # Locate the matching K-paired axis, if any.
    k_paired_axis: int | None = None
    if t.k_pairing:
        for ki, p in enumerate(t.k_pairing):
            if p == axis and t.k_state[ki] == "broadcast":
                k_paired_axis = ki
                break
    # Resolve absolute axis on data.
    if isinstance(t, TensorWrapper):
        base_ndim = type(t).BASE_NDIM
        sb_start = t.data.ndim - base_ndim - sb_ndim
    else:
        base_ndim = t.base_ndim
        sb_start = t.batch_ndim + getattr(t, "k_ndim", 0)
    data_axis = sb_start + axis
    sb_state = getattr(t, "sub_batch_state", ())
    if k_paired_axis is None:
        # Plain reduction over the sub axis.
        # Materialise broadcast sub axis first if needed.
        if sb_state and sb_state[axis] == "broadcast":
            if isinstance(t, TensorWrapper):
                t = t.materialize()
            else:
                # Tensor analogue: just expand the size-1 axis.
                pass
        new_data = torch.sum(t.data, dim=data_axis)
        from neml2.types._base import drop_sub_batch_state_axes  # local to avoid import cycle

        new_state, new_meta = drop_sub_batch_state_axes(t.sub_batch_state, t.sub_batch_meta, [axis])
        # K_pairing renumber: any K axis paired with a sub axis > `axis`
        # shifts down by 1 after the drop; pairing with `axis` itself becomes
        # None (no expose path here, just dropping the relationship).
        new_k_pairing: tuple[int | None, ...] = tuple(
            None if p == axis else (p - 1 if (p is not None and p > axis) else p)
            for p in t.k_pairing
        )
        if isinstance(t, TensorWrapper):
            return t._rewrap(
                new_data,
                sub_batch_ndim=sb_ndim - 1,
                sub_batch_state=new_state,
                sub_batch_meta=new_meta,
                k_ndim=t.k_ndim,
                k_state=t.k_state,
                k_pairing=new_k_pairing,
            )
        return Tensor(
            new_data,
            t.batch_ndim,
            sb_ndim - 1,
            k_ndim=t.k_ndim,
            k_state=t.k_state,
            k_pairing=new_k_pairing,
        )
    # Expose path: K_paired broadcast becomes full, drop sub axis + pairing.
    # The paired-broadcast K axis is size 1 in data; the sub axis is the
    # paired one. The result tangent K axis enumerates the per-site values.
    # Materialise the K paired axis to size N, copying the per-site values.
    # The original broadcast K stores per-site value at its data position;
    # squeeze the sub axis (which carries per-site values) into the K axis.
    # Concretely: data has shape (..., K_i=1, ..., sub_axis=N or 1, ...).
    # If sub axis is full, the K row at site g equals data[..., g, ...].
    # If sub axis is also broadcast (placeholder), values are identical per
    # site; sum is N * value, expose is the value itself per K row.
    # Simplest: materialise sub axis, squeeze K paired axis (size 1), insert
    # back at K position with sub axis content.
    # Materialise sub axis to size N if currently broadcast.
    if t.sub_batch_state and t.sub_batch_state[axis] == "broadcast":
        if isinstance(t, TensorWrapper):
            t = t.materialize()
        # else: Tensor case
    # Swap K_paired (size 1) with sub (size N), then squeeze the size-1
    # axis at sub's old position.
    swapped = t.data.transpose(k_paired_axis, data_axis)
    # swapped has size N at k_paired_axis and size 1 at data_axis.
    new_data = swapped.squeeze(data_axis)
    from neml2.types._base import drop_sub_batch_state_axes

    new_sb_state, new_sb_meta = drop_sub_batch_state_axes(
        t.sub_batch_state, t.sub_batch_meta, [axis]
    )
    new_k_state: tuple[KStateFlag, ...] = tuple(
        cast(KStateFlag, "full" if i == k_paired_axis else s) for i, s in enumerate(t.k_state)
    )
    # K_pairing: exposed K axis (k_paired_axis) becomes None; other K axes
    # paired with sub axes > `axis` shift down by 1.
    new_k_pairing = tuple(
        None
        if i == k_paired_axis
        else (None if p == axis else (p - 1 if (p is not None and p > axis) else p))
        for i, p in enumerate(t.k_pairing)
    )
    if isinstance(t, TensorWrapper):
        return t._rewrap(
            new_data,
            sub_batch_ndim=sb_ndim - 1,
            sub_batch_state=new_sb_state,
            sub_batch_meta=new_sb_meta,
            k_ndim=t.k_ndim,
            k_state=new_k_state,
            k_pairing=new_k_pairing,
        )
    return Tensor(
        new_data,
        t.batch_ndim,
        sb_ndim - 1,
        k_ndim=t.k_ndim,
        k_state=new_k_state,
        k_pairing=new_k_pairing,
    )


def mean(view: DynamicBatchView[_TW] | SubBatchView[_TW], dim: int = 0) -> _TW:
    """Mean over one axis of a region view.

    ``view`` must be ``t.dynamic_batch`` or ``t.sub_batch``. Always
    collapses the axis (no ``keepdim``); reducing a sub-batch axis drops
    ``sub_batch_ndim`` by 1. Returns the same wrapper type as the view's
    underlying wrapper.

    Like :func:`sum`, materialises any ``"broadcast"`` sub-batch axis
    before reducing so the per-site mean counts every site.
    """
    w, start, end = _reduce_view_bounds(view, "mean")
    d = _normalize_dim(dim, start, end)
    # Expose path: K-paired-broadcast sub axis -> sum_sub_batch then /N.
    if isinstance(view, SubBatchView):
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        axis = d - sb_start
        if w.k_pairing and any(
            p == axis and w.k_state[i] == "broadcast" for i, p in enumerate(w.k_pairing)
        ):
            # Logical extent N from the paired sub axis (broadcast meta).
            N = (
                int(w.sub_batch_meta[axis])
                if w.sub_batch_state and w.sub_batch_state[axis] == "broadcast"
                else int(w.data.shape[d])
            )
            summed = cast(_TW, sum_sub_batch(w, axis))
            return summed._rewrap(
                summed.data / N,
                sub_batch_ndim=summed.sub_batch_ndim,
            )
    if isinstance(view, SubBatchView) and w.sub_batch_state:
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        rel = d - sb_start
        if w.sub_batch_state[rel] == "broadcast":
            w = w.materialize()
    new_sb = w.sub_batch_ndim - (1 if isinstance(view, SubBatchView) else 0)
    # K_pairing renumber for sub axis drop (analogous to sum's path).
    new_k_pairing = w.k_pairing
    if isinstance(view, SubBatchView):
        sb_start = w.data.ndim - w.BASE_NDIM - w.sub_batch_ndim
        dropped_rel = d - sb_start
        new_k_pairing = tuple(
            None if (p is None or p == dropped_rel) else (p - 1 if p > dropped_rel else p)
            for p in w.k_pairing
        )
    return w._rewrap(
        torch.mean(w.data, dim=d),
        sub_batch_ndim=new_sb,
        k_ndim=w.k_ndim,
        k_state=w.k_state,
        k_pairing=new_k_pairing,
    )


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


@overload
def stack(
    views: Sequence[SubBatchView[_TW] | DynamicBatchView[_TW]],
    dim: int = 0,
) -> _TW: ...
@overload
def stack(views: Sequence[_TensorRegionView], dim: int = 0) -> Tensor: ...
def stack(views, dim: int = 0):
    """Stack values along a NEW axis inside a chosen region view.

    Dispatches on view type:

    - :class:`~neml2.types._base.DynamicBatchView` /
      :class:`~neml2.types._base.SubBatchView` over a fixed-base
      :class:`~neml2.types.TensorWrapper` -> typed wrapper output.
    - :class:`~neml2.types.tensor._RegionView` over a dynamic-base
      :class:`~neml2.types.Tensor` -> Tensor output.

    All views must be the same kind over operands sharing region ndims
    and (apart from the new axis) data shape.

    Example
    -------
    >>> v0 = Vec.fill(6.0, 4.0, 0.0)
    >>> v1 = Vec.fill(8.0, 5.0, 0.0)
    >>> stack([v0.dynamic_batch, v1.dynamic_batch]).data.shape
    torch.Size([2, 3])
    """
    if not views:
        raise ValueError("stack: views must be non-empty")
    first = views[0]
    if isinstance(first, _TensorRegionView):
        return _tensor_stack(views, dim=dim)
    if not isinstance(first, DynamicBatchView | SubBatchView):
        raise TypeError(
            f"stack: views must be t.dynamic_batch, t.sub_batch, or a Tensor "
            f"region view, got {type(first).__name__}"
        )
    region_type = type(first)
    first_w = first._w
    for v in views[1:]:
        if type(v) is not region_type:
            raise TypeError(
                f"stack: heterogeneous view types {region_type.__name__} vs {type(v).__name__}"
            )
        w = v._w
        if type(w) is not type(first_w):
            raise TypeError(
                f"stack: heterogeneous wrapper types {type(first_w).__name__} vs {type(w).__name__}"
            )
        if w.sub_batch_ndim != first_w.sub_batch_ndim:
            raise ValueError(
                f"stack: mismatched sub_batch_ndim {first_w.sub_batch_ndim} vs {w.sub_batch_ndim}"
            )
        if w.data.shape != first_w.data.shape:
            raise ValueError(
                f"stack: mismatched data shapes "
                f"{tuple(first_w.data.shape)} vs {tuple(w.data.shape)}"
            )
    axis = first._resolve_insert_dim(dim)
    new_data = torch.stack([v._w.data for v in views], dim=axis)
    new_sb = first._new_sub_batch_ndim(axes_added=1)
    return first_w._rewrap(new_data, sub_batch_ndim=new_sb)


@overload
def cat(
    views: Sequence[SubBatchView[_TW] | DynamicBatchView[_TW] | BatchView[_TW] | BaseView[_TW]],
    dim: int = -1,
) -> _TW: ...
@overload
def cat(views: Sequence[_TensorBaseView], dim: int = -1) -> Tensor: ...
def cat(views, dim: int = -1):
    """Concatenate wrappers along a region-relative axis.

    Each element of ``views`` must be the same region kind over wrappers
    that share ``batch_ndim`` / ``sub_batch_ndim``. The cat-axis size is
    the only thing allowed to vary. Works uniformly on dynamic-base
    :class:`~neml2.types.Tensor` views (``.batch`` / ``.sub_batch`` /
    ``.base``) and on static-base :class:`~neml2.types.TensorWrapper`
    region views from the same axis convention (the static-base
    ``.base`` is fixed and not in the cat-able set, but ``.batch`` /
    ``.dynamic_batch`` / ``.sub_batch`` work).

    Example
    -------
    >>> a = Tensor.zeros(batch_shape=(2,), base_shape=(3,))
    >>> b = Tensor.zeros(batch_shape=(2,), base_shape=(4,))
    >>> cat([a.base, b.base]).data.shape
    torch.Size([2, 7])
    """
    if not views:
        raise ValueError("cat: views must be non-empty")
    first = views[0]
    # Dispatch by which family of region view the caller passed:
    # dynamic-base views expose ``._t``; static-base views expose
    # ``._w``. Both produce wrappers via the same data-cat semantics;
    # we just thread through the corresponding accessor / rewrap to
    # preserve K / sub_batch metadata on the result.
    if hasattr(first, "_t"):
        return _tensor_cat(views, dim=dim)
    head = first._w
    head_type = type(first)
    for v in views[1:]:
        if type(v) is not head_type:
            raise TypeError(
                f"cat: heterogeneous region views {head_type.__name__} vs {type(v).__name__}"
            )
        if type(v._w) is not type(head):
            raise TypeError(
                f"cat: mixed wrapper types {type(head).__name__} vs {type(v._w).__name__}"
            )
        if v._w.sub_batch_ndim != head.sub_batch_ndim:
            raise ValueError(
                f"cat: mismatched sub_batch_ndim {head.sub_batch_ndim} vs {v._w.sub_batch_ndim}"
            )
    axis = first._resolve_dim(dim)
    new_data = torch.cat([v._w.data for v in views], dim=axis)
    return head._rewrap(new_data, sub_batch_ndim=head.sub_batch_ndim)


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
    state, meta = combine_sub_batch_state(aa, bb)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, bb)
    return aa._rewrap(
        op(aa.data, b_data),
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    state, meta = combine_sub_batch_state(cc, aa, bb)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(cc, aa, bb)
    return aa._rewrap(
        torch.where(c_data, aa.data, bb.data),
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


def _gather_along_last(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``table[..., idx]`` with broadcast over the leading batch.

    ``table.shape == (*table_batch, N)``, ``idx.shape == (*idx_batch,)``.
    Returns shape ``broadcast(table_batch, idx_batch)``.

    The naive ``table[..., idx]`` uses fancy indexing along the last axis
    and produces ``table_batch + idx_batch`` -- an outer product. That's
    only correct when ``table_batch == ()`` and ``idx`` shares its leading
    axes with no parent batch. For per-sample tables (``ordinate`` whose
    leading dim is the parameter's batch axis) we want the gather to
    broadcast leading dims against ``idx_batch`` rather than concatenate
    them. ``torch.gather`` with explicit broadcast is the safe primitive.

    When ``table_batch`` and ``idx`` are both scalar (``broadcast_shapes
    -> ()``), there's no broadcast work to do -- ``table[..., idx]`` falls
    through to plain index-along-last and returns a 0-d tensor.
    """
    common = torch.broadcast_shapes(table.shape[:-1], idx.shape)
    if not common:
        return table[..., idx]
    table_b = table.expand(*common, table.shape[-1])
    idx_b = idx.expand(*common).unsqueeze(-1)
    return torch.gather(table_b, -1, idx_b).squeeze(-1)


def linear_interpolation(argument: Scalar, abscissa: Scalar, ordinate: Scalar) -> Scalar:
    """Piecewise-linear interpolation of a Scalar table.

    ``abscissa`` and ``ordinate`` may carry their own leading batch (e.g.
    per-sample interpolation tables introduced by pyzag-style parameter
    calibration); the broadcast-safe gather in :func:`_gather_along_last`
    handles those naturally.
    """
    x = argument.data
    X = abscissa.data
    Y = ordinate.data
    n = X.shape[-1]
    # ``torch.searchsorted`` requires its ``values`` argument to be contiguous
    # for the fast path; otherwise it copies internally and warns once per
    # process. Several callers (chain-rule tangents, broadcasted argument
    # batches) hand us views, so normalize here.
    idx = torch.searchsorted(X, x.contiguous(), right=True).clamp(1, n - 1)
    x1 = _gather_along_last(X, idx - 1)
    x2 = _gather_along_last(X, idx)
    y1 = _gather_along_last(Y, idx - 1)
    y2 = _gather_along_last(Y, idx)
    slope = (y2 - y1) / (x2 - x1)
    return wrap_like(Scalar, y1 + slope * (x - x1), argument)


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
    idx = torch.searchsorted(X, x.contiguous(), right=True).clamp(1, n - 1)
    x1 = _gather_along_last(X, idx - 1)
    x2 = _gather_along_last(X, idx)
    y1 = _gather_along_last(Y, idx - 1)
    y2 = _gather_along_last(Y, idx)
    slope = (y2 - y1) / (x2 - x1)
    [arg_a, dx_a], sb = align_sub_batch(argument, dargument)
    state, meta = combine_sub_batch_state(arg_a, dx_a)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(arg_a, dx_a)
    return Scalar(
        slope * dargument.data,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    idx1 = (
        torch.searchsorted(X1b, x1b.unsqueeze(-1).contiguous(), right=True)
        .squeeze(-1)
        .clamp(1, N1 - 1)
    )
    idx2 = (
        torch.searchsorted(X2b, x2b.unsqueeze(-1).contiguous(), right=True)
        .squeeze(-1)
        .clamp(1, N2 - 1)
    )

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
    # Pyright loses _TW precision through the chained ``_TW + _TW * Scalar``
    # algebra (TensorWrapper's abstract operator stubs return TensorWrapper).
    # The runtime concrete type is correct; cast to keep callers cast-free.
    return cast(_TW, Y00 + c1 * xi + c2 * eta + c3 * (xi * eta))


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
    # See the note in :func:`bilinear_interpolation` for why the cast is needed.
    return cast(_TW, dP_dx1), cast(_TW, dP_dx2)


def sqrt(s: Scalar) -> Scalar:
    # ``torch.sqrt``'s reverse-mode backward saves its OUTPUT (``grad/(2*out)``),
    # which AOTInductor cannot lower under strict + dynamic-batch export
    # (pytorch/pytorch#187907). Off the AD path keep the dedicated sqrt
    # instruction; on it use the input-recompute Function. See ``_ad_safe``.
    return s._rewrap(_ad_safe(s.data, torch.sqrt, _sqrt_ad), sub_batch_ndim=s.sub_batch_ndim)


def exp(s: Scalar) -> Scalar:
    # ``torch.exp``'s reverse-mode backward saves its OUTPUT; recompute from the
    # input on the AD path so AOTI can lower it (pytorch/pytorch#187907).
    return s._rewrap(_ad_safe(s.data, torch.exp, _exp_ad), sub_batch_ndim=s.sub_batch_ndim)


def tanh(s: Scalar) -> Scalar:
    """Hyperbolic tangent. Matches ``neml2::tanh(const Scalar&)``."""
    # ``torch.tanh``'s reverse-mode backward saves its OUTPUT; recompute from the
    # input on the AD path so AOTI can lower it (pytorch/pytorch#187907).
    return s._rewrap(_ad_safe(s.data, torch.tanh, _tanh_ad), sub_batch_ndim=s.sub_batch_ndim)


def reciprocal(s: Scalar) -> Scalar:
    """Element-wise reciprocal ``1/s``. Matches ``neml2::reciprocal``.

    ``torch.reciprocal``'s reverse-mode backward saves its OUTPUT
    (``-grad*out**2``), the same AOTI lowering hazard as ``sqrt`` / ``exp``
    (pytorch/pytorch#187907). Off the AD path keep the plain instruction; on it
    use the input-recompute Function (backward ``-x**-2``).
    """
    return s._rewrap(
        _ad_safe(s.data, torch.reciprocal, _reciprocal_ad), sub_batch_ndim=s.sub_batch_ndim
    )


def cosh(s: Scalar) -> Scalar:
    """Hyperbolic cosine. Matches ``neml2::cosh(const Scalar&)``."""
    return s._rewrap(torch.cosh(s.data), sub_batch_ndim=s.sub_batch_ndim)


def sinh(s: Scalar) -> Scalar:
    """Hyperbolic sine. Matches ``neml2::sinh(const Scalar&)``."""
    return s._rewrap(torch.sinh(s.data), sub_batch_ndim=s.sub_batch_ndim)


def log(s: Scalar) -> Scalar:
    """Natural logarithm. Matches ``neml2::log(const Scalar&)``."""
    return s._rewrap(torch.log(s.data), sub_batch_ndim=s.sub_batch_ndim)


def log10(s: Scalar) -> Scalar:
    """Base-10 logarithm. Matches ``neml2::log10(const Scalar&)``."""
    return s._rewrap(torch.log10(s.data), sub_batch_ndim=s.sub_batch_ndim)


def _broadcast_exponent(a: TensorWrapper, n: float | int | Scalar) -> torch.Tensor:
    """Lift the exponent to a tensor right-aligned with ``a.data``."""
    if isinstance(n, Scalar):
        n_data = n.data
        # Pad with trailing size-1 axes so n.data right-aligns with a.data
        # through the typed-tensor base axes. Sub-batch left-alignment is
        # implicit in torch's broadcast rules.
        for _ in range(a.BASE_NDIM):
            n_data = n_data.unsqueeze(-1)
        return n_data
    # Python scalar exponent: lift to a 0-d tensor (cheap, no expand).
    # ``torch.pow`` inside the op broadcasts the 0-d against any shape.
    return a.data.new_full((), float(n))


def pow(a: _TW, n: float | int | Scalar) -> _TW:  # noqa: A001
    """Element-wise power. Calls ``torch.pow`` directly.

    Transparent to Inductor: fuses with surrounding pointwise ops. The
    sensible default for new leaves. If profiling on a representative
    benchmark shows the pow being recomputed redundantly inside a fused
    reduction kernel (Triton can do this even when the reduction itself
    looks small), switch the call site to :func:`opaque_pow` and re-time.
    """
    return a._rewrap(torch.pow(a.data, _broadcast_exponent(a, n)), sub_batch_ndim=a.sub_batch_ndim)


def opaque_pow(a: _TW, n: float | int | Scalar) -> _TW:
    """Element-wise power routed through the ``neml2::opaque_pow`` custom op.

    Inductor treats the custom op as a fusion barrier, which prevents the
    pow from being inlined into a downstream reduction's per-output
    recompute. Profiled wins so far:

    * ``PowerLawSlipRule`` -> ``SumSlipRates`` -> K-tangent (scpcoup CUDA
      B=8192: 4.95 s without the barrier vs 2.14 s with it, 2.3x; across
      the CP suite 2-3x).
    * ``PerzynaPlasticFlowRate`` (isoharden CUDA B=8192: 117 ms without
      the barrier vs 102 ms with it, 1.15x).

    The barrier costs a real fusion opportunity at small batches -- the
    same isoharden case at B=1024 is 90 ms transparent vs 100 ms opaque,
    so opaque is a net loss when the reduction redundancy doesn't
    dominate. ``opaque_pow`` is leaf-specific opt-in: profile the leaf,
    pick whichever is faster on the batches you care about.
    """
    return a._rewrap(
        torch.ops.neml2.opaque_pow(a.data, _broadcast_exponent(a, n)),
        sub_batch_ndim=a.sub_batch_ndim,
    )


# ---- Cross-type products on SR2 ----


def outer(a: SR2, b: SR2) -> SSR4:
    """Tensor product ``a ⊗ b`` of two SR2s, producing an SSR4 in Mandel packing."""
    [aa, bb], sb = align_sub_batch(a, b)
    state, meta = combine_sub_batch_state(aa, bb)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, bb)
    return SSR4(
        aa.data.unsqueeze(-1) * bb.data.unsqueeze(-2),
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    state, meta = combine_sub_batch_state(aa, bb)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, bb)
    return Scalar(
        out,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


# ---- Second-order tensor determinant / inverse ----
#
# Hand-unrolled 3x3 cofactor expansion, mirroring v2's ``neml2::det`` /
# ``neml2::inv`` (``src/neml2/tensors/functions/det.cxx`` and ``inv.cxx``).
# ``torch.linalg.det`` / ``torch.linalg.inv`` are accurate but pay two costs
# the hand-unroll avoids:
#
# 1. **Per-call dispatch.** LAPACK / MAGMA carry a fixed cost per call that
#    swamps the actual 27-FLOP cofactor expansion on a 3x3. Under AOTI the
#    Inductor warning ``aten.linalg_inv_ex.default is missing a c-shim
#    implementation, using proxy executor as fallback`` confirms each call
#    bounces through the proxy executor.
# 2. **No Inductor fusion.** The pointwise cofactor formulas fuse with the
#    upstream and downstream pointwise ops; ``torch.linalg.inv`` is a
#    fusion-opaque external kernel.
#
# The SR2 path also avoids the previous ``r2_from_sr2`` round-trip plus
# ``sym(R2(...))`` re-pack -- the symmetric 3x3 has 6 unique cofactors that
# we compute directly on the Mandel-packed input.


def _det_3x3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    i: torch.Tensor,
) -> torch.Tensor:
    """Determinant of the 3x3 ``[[a,b,c],[d,e,f],[g,h,i]]`` via cofactor row 0."""
    return a * (e * i - h * f) - b * (d * i - g * f) + c * (d * h - e * g)


def _det_R2_data(data: torch.Tensor) -> torch.Tensor:
    """Determinant of a full 3x3 R2 (``(..., 3, 3)`` -> ``(...,)``)."""
    a = data[..., 0, 0]
    b = data[..., 0, 1]
    c = data[..., 0, 2]
    d = data[..., 1, 0]
    e = data[..., 1, 1]
    f = data[..., 1, 2]
    g = data[..., 2, 0]
    h = data[..., 2, 1]
    i = data[..., 2, 2]
    return _det_3x3(a, b, c, d, e, f, g, h, i)


def _unpack_sr2(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a Mandel-packed SR2 ``(..., 6)`` into the 6 unique components.

    Layout: ``[A00, A11, A22, sqrt(2)*A12, sqrt(2)*A02, sqrt(2)*A01]``.
    Returns ``(a, b, c, e, f, i)`` matching the v2 naming:

        A = [[a, b, c],
             [b, e, f],
             [c, f, i]]
    """
    a = data[..., 0]
    e = data[..., 1]
    i = data[..., 2]
    f = data[..., 3] * _INV_SQRT2
    c = data[..., 4] * _INV_SQRT2
    b = data[..., 5] * _INV_SQRT2
    return a, b, c, e, f, i


def _det_SR2_data(data: torch.Tensor) -> torch.Tensor:
    """Determinant of a Mandel-packed symmetric 3x3 (``(..., 6)`` -> ``(...,)``)."""
    a, b, c, e, f, i = _unpack_sr2(data)
    return a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c)


def _inv_R2_data(data: torch.Tensor) -> torch.Tensor:
    """Hand-unrolled inverse of a full 3x3 R2 (``(..., 3, 3)`` -> same)."""
    a = data[..., 0, 0]
    b = data[..., 0, 1]
    c = data[..., 0, 2]
    d = data[..., 1, 0]
    e = data[..., 1, 1]
    f = data[..., 1, 2]
    g = data[..., 2, 0]
    h = data[..., 2, 1]
    i = data[..., 2, 2]
    inv_det = 1.0 / _det_3x3(a, b, c, d, e, f, g, h, i)
    # adjugate / det. Pre-multiply each cofactor by 1/det so the final stack
    # produces the inverse directly (one fused divide instead of nine).
    a00 = (e * i - h * f) * inv_det
    a01 = (c * h - b * i) * inv_det
    a02 = (b * f - c * e) * inv_det
    a10 = (f * g - d * i) * inv_det
    a11 = (a * i - c * g) * inv_det
    a12 = (c * d - a * f) * inv_det
    a20 = (d * h - e * g) * inv_det
    a21 = (b * g - a * h) * inv_det
    a22 = (a * e - b * d) * inv_det
    row0 = torch.stack([a00, a01, a02], dim=-1)
    row1 = torch.stack([a10, a11, a12], dim=-1)
    row2 = torch.stack([a20, a21, a22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _inv_SR2_data(data: torch.Tensor) -> torch.Tensor:
    """Hand-unrolled inverse of a Mandel-packed symmetric 3x3 (``(..., 6)`` -> same).

    Output is Mandel-packed in the same layout. No ``r2_from_sr2`` /
    ``sym`` round-trip -- the symmetric inverse has 6 unique entries and
    we re-pack the off-diagonals with their Mandel ``sqrt(2)`` factors
    inline.
    """
    a, b, c, e, f, i = _unpack_sr2(data)
    inv_det = 1.0 / (a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c))
    inv00 = (e * i - f * f) * inv_det
    inv11 = (a * i - c * c) * inv_det
    inv22 = (a * e - b * b) * inv_det
    inv12 = (c * b - a * f) * inv_det
    inv02 = (b * f - c * e) * inv_det
    inv01 = (c * f - b * i) * inv_det
    return torch.stack(
        [inv00, inv11, inv22, inv12 * _SQRT2, inv02 * _SQRT2, inv01 * _SQRT2],
        dim=-1,
    )


def det(A: TensorWrapper) -> Scalar:
    """Determinant of a (..., 3, 3) second-order tensor wrapper.

    Accepts ``R2`` (full 3x3) or ``SR2`` (Mandel-packed). Returns a
    ``Scalar`` over the wrapper's batch + sub-batch axes. Mirrors
    ``neml2::det``.
    """
    if isinstance(A, R2):
        return wrap_like(Scalar, _det_R2_data(A.data), A)
    if isinstance(A, SR2):
        return wrap_like(Scalar, _det_SR2_data(A.data), A)
    raise TypeError(f"det requires R2 or SR2; got {type(A).__name__}")


def inv(A: _TW) -> _TW:
    """Matrix inverse of a (..., 3, 3) second-order tensor wrapper.

    For ``R2`` returns the full inverse as an ``R2``; for ``SR2`` returns
    the inverse repacked into Mandel form (the inverse of a symmetric
    tensor is symmetric). Mirrors ``neml2::inv``.
    """
    if isinstance(A, R2):
        return A._rewrap(_inv_R2_data(A.data), sub_batch_ndim=A.sub_batch_ndim)
    if isinstance(A, SR2):
        return A._rewrap(_inv_SR2_data(A.data), sub_batch_ndim=A.sub_batch_ndim)
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
# (9, 6) projection" form was tried for ``r2_from_sr2`` / ``r2_from_wr2`` /
# ``sym`` / ``skew``. It improved CPU wall-time by ~10 % but regressed CUDA wall-time
# by ~9 % at small batch (B ≤ 1,024): each matmul launches a separate
# kernel that Inductor cannot fuse with surrounding pointwise ops, while
# the explicit select+stack chains do fuse into a single Triton kernel
# via Inductor's pointwise scheduler. Sticking with the pointwise form
# is the CUDA-favourable choice. A hand-rolled ``unrolled_matmul`` that
# emits fusible pointwise ops (instead of dispatching to BLAS / Triton's
# matmul kernel) is a candidate future optimisation that could unify the
# CPU and CUDA wins — see the note on bench results in
# ``python/neml2/native/README.md``.


def matmul_3x3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3×3 matrix product on the trailing two axes; device-dispatched.

    On **CUDA**, dispatches to a hand-rolled pointwise implementation
    (27 multiplies + 18 adds + a final stack). Equivalent to ``a @ b``
    semantically, but emits explicit pointwise ops instead of a
    ``torch.matmul`` kernel — so Inductor / Triton can fuse the whole
    rotation chain (e.g. ``R @ S @ R.T``) into a single kernel together
    with surrounding pointwise ops. Eliminates the intermediate-tensor
    HBM round trip a separate cuBLAS dispatch would force. Measured
    win on ``scpdecoup`` CUDA at B=4096: 2.05× end-to-end speedup.

    On **CPU**, dispatches to ``torch.matmul`` (MKL). MKL has highly
    tuned small-matrix kernels for fixed sizes like 3×3; the hand-rolled
    version regresses CPU by ~1.5× because the explicit stack-of-stack
    construction is a fusion barrier under the C++ Inductor backend
    while MKL's specialised path isn't. The CPU-vs-CUDA split mirrors
    the long-standing note in this module about ``r2_from_sr2`` (same
    underlying tension: fusion-friendly pointwise vs vendor BLAS).

    The device branch is on the input's ``device.type`` attribute,
    which torch.export evaluates statically when tracing with a sample
    input on the target device — only the chosen branch lands in the
    AOTI artifact, so there's no runtime dispatch overhead.
    """
    if a.device.type != "cuda":
        return a @ b
    a00, a01, a02 = a[..., 0, 0], a[..., 0, 1], a[..., 0, 2]
    a10, a11, a12 = a[..., 1, 0], a[..., 1, 1], a[..., 1, 2]
    a20, a21, a22 = a[..., 2, 0], a[..., 2, 1], a[..., 2, 2]
    b00, b01, b02 = b[..., 0, 0], b[..., 0, 1], b[..., 0, 2]
    b10, b11, b12 = b[..., 1, 0], b[..., 1, 1], b[..., 1, 2]
    b20, b21, b22 = b[..., 2, 0], b[..., 2, 1], b[..., 2, 2]
    row0 = torch.stack(
        [
            a00 * b00 + a01 * b10 + a02 * b20,
            a00 * b01 + a01 * b11 + a02 * b21,
            a00 * b02 + a01 * b12 + a02 * b22,
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            a10 * b00 + a11 * b10 + a12 * b20,
            a10 * b01 + a11 * b11 + a12 * b21,
            a10 * b02 + a11 * b12 + a12 * b22,
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            a20 * b00 + a21 * b10 + a22 * b20,
            a20 * b01 + a21 * b11 + a22 * b21,
            a20 * b02 + a21 * b12 + a22 * b22,
        ],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def matvec_3x3(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """3×3 matrix-vector product on the trailing axes; device-dispatched.

    Equivalent to ``(m @ v.unsqueeze(-1)).squeeze(-1)`` for ``m`` of
    trailing shape ``(3, 3)`` and ``v`` of trailing shape ``(3,)``,
    but the CUDA branch hand-rolls 9 multiplies + 6 adds + a final
    stack-of-3 — same fusion-quality reasoning as :func:`matmul_3x3`.
    The matvec kernel cuBLAS would dispatch is a separate boundary
    that prevents fusion with surrounding pointwise ops; the hand-
    rolled form lets Inductor fold ``R @ vec`` chains together with
    the surrounding ``cross`` / ``+`` / ``where`` operations.

    Used by :func:`jvp_compose` (orientation Newton step's chain
    rule) and any other ``(3, 3) · (3,)`` site. For per-orientation-
    seed scenarios this is on the bottleneck path; the same
    device-dispatched approach as :func:`matmul_3x3` applies.
    """
    if m.device.type != "cuda":
        return (m @ v.unsqueeze(-1)).squeeze(-1)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            m00 * v0 + m01 * v1 + m02 * v2,
            m10 * v0 + m11 * v1 + m12 * v2,
            m20 * v0 + m21 * v1 + m22 * v2,
        ],
        dim=-1,
    )


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
    return wrap_like(R2, torch.stack([row0, row1, row2], dim=-2), s)


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
    return wrap_like(R2, torch.stack([row0, row1, row2], dim=-2), w)


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
    return wrap_like(SR2, torch.stack([a, b, c, yz, xz, xy], dim=-1), t)


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
    return wrap_like(WR2, torch.stack([w0, w1, w2], dim=-1), t)


# ---- Rot ↔ R2 (Euler-Rodrigues mapping) ----


def euler_rodrigues(r: Rot) -> R2:
    """Convert an MRP rotation to its 3x3 rotation matrix.

    Mirrors ``Rot::euler_rodrigues`` in ``src/neml2/tensors/Rot.cxx``::

        R = (1+rr)^-2 * ( (1+rr)^2 * I + 4(1-rr) W + 8 W^2 )

    where $W$ is the skew-symmetric matrix of $r$ (via ``R2::skew``)
    and $rr = ||r||^2$.

    Implementation: closed-form per-element, no ``W @ W`` matmul. Since
    ``W = [[0,-w2,w1],[w2,0,-w0],[-w1,w0,0]]`` is skew, ``W^2`` is symmetric
    with only 6 independent components -- computed below from raw
    ``w0/w1/w2`` to skip the ``aten.bmm`` lowering, which triggers a PyTorch
    Inductor codegen bug under dynamic-batch export (``int_array_0``
    referenced in the generated wrapper without ever being declared --
    see ``benchmark/scpdecoup`` for the failing pattern).
    """
    rr = (r.data * r.data).sum(dim=-1)  # (...,)
    w0, w1, w2 = r.data[..., 0], r.data[..., 1], r.data[..., 2]
    # 1/(1+rr)^2 = inverse of the formula's denominator. Pre-divide every
    # row component once instead of dividing nine times.
    inv = 1.0 / ((1.0 + rr) ** 2)  # (...,)
    a = (1.0 + rr) ** 2 * inv  # (...,) -- identity prefactor; structurally 1
    b = 4.0 * (1.0 - rr) * inv  # (...,) -- W prefactor
    c = 8.0 * inv  # (...,) -- W^2 prefactor
    # W^2 has the closed form (symmetric):
    #   diag    : -(w1^2+w2^2),  -(w0^2+w2^2),  -(w0^2+w1^2)
    #   off-diag: W2[i,j] = w_i * w_j (and = W2[j,i])
    w0w0, w1w1, w2w2 = w0 * w0, w1 * w1, w2 * w2
    w0w1, w0w2, w1w2 = w0 * w1, w0 * w2, w1 * w2
    # R[i,j] = a*delta[i,j] + b*W[i,j] + c*W^2[i,j], all pointwise. No bmm
    # means no aten.bmm-with-dynamic-3x3 lowering, which sidesteps a torch
    # Inductor codegen bug where the generated wrapper references an
    # ``int_array_0`` (the (B,3,3) shape sentinel) without ever declaring it.
    row0 = torch.stack([a - c * (w1w1 + w2w2), -b * w2 + c * w0w1, b * w1 + c * w0w2], dim=-1)
    row1 = torch.stack([b * w2 + c * w0w1, a - c * (w0w0 + w2w2), -b * w0 + c * w1w2], dim=-1)
    row2 = torch.stack([-b * w1 + c * w0w2, b * w0 + c * w1w2, a - c * (w0w0 + w1w1)], dim=-1)
    R_mat = torch.stack([row0, row1, row2], dim=-2)
    return wrap_like(R2, R_mat, r)


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
    # ``num / den`` with a parameter-dependent ``den`` would trip AOTI's
    # saved-output reciprocal lowering (pytorch/pytorch#187907); recompute the
    # reciprocal from the input on the AD path, plain divide off it.
    quot = num * reciprocal_ad(den) if den.requires_grad else num / den
    state, meta = combine_sub_batch_state(aa, bb)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, bb)
    return Rot(
        quot,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    state, meta = combine_sub_batch_state(a1, a2)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(a1, a2)
    return R2(
        res,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    state, meta = combine_sub_batch_state(a1, a2)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(a1, a2)
    return R2(
        res,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    return wrap_like(Rot, out, w)


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
    return wrap_like(R2, out, w)


# ---- Rotation of SR2 / WR2 by an R2 (R * X * R^T projected back) ----
#
# Used in CP: every per-slip Schmid tensor (M : SR2, W : WR2) is rotated by
# the per-crystal orientation matrix R before being summed.


def _rotate_sym(s: SR2, R: R2) -> SR2:
    """$sym(R S R^T)$ packed back to Mandel; the symmetric tensor rotation."""
    [ss, rr], sb = align_sub_batch(s, R)
    S_full = r2_from_sr2(ss).data  # (...,3,3) — sub_batch already aligned with rr
    # Hand-rolled 3×3 matmul → fuses with surrounding pointwise (the
    # SR2 → R2 unpack above and the sym repack below) into a single
    # Inductor kernel. See :func:`matmul_3x3`.
    rotated = matmul_3x3(matmul_3x3(rr.data, S_full), rr.data.transpose(-2, -1))
    state, meta = combine_sub_batch_state(ss, rr)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(ss, rr)
    return sym(
        R2(
            rotated,
            sub_batch_ndim=sb,
            sub_batch_state=state,
            sub_batch_meta=meta,
            k_ndim=_k_ndim,
            k_state=_k_state,
            k_pairing=_k_pairing,
        )
    )


def _rotate_skew(w: WR2, R: R2) -> WR2:
    """$skew(R W R^T)$ packed back to an axial vector."""
    [ww, rr], sb = align_sub_batch(w, R)
    W_full = r2_from_wr2(ww).data
    rotated = matmul_3x3(matmul_3x3(rr.data, W_full), rr.data.transpose(-2, -1))
    state, meta = combine_sub_batch_state(ww, rr)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(ww, rr)
    return skew(
        R2(
            rotated,
            sub_batch_ndim=sb,
            sub_batch_state=state,
            sub_batch_meta=meta,
            k_ndim=_k_ndim,
            k_state=_k_state,
            k_pairing=_k_pairing,
        )
    )


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
    rotated = matmul_3x3(matmul_3x3(dRm, S), Rm.transpose(-2, -1)) + matmul_3x3(
        matmul_3x3(Rm, S), dRm.transpose(-2, -1)
    )
    state, meta = combine_sub_batch_state(ss, RR, dRR)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(ss, RR, dRR)
    return sym(
        R2(
            rotated,
            sub_batch_ndim=sb,
            sub_batch_state=state,
            sub_batch_meta=meta,
            k_ndim=_k_ndim,
            k_state=_k_state,
            k_pairing=_k_pairing,
        )
    )


def _jvp_rotate_skew(w: WR2, R: R2, dR: R2) -> WR2:
    """Pushforward of :func:`rotate` (WR2 overload) w.r.t. $R$ along ``dR``.

    ``rotate(w, R) = skew(R W Rᵀ)`` (linear in $w$); the $R$-direction is
    ``skew(dR W Rᵀ + R W dRᵀ)``.
    """
    [ww, RR, dRR], sb = align_sub_batch(w, R, dR)
    W = r2_from_wr2(ww).data
    Rm = RR.data
    dRm = dRR.data
    rotated = matmul_3x3(matmul_3x3(dRm, W), Rm.transpose(-2, -1)) + matmul_3x3(
        matmul_3x3(Rm, W), dRm.transpose(-2, -1)
    )
    state, meta = combine_sub_batch_state(ww, RR, dRR)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(ww, RR, dRR)
    return skew(
        R2(
            rotated,
            sub_batch_ndim=sb,
            sub_batch_state=state,
            sub_batch_meta=meta,
            k_ndim=_k_ndim,
            k_state=_k_state,
            k_pairing=_k_pairing,
        )
    )


# ---- Rotation of a general (asymmetric) R2 by an R2 ----
#
# Used in CP for the plastic spatial velocity gradient
# ``l^p = Q (sum_i gamma_i d_i (x) n_i) Q^T`` where the per-slip Schmid tensor
# is the full asymmetric outer product (no sym/skew projection).


def _rotate_r2(a: R2, R: R2) -> R2:
    """``R A Rᵀ`` — the full (asymmetric) 3x3 rotation, no projection."""
    [aa, rr], sb = align_sub_batch(a, R)
    rotated = matmul_3x3(matmul_3x3(rr.data, aa.data), rr.data.transpose(-2, -1))
    state, meta = combine_sub_batch_state(aa, rr)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, rr)
    return R2(
        rotated,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    rotated = matmul_3x3(matmul_3x3(dRm, A), Rm.transpose(-2, -1)) + matmul_3x3(
        matmul_3x3(Rm, A), dRm.transpose(-2, -1)
    )
    state, meta = combine_sub_batch_state(aa, RR, dRR)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, RR, dRR)
    return R2(
        rotated,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    the formula becomes ``A_ij · B_kl``. Used by :func:`jvp_rotate` (SSR4 case) to
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


def _rotate_ssr4(T: SSR4, R: R2) -> SSR4:
    """``T'_ijkl = R_im R_jn R_kp R_lq T_mnpq`` performed in Mandel packing.

    Builds the 6x6 Mandel basis rotation ``Q(R)`` and forms $Q T Q^T$;
    matches the C++ ``SSR4::rotate(R2)`` semantics.
    """
    [TT, RR], sb = align_sub_batch(T, R)
    Q = _mandel_basis_matrix(RR.data)
    rotated = Q @ TT.data @ Q.transpose(-2, -1)
    state, meta = combine_sub_batch_state(TT, RR)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(TT, RR)
    return SSR4(
        rotated,
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


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
    [r, dr], _ = align_sub_batch(r, dr)
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
    dR = matmul_3x3(R_mat, skew)  # (K, *batch, 3, 3)
    return wrap_like(R2, dR, dr)


def jvp_exp_map(w: WR2, dw: WR2) -> Rot:
    """Pushforward of :func:`exp_map` (WR2→Rot) along the tangent ``dw``.

    Closed-form rank-1-plus-identity: $dexp_map(w) = a(|w|²) I + b(|w|²) w wᵀ$,
    so the action is $dr = a·dw + b·(w·dw)·w$ — two vector ops, no 3×3
    matrix materialised. $a$ and $b$ are the same scalar coefficients
    :func:`dexp_map` builds the matrix from, with the same Taylor branch
    near ``||w||² ≈ 0`` to avoid the origin singularity.
    """
    # Align so a global tangent against a per-crystal primal doesn't
    # collide the primal's sub-batch axis with the tangent's dyn axis.
    [w, dw], _ = align_sub_batch(w, dw)
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
    return wrap_like(Rot, dr, dw)


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
        [r1a, r2a, dr1a], _ = align_sub_batch(r1, r2, dr1)
        term = matvec_3x3(drotate(r2a, r1a).data, dr1a.data)
        acc = term if acc is None else acc + term
    if dr2 is not None:
        [r1b, r2b, dr2b], _ = align_sub_batch(r1, r2, dr2)
        term = matvec_3x3(drotate_self(r2b, r1b).data, dr2b.data)
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("jvp_compose requires at least one of dr1, dr2")
    src: TensorWrapper = dr1 if dr1 is not None else dr2  # type: ignore[assignment]
    return wrap_like(Rot, acc, src)


def _jvp_rotate_ssr4(T: SSR4, R: R2, dR: R2) -> SSR4:
    """Pushforward of :func:`rotate` (SSR4 overload) w.r.t. $R$ along ``dR``.

    ``rotate(T, R)`` is ``Q(R) T Q(R)ᵀ`` with the 6×6 Mandel rotation $Q$
    quadratic in $R$; the directional derivative is
    ``dQ T Qᵀ + Q T dQᵀ`` with
    $dQ = bilinear(R, dR) + bilinear(dR, R)$. $T$ is held fixed (the
    parameter), so only the $R$-dependence is pushed forward. ``dR`` is a
    leading-K ``R2`` tangent; ``_mandel_basis_bilinear`` broadcasts $K$.
    """
    [TT, RR, dRR], _ = align_sub_batch(T, R, dR)
    Q = _mandel_basis_matrix(RR.data)  # (*batch, 6, 6)
    dQ = _mandel_basis_bilinear(RR.data, dRR.data) + _mandel_basis_bilinear(dRR.data, RR.data)
    T_d = TT.data  # (*batch, 6, 6)
    dTrot = dQ @ T_d @ Q.transpose(-2, -1) + Q @ T_d @ dQ.transpose(-2, -1)
    return wrap_like(SSR4, dTrot, dRR)


# ---- Unified rotate / jvp_rotate entry points ----
#
# The public surface is three names, each overloaded on the operand type. The
# underlying ``_rotate_*`` / ``_jvp_rotate_*`` kernels
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


# ---- Vec helpers ----


def _as_raw(value: TensorWrapper | Tensor | torch.Tensor) -> torch.Tensor:
    """Return the underlying raw ``torch.Tensor`` for a wrapper or
    pass-through for an already-raw tensor. Used by :func:`equal` and
    :func:`allclose` so callers outside ``neml2/types/`` never need
    ``.data``. Centralising the unwrap inside the types package keeps
    the wrapper-discipline rule (CLAUDE.md) intact.
    """
    if isinstance(value, (TensorWrapper, Tensor)):
        return value.data
    return value


def equal(
    a: TensorWrapper | Tensor | torch.Tensor,
    b: TensorWrapper | Tensor | torch.Tensor,
) -> bool:
    """Exact element-wise equality across two values that may each be a
    typed wrapper, a :class:`~neml2.types.Tensor`, or a raw
    ``torch.Tensor``. Returns ``True`` iff the underlying storage tensors
    are identical (same shape, same dtype, every element equal).
    """
    return torch.equal(_as_raw(a), _as_raw(b))


def allclose(
    a: TensorWrapper | Tensor | torch.Tensor,
    b: TensorWrapper | Tensor | torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    """Approximate element-wise equality across two values that may each
    be a typed wrapper, a :class:`~neml2.types.Tensor`, or a raw
    ``torch.Tensor``. Mirrors ``torch.allclose`` with the same default
    tolerances.
    """
    return torch.allclose(_as_raw(a), _as_raw(b), rtol=rtol, atol=atol, equal_nan=equal_nan)


def vec_component(v: Vec, i: int) -> Scalar:
    """Extract the ``i``-th Scalar component of a ``Vec`` (i in 0, 1, 2).

    Mirrors the C++ ``Vec::operator()(int)`` slot access used by leaves like
    ``VecComponents`` that decompose a Vec into per-axis Scalars. Preserves
    sub-batch metadata; works inside a leaf's forward without dropping out
    of wrapper algebra.
    """
    if i < 0 or i > 2:
        raise IndexError(f"vec_component index {i} out of range [0, 3)")
    return wrap_like(Scalar, v.data[..., i], v)


def vec_from_scalars(s0: Scalar, s1: Scalar, s2: Scalar) -> Vec:
    """Assemble a ``Vec`` from three ``Scalar`` components.

    Mirrors the C++ ``Vec::fill(Scalar, Scalar, Scalar)`` factory: stacks the
    three Scalar values along a fresh trailing axis to produce a ``(..., 3)``
    ``Vec``. All three inputs must share dtype/device; sub-batch alignment
    flows through :func:`align_sub_batch` so per-sub-batch and global Scalars
    combine cleanly.
    """
    [aa, bb, cc], sb = align_sub_batch(s0, s1, s2)
    state, meta = combine_sub_batch_state(aa, bb, cc)
    _k_ndim, _k_state, _k_pairing = _combine_k_from_operands(aa, bb, cc)
    return Vec(
        torch.stack([aa.data, bb.data, cc.data], dim=-1),
        sub_batch_ndim=sb,
        sub_batch_state=state,
        sub_batch_meta=meta,
        k_ndim=_k_ndim,
        k_state=_k_state,
        k_pairing=_k_pairing,
    )


__all__ = [
    "abs",
    "bilinear_interpolation",
    "bilinear_interpolation_slopes",
    "allclose",
    "compose",
    "cosh",
    "det",
    "dev",
    "dexp_map",
    "diff",
    "drotate",
    "drotate_self",
    "equal",
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
    "mean",
    "norm",
    "outer",
    "pow",
    "r2_from_sr2",
    "r2_from_wr2",
    "rotate",
    "skew",
    "sign",
    "sinh",
    "sqrt",
    "sum",
    "sym",
    "tanh",
    "tr",
    "unit",
    "vec_component",
    "vec_from_scalars",
    "vol",
]
