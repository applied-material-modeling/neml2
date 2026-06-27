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

"""Reverse-mode parameter derivatives: ``d(output)/d(parameter)`` via torch autograd.

Single source of truth for the parameter-derivative autodiff shared by every
runtime route (``py-eager`` / ``cpp-eager`` today; the AOTI export wrapper
later). It is a *separate* path from the forward-mode input chain rule
(``Model.forward(..., v=)``), which is left entirely untouched.

Why reverse-mode raw ``torch.autograd.grad`` (and not ``torch.func``)? The
Phase-0 feasibility spike found that on torch 2.12 *only* a reverse-mode
``autograd.grad`` graph survives ``torch.export`` -> AOTInductor lowering --
every ``torch.func`` transform (``jvp`` / ``jacfwd`` / ``jacrev``) fails at
export, and forward-mode dual tensors export but fail to lower. Building the
parameter derivative on ``autograd.grad`` therefore lets the *same* code serve
the eager routes now and be wrapped into an exportable graph in Phase 3.

For ``d(output)/d(param)`` the output base is small (e.g. an ``SR2`` stress has
6 components) and the parameter count may be large, so reverse-mode is also the
efficient regime: the dense block costs ``n_out`` backward passes, independent
of the number of parameters.

Parameters are addressed by their :meth:`torch.nn.Module.named_parameters`
qualified names (e.g. ``"elasticity.E"``). :func:`torch.func.functional_call`
swaps a per-batch-expanded, grad-tracking copy of each selected parameter into
the module functionally -- no module mutation, no per-leaf edits.

Framework boundary (CLAUDE.md "Hard rules" 1 & 2): like
:mod:`neml2.types._boundary`, this is a sanctioned raw-tensor / autograd site.
The ``.data`` reads below are the autograd boundary and bear ``# data-ok``.
The reverse-mode ``autograd.grad`` runs at forward-depth 0 (after the model's
``forward`` returns), so the :mod:`neml2.models._guard` AD ban is never armed --
no ``allow_autograd`` region is needed.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from ..types import TensorWrapper

__all__ = ["call_batch_shape", "enumerate_typed_params", "param_jacobian", "param_vjp"]


def enumerate_typed_params(model: torch.nn.Module) -> list[tuple[str, type[TensorWrapper]]]:
    """``(qualified_name, TensorWrapper class)`` for every typed parameter of *model*.

    Qualified names match :meth:`torch.nn.Module.named_parameters` keys (dotted
    module path + parameter name). The typed class is recovered from the owning
    leaf's ``_typed_storage_classes`` registry so callers can report base shapes
    and re-wrap blocks. Parameters with no registered typed class (none, for a
    well-formed NEML2 model) are skipped.
    """
    out: list[tuple[str, type[TensorWrapper]]] = []
    for qname, _ in model.named_parameters():
        module_path, _, pname = qname.rpartition(".")
        owner = model.get_submodule(module_path) if module_path else model
        type_cls = getattr(owner, "_typed_storage_classes", {}).get(pname)
        if type_cls is not None:
            out.append((qname, type_cls))
    return out


def call_batch_shape(
    typed_inputs: tuple[TensorWrapper, ...],
    model: torch.nn.Module,
    param_names: list[str],
    param_base_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[int, ...]:
    """Broadcast of all input batch shapes with all selected parameters' batch shapes.

    A parameter's value may itself carry a batch dimension (e.g. a Scalar ``E`` set
    to ``(B,)`` for per-batch-element properties). The forward broadcasts every
    parameter's batch into the output, so the call's batch -- the shape derivative
    seeds and per-batch parameter copies must be built at -- is the broadcast of the
    inputs' batches AND the parameters' batches. A parameter's batch is its stored
    shape with its natural base (``BASE_SHAPE``, via *param_base_shapes*) stripped
    from the trailing end.
    """
    params = dict(model.named_parameters())
    shapes: list[tuple[int, ...]] = [tuple(ti.batch_shape) for ti in typed_inputs]
    for q in param_names:
        base = tuple(param_base_shapes.get(q, ()))
        pshape = tuple(params[q].shape)
        shapes.append(pshape[: len(pshape) - len(base)])
    return tuple(torch.broadcast_shapes(*shapes)) if shapes else ()


def _expanded_leaf_params(
    model: torch.nn.Module,
    param_names: list[str],
    call_batch: tuple[int, ...],
    param_base_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    """Per-batch-expanded, grad-tracking copies of the selected parameters.

    Each parameter is broadcast from its stored ``(*pbatch, *base)`` to
    ``(*call_batch, *base)`` (right-aligned: ``pbatch`` broadcasts into
    ``call_batch`` by construction; *base* is the natural base from
    *param_base_shapes*). Expanding to the call batch keeps the reverse-mode
    gradient *per batch element*: every element's output then depends only on its
    own copy of the parameter, so :func:`torch.autograd.grad` does not sum the
    contribution across the batch -- exactly what a dense ``d(output)/d(param)``
    block needs. An already-batched parameter (``pbatch == call_batch``) is left
    per-element; a scalar / unbatched parameter is expanded to per-element copies.
    The expanded copy is a fresh autograd leaf (``.contiguous().requires_grad_(True)``).
    """
    params = dict(model.named_parameters())
    leaf: dict[str, torch.Tensor] = {}
    for q in param_names:
        base = tuple(param_base_shapes.get(q, ()))
        p = params[q].detach()  # drop the nn.Parameter graph; fresh autograd leaf below
        pe = p.broadcast_to(tuple(call_batch) + base).contiguous()
        leaf[q] = pe.requires_grad_(True)
    return leaf


@contextlib.contextmanager
def _swapped_parameters(
    model: torch.nn.Module, leaf: dict[str, torch.Tensor]
) -> Generator[None, None, None]:
    """Temporarily install *leaf* tensors in place of the model's parameters.

    Like :func:`torch.func.functional_call` this swaps parameter tensors without
    mutating the parameters' values permanently -- but unlike ``functional_call``
    (which restores on *forward* return) the swap stays active for the whole
    ``with`` block, crucially **through the reverse-mode** ``autograd.grad``
    **backward**. :class:`~neml2.models.common.ImplicitUpdate`'s autograd Function
    (``_ImplicitUpdateFn.backward``) re-reads ``owner.parameters()`` at backward
    time to rebuild the residual graph for its implicit-function-theorem adjoint;
    the swap must still be in place then so that adjoint differentiates w.r.t. the
    per-batch-expanded copies. With ``functional_call`` the parameters are already
    restored by backward time, breaking the implicit (Newton-solve) path.

    The tensors are assigned directly into each owning leaf's ``_parameters`` dict
    (the same mechanism ``functional_call`` uses internally); NEML2's
    ``Model.__getattr__`` re-wraps them to their typed class on read.
    """
    saved: list[tuple[torch.nn.Module, str, object]] = []
    try:
        for qname, tensor in leaf.items():
            module_path, _, pname = qname.rpartition(".")
            owner = model.get_submodule(module_path) if module_path else model
            saved.append((owner, pname, owner._parameters[pname]))
            owner._parameters[pname] = tensor  # type: ignore[assignment]
        yield
    finally:
        for owner, pname, original in saved:
            owner._parameters[pname] = original  # type: ignore[assignment]


def param_jacobian(
    model: torch.nn.Module,
    typed_inputs: tuple[TensorWrapper, ...],
    param_names: list[str],
    output_names: list[str],
    output_spec: dict[str, type[TensorWrapper]],
    param_base_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Dense ``d(output)/d(param)`` blocks via reverse-mode autograd.

    Returns the nested ``{out_name: {param_qname: block}}`` map, each block a raw
    tensor at ``(*batch, *out_base, *param_base)`` -- the same natural per-pair
    convention as the input Jacobian (a Scalar parameter contributes no trailing
    axis). Rows in ``output_names`` order; columns in ``param_names`` order.

    One reverse pass per output base-component (``n_out`` passes total per
    output, independent of the number of parameters). A parameter that does not
    influence an output yields an explicit zero block. Composition through an
    :class:`~neml2.models.common.ImplicitUpdate` (Newton solve) is handled for
    free: the reverse pass traverses its autograd Function, whose backward is the
    implicit-function-theorem adjoint.

    *typed_inputs* are the model's positional inputs as typed wrappers, already
    broadcast to a common batch (the caller's responsibility). The model forward
    runs with grad-tracking parameter copies swapped in (see
    :func:`_swapped_parameters`); the swap is held across every ``autograd.grad``
    so the implicit backward path differentiates against the same copies.
    """
    if not param_names:
        return {o: {} for o in output_names}

    batch = call_batch_shape(typed_inputs, model, param_names, param_base_shapes)
    leaf = _expanded_leaf_params(model, param_names, batch, param_base_shapes)
    leaf_list = [leaf[q] for q in param_names]

    jac: dict[str, dict[str, torch.Tensor]] = {}
    with _swapped_parameters(model, leaf):
        result = model(*tuple(typed_inputs))
        typed_outs = result if isinstance(result, tuple) else (result,)

        for o_typed, o_name in zip(typed_outs, output_names, strict=True):
            odata = o_typed.data  # data-ok autograd boundary  (*batch, *out_base)
            o_base = tuple(int(s) for s in output_spec[o_name].BASE_SHAPE)
            n_out = 1
            for s in o_base:
                n_out *= s

            cols: dict[str, list[torch.Tensor]] = {q: [] for q in param_names}
            for k in range(n_out):
                # Basis cotangent on output component k (built flat to avoid any
                # contiguity assumption on odata, then viewed to the output shape).
                seed_flat = torch.zeros(*batch, n_out, dtype=odata.dtype, device=odata.device)
                seed_flat[..., k] = 1.0
                seed = seed_flat.reshape(odata.shape)
                grads = torch.autograd.grad(
                    odata,
                    leaf_list,
                    grad_outputs=seed,
                    retain_graph=True,
                    allow_unused=True,
                )
                for q, g in zip(param_names, grads, strict=True):
                    cols[q].append(g if g is not None else torch.zeros_like(leaf[q]))

            row: dict[str, torch.Tensor] = {}
            for q in param_names:
                pbase = tuple(param_base_shapes.get(q, ()))  # natural base, not stored shape
                # (*batch, n_out, *param_base) -> (*batch, *out_base, *param_base)
                stacked = torch.stack(cols[q], dim=len(batch))
                row[q] = stacked.reshape(*batch, *o_base, *pbase).contiguous()
            jac[o_name] = row
    return jac


def param_vjp(
    model: torch.nn.Module,
    typed_inputs: tuple[TensorWrapper, ...],
    param_names: list[str],
    output_names: list[str],
    cotangents: Mapping[str, torch.Tensor | TensorWrapper],
) -> dict[str, torch.Tensor]:
    r"""Reverse-mode parameter adjoint ``dL/d\theta`` for ``L = sum_o <w_o, out_o>``.

    Returns ``{param_qname: grad}`` at each parameter's natural shape (a scalar
    parameter yields a scalar, the batch summed out -- exactly ``dL/d\theta`` for
    a global parameter). One reverse pass total, independent of the number of
    parameters; this is the cheap form for many-parameter (inverse-optimization)
    gradients and the eager analogue of the AOTI ``param_vjp``.

    Unlike :func:`param_jacobian` the parameters are swapped at their NATURAL
    shape (not per-batch-expanded), so :func:`torch.autograd.grad` sums each
    parameter's contribution over the batch -- which is the global-parameter
    gradient. The swap is held across the backward so an
    :class:`~neml2.models.common.ImplicitUpdate`'s implicit-function-theorem
    adjoint differentiates against the same swapped tensors. A parameter that
    does not influence any output yields an explicit zero gradient.

    *cotangents* maps each output name to its cotangent ``w_o`` (raw tensor or
    typed wrapper) at the output's ``(*batch, *out_base)`` shape.
    """
    if not param_names:
        return {}

    params = dict(model.named_parameters())
    # Natural-shape grad-tracking copies (no per-batch expand -> the gradient
    # sums over the batch, the global-parameter adjoint).
    leaf: dict[str, torch.Tensor] = {
        q: params[q].detach().clone().requires_grad_(True) for q in param_names
    }
    leaf_list = [leaf[q] for q in param_names]

    with _swapped_parameters(model, leaf):
        result = model(*tuple(typed_inputs))
        typed_outs = result if isinstance(result, tuple) else (result,)
        loss: torch.Tensor | None = None
        for o_typed, o_name in zip(typed_outs, output_names, strict=True):
            w = cotangents[o_name]
            wdata = w.data if hasattr(w, "data") else w  # data-ok autograd boundary
            term = (o_typed.data * wdata).sum()  # data-ok autograd boundary
            loss = term if loss is None else loss + term
        assert loss is not None  # at least one output -> at least one term
        grads = torch.autograd.grad(loss, leaf_list, retain_graph=False, allow_unused=True)

    return {
        q: (g if g is not None else torch.zeros_like(leaf[q]))
        for q, g in zip(param_names, grads, strict=True)
    }
