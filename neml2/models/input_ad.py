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

"""Reverse-mode local *input* Jacobians for the ``request_AD`` authoring feature.

``request_AD`` (:meth:`neml2.models.model.Model.request_AD`) lets a leaf declare a
set of ``(output, input)`` pairs whose first-order chain rule the framework
derives automatically -- the author writes only the value ``forward`` and never a
``v=`` chain-rule branch. This module builds the dense local Jacobian block
``d(out)/d(in)`` for each requested pair via reverse-mode ``torch.autograd.grad``
(``n_out`` backward passes per output, independent of the input count).

It is the *input*-side analogue of the parameter Jacobian in
:mod:`neml2.models.param_ad` and shares its single-reverse-mode design: reverse
mode is the only autodiff that lowers through AOTI (see memory
``request_ad_spike_verdict`` / ``aoti_reverse_mode_ad_lowering``), so the same
code serves every route -- eager today, the AOTI export wrapper in a later phase.
The dense block is then contracted with the incoming chain-rule tangent by
:func:`neml2.types._boundary.contract_jacobian_block`, and the contributions are
accumulated by :meth:`~neml2.models.model.Model.apply_chain_rule`.

Framework boundary (CLAUDE.md "Hard rules" 1 & 2): like :mod:`neml2.models.param_ad`
this is a sanctioned raw-tensor / autograd site; the ``.data`` reads below bear
``# noqa: data-ok``. The reverse-mode ``autograd.grad`` runs at forward depth > 0
(inside the leaf's ``__call__``), so the :mod:`neml2.models._guard` AD ban is
armed -- the build runs inside an :func:`~neml2.models._guard.allow_autograd`
window (framework-managed, sanctioned use).

v1 scope: plain-batch only. A sub-batched input/output raises -- the AD-derived
block does not yet thread sub-batch structure (mirrors the eager runtime's
plain-batch-only constraint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._guard import allow_autograd

if TYPE_CHECKING:
    from ..types import TensorWrapper
    from .model import Model

__all__ = ["local_input_jacobians"]


def local_input_jacobians(
    model: Model,
    typed_args: tuple[TensorWrapper, ...],
    pairs: set[tuple[str, str]],
    out_names: list[str],
    output_spec: dict[str, type[TensorWrapper]],
) -> tuple[tuple[TensorWrapper, ...], dict[tuple[str, str], torch.Tensor]]:
    """Run *model*'s value ``forward`` once and reverse-mode the requested pairs.

    Returns ``(detached_typed_outputs, blocks)`` where ``blocks[(out, in)]`` is the
    dense Jacobian ``(*batch, *out_base, *in_base)`` for one requested pair (the
    same natural per-pair convention as the input Jacobian elsewhere). The value
    ``forward`` runs once with grad-tracking copies of the AD inputs swapped in;
    each output's block costs ``n_out`` reverse passes (one per output base
    component), independent of the input count -- the efficient regime when
    ``out_base <= in_base``. Outputs not appearing in any pair contribute no
    backward passes.

    *typed_args* are the leaf's positional inputs (structural inputs + any promoted
    runtime parameters) as typed wrappers, in ``input_spec`` order. *pairs* is the
    leaf's resolved ``{(out_name, in_name)}`` request set; only the inputs it names
    are differentiated. Plain-batch only.
    """
    input_names = list(model.input_spec)
    ad_in_names = {i for (_, i) in pairs}

    leaves: dict[str, torch.Tensor] = {}
    grad_args: list = []
    for name, ta in zip(input_names, typed_args, strict=True):
        if name in ad_in_names:
            if ta.sub_batch_ndim:
                raise NotImplementedError(
                    f"request_AD is plain-batch only (v1): input {name!r} of "
                    f"{type(model).__name__} carries sub_batch_ndim={ta.sub_batch_ndim}."
                )
            # Fresh autograd leaf: drop any incoming graph, track grad for the J build.
            leaf = ta.data.detach().clone().requires_grad_(True)  # noqa: data-ok autograd boundary
            leaves[name] = leaf
            grad_args.append(type(ta)(leaf))
        else:
            grad_args.append(ta)

    # EAGER needs both context managers: ``allow_autograd`` lifts the guard's
    # autograd ban inside the forward window, and ``torch.enable_grad`` re-enables
    # grad because a request_AD leaf inside an
    # :class:`~neml2.models.common.ImplicitUpdate` is evaluated within the Newton
    # solve's :class:`torch.autograd.Function` (``_ImplicitUpdateFn.forward``, which
    # runs under no_grad). Under ``torch.export`` BOTH are skipped: the guard is
    # already inert while compiling (see :func:`neml2.models._guard._armed`), grad
    # is already on, and neither the ``@contextmanager`` escape hatch nor
    # ``torch.enable_grad`` is Dynamo-traceable. Running the reverse pass with NO
    # context managers under compilation is exactly what lets the forward-mode
    # chain-rule graph (``_ForwardJacobianModule``) lower with an AD leaf embedded,
    # so the analytic leaves keep their hand-written chain rule and only the AD leaf
    # falls back to autograd.
    if torch.compiler.is_compiling():
        return _reverse_blocks(model, grad_args, leaves, pairs, out_names, output_spec)
    with (
        allow_autograd("request_AD: framework-managed reverse-mode local Jacobian"),
        torch.enable_grad(),
    ):
        return _reverse_blocks(model, grad_args, leaves, pairs, out_names, output_spec)


def _reverse_blocks(
    model: Model,
    grad_args: list,
    leaves: dict[str, torch.Tensor],
    pairs: set[tuple[str, str]],
    out_names: list[str],
    output_spec: dict[str, type[TensorWrapper]],
) -> tuple[tuple[TensorWrapper, ...], dict[tuple[str, str], torch.Tensor]]:
    """Value forward + reverse-mode per-pair blocks. Caller arranges grad mode /
    guard (eager wraps this in ``allow_autograd`` + ``enable_grad``; under
    ``torch.export`` it is called bare so the graph lowers). See
    :func:`local_input_jacobians`.
    """
    blocks: dict[tuple[str, str], torch.Tensor] = {}
    detached: list[TensorWrapper] = []
    result = model.forward(*grad_args)
    typed_outs = result if isinstance(result, tuple) else (result,)
    for o_typed, o_name in zip(typed_outs, out_names, strict=True):
        if o_typed.sub_batch_ndim:
            raise NotImplementedError(
                f"request_AD is plain-batch only (v1): output {o_name!r} of "
                f"{type(model).__name__} carries sub_batch_ndim={o_typed.sub_batch_ndim}."
            )
        odata = o_typed.data  # noqa: data-ok autograd boundary  (*batch, *out_base)
        detached.append(type(o_typed)(odata.detach()))
        req_ins = [i for (o, i) in pairs if o == o_name]
        if not req_ins:
            continue
        o_base = tuple(int(s) for s in output_spec[o_name].BASE_SHAPE)
        n_out = 1
        for s in o_base:
            n_out *= s
        batch = tuple(o_typed.batch_shape)
        leaf_list = [leaves[i] for i in req_ins]
        cols: dict[str, list[torch.Tensor]] = {i: [] for i in req_ins}
        for k in range(n_out):
            # One-hot cotangent on output component k (built flat, then viewed to
            # the output shape -- no contiguity assumption on odata).
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
            for i, g in zip(req_ins, grads, strict=True):
                cols[i].append(g if g is not None else torch.zeros_like(leaves[i]))
        for i in req_ins:
            i_base = tuple(int(s) for s in model.input_spec[i].BASE_SHAPE)
            # (*batch, n_out, *in_base) -> (*batch, *out_base, *in_base)
            stacked = torch.stack(cols[i], dim=len(batch))
            blocks[(o_name, i)] = stacked.reshape(*batch, *o_base, *i_base).contiguous()
    return tuple(detached), blocks
