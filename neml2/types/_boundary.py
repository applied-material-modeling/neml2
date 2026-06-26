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

"""Shared raw-tensor ⇄ typed-wrapper boundary helpers.

Both of NEML2's C++-facing runtimes accept *raw* ``torch.Tensor`` inputs keyed
by variable name and must normalize them before handing them to typed code:

* the **AOTI shim** (:mod:`neml2.aoti._shim`) -- inputs cross from the C++
  ``neml2::aoti::Model`` pybind binding;
* the **eager embed bridge** (:mod:`neml2.eager`) -- inputs cross from the C++
  ``neml2::eager::Model`` via the embedded interpreter.

The two pieces of normalization both runtimes need are pure functions of
``(tensor, spec, device, dtype)``, so they live here as a single source of
truth (CLAUDE.md rule 3): a fix to the device/dtype validation or the batch
broadcasting lands once, not twice.

This module sits at a framework boundary -- it is one of the few places that
legitimately handles raw ``torch.Tensor`` (see CLAUDE.md "Hard rules / Rule
1"). The raw tensors here are inbound from a C++ boundary; callers re-wrap them
into typed wrappers immediately after these helpers run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import TensorWrapper


def check_tensor(
    t: torch.Tensor,
    name: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
    *,
    kind: str,
    context: str,
    hint: str = "",
) -> None:
    """Strict device + dtype check; raise ``TypeError`` on mismatch.

    ``kind`` is ``'input'`` or ``'parameter'`` and shows up in the error
    message so the caller can tell whether to fix the call site or the
    parameter setter. ``context`` is the runtime label (``'AOTIModel'`` /
    ``'EagerModel'``) that prefixes the message; ``hint`` is appended verbatim
    so each runtime can give its own remediation guidance.

    Only floating-point dtype is checked -- bool / integer tensors are passed
    through unchanged because their widths are not part of the contract.

    The device comparison is index-aware. ``torch.device('cuda')`` and
    ``torch.device('cuda:0')`` refer to the same physical GPU but ``__eq__``
    reports them unequal; we treat them as equal whenever the target lacks an
    explicit index. Same-type + same-index (when the target pins an index) is
    required.
    """
    device_ok = t.device.type == target_device.type and (
        target_device.index is None or t.device.index == target_device.index
    )
    problems = []
    if not device_ok:
        problems.append(f"device={t.device} (expected {target_device})")
    if t.is_floating_point() and t.dtype != target_dtype:
        problems.append(f"dtype={t.dtype} (expected {target_dtype})")
    if not problems:
        return
    msg = f"{context} {kind} {name!r}: {', '.join(problems)}."
    if hint:
        msg += " " + hint
    raise TypeError(msg)


def broadcast_to_common_batch(
    raw_inputs: dict[str, torch.Tensor],
    input_spec: dict[str, type],
    per_input_sub_ndim: dict[str, int],
) -> tuple[dict[str, torch.Tensor], torch.Size]:
    """Bring every input tensor to its declared ``(*dyn, *sub, *base)`` shape,
    with the dynamic-batch axes broadcast to a single common shape.

    Each input has three trailing-axis regions:

    * ``base`` -- ``BASE_SHAPE`` from the TensorWrapper class
      (``input_spec[name]``);
    * ``sub`` -- the per-input ``sub_batch_ndim`` resolved by the caller
      (``per_input_sub_ndim[name]``);
    * ``dyn`` -- everything in front, broadcast across all inputs (callers
      freely pass base-only defaults / single-step slices and rely on the
      runtime to lift them to the common dyn shape).

    The (dyn, sub, base) split is per-input -- a global Scalar with no
    sub-batch is reshaped to ``(*common_dyn, *base)`` while a per-crystal SR2
    with ``sub_batch_ndim=1`` is reshaped to ``(*common_dyn, sub_size,
    *base)``. Without the per-input sub-batch carve-out, a naive
    ``torch.broadcast_shapes`` over the full batch region would collide the
    per-crystal axis against the global input's dyn axis.
    """
    dyn_shapes: list[torch.Size] = []
    for name, t in raw_inputs.items():
        base_ndim = input_spec[name].BASE_NDIM
        sub_ndim = per_input_sub_ndim[name]
        trail = base_ndim + sub_ndim
        dyn_shapes.append(t.shape if trail == 0 else t.shape[:-trail])
    common_dyn = torch.broadcast_shapes(*dyn_shapes)

    out: dict[str, torch.Tensor] = {}
    for name, t in raw_inputs.items():
        base_ndim = input_spec[name].BASE_NDIM
        base_shape = tuple(input_spec[name].BASE_SHAPE)
        sub_ndim = per_input_sub_ndim[name]
        # Read the sub-batch shape from the input tensor's current trailing
        # axes (the caller's wrapper already has the right sub_batch_ndim; the
        # raw tensor's trailing-axis sizes are the sub-batch extents).
        sub_shape = (
            tuple(t.shape[len(t.shape) - base_ndim - sub_ndim : len(t.shape) - base_ndim])
            if sub_ndim
            else ()
        )
        target_shape = tuple(common_dyn) + sub_shape + base_shape
        if tuple(t.shape) == target_shape:
            out[name] = t
            continue
        view_shape = (1,) * (len(target_shape) - t.ndim) + tuple(t.shape)
        out[name] = t.view(view_shape).expand(target_shape).contiguous()
    return out, common_dyn


def contract_jacobian_block(
    block: torch.Tensor,
    v: TensorWrapper,
    out_type: type[TensorWrapper],
) -> TensorWrapper:
    """Contract a raw local Jacobian block with a typed chain-rule tangent.

    The ``request_AD`` boundary: reverse-mode autograd yields a dense local
    Jacobian ``block`` of shape ``(*B, *out_base, *in_base)`` (see
    :func:`neml2.models.input_ad.local_input_jacobians`); ``v`` is the incoming
    first-order tangent of the *input* type. Two seed conventions are accepted,
    matching exactly what a hand-written chain-rule action accepts -- this is
    what makes a ``request_AD`` leaf indistinguishable from a hand-written one
    at the ``v=`` boundary:

    - **Leading-K** (``k_ndim == 1``): ``v`` is ``(K, *B, *in_base)`` -- a stack
      of ``K`` directional seeds (e.g. an identity seed read off a whole
      Jacobian column-by-column). The result keeps the same single K axis:
      ``(K, *B, *out_base)``.
    - **Single-direction** (``k_ndim == 0``): ``v`` is ``(*B, *in_base)`` -- one
      directional seed, as ``ModelUnitTest`` and the ``Vec(eye(n))`` idiom send.
      The result has no K axis: ``(*B, *out_base)``.

    Either way the contraction is ``J @ v`` summed over the input base and
    broadcast over the batch (and K, when present).
    :meth:`~neml2.models.model.Model.apply_chain_rule` accumulates the
    contributions across input edges.

    Plain-batch only (``v.sub_batch_ndim == 0``): the request_AD primitive does
    not yet thread sub-batch structure through the AD-derived block.

    Framework boundary (CLAUDE.md "Hard rules / Rule 1"): the block is raw
    (reverse-mode autograd output), so this read of ``v.data`` + raw matmul is a
    sanctioned raw-tensor site, re-wrapped to typed on return.
    """
    if v.sub_batch_ndim:
        raise NotImplementedError(
            "request_AD is plain-batch only (v1): the AD-derived Jacobian block "
            f"cannot contract a sub-batched tangent ({type(v).__name__} with "
            f"sub_batch_ndim={v.sub_batch_ndim})."
        )
    if v.k_ndim not in (0, 1):
        raise NotImplementedError(
            "contract_jacobian_block expects a single-direction (k_ndim=0) or "
            f"single leading-K (k_ndim=1) tangent, got k_ndim={v.k_ndim}."
        )
    in_base = tuple(int(s) for s in type(v).BASE_SHAPE)
    out_base = tuple(int(s) for s in out_type.BASE_SHAPE)
    n_in = 1
    for s in in_base:
        n_in *= s
    n_out = 1
    for s in out_base:
        n_out *= s
    vd = v.data  # noqa: data-ok request_AD autograd boundary
    nb = block.ndim - len(out_base) - len(in_base)
    block_flat = block.reshape(*block.shape[:nb], n_out, n_in)  # (*Bblk, n_out, n_in)
    if v.k_ndim == 1:
        # Leading-K: vd is (K, *Bv, *in_base). Keep the K axis through the matmul
        # by giving the block a singleton K (broadcast across all seeds).
        v_batch = vd.shape[1 : vd.ndim - len(in_base)]
        v_flat = vd.reshape(vd.shape[0], *v_batch, n_in)
        # (1, *Bblk, n_out, n_in) @ (K, *Bv, n_in, 1) -> (K, *B, n_out, 1) -> (K, *B, n_out)
        res = (block_flat.unsqueeze(0) @ v_flat.unsqueeze(-1)).squeeze(-1)
        res = res.reshape(*res.shape[:-1], *out_base)
        return out_type(res, k_ndim=1, k_state=v.k_state, k_pairing=v.k_pairing)
    # Single-direction: vd is (*Bv, *in_base), no K axis. matmul broadcasts the
    # block's batch against the tangent's; the result carries no K axis either.
    v_batch = vd.shape[: vd.ndim - len(in_base)]
    v_flat = vd.reshape(*v_batch, n_in)
    # (*Bblk, n_out, n_in) @ (*Bv, n_in, 1) -> (*B, n_out, 1) -> (*B, n_out)
    res = (block_flat @ v_flat.unsqueeze(-1)).squeeze(-1)
    res = res.reshape(*res.shape[:-1], *out_base)
    return out_type(res)


def unwrap_outputs(
    typed_outs: tuple[TensorWrapper, ...],
    output_names: list[str],
) -> dict[str, torch.Tensor]:
    """Unwrap a native model's typed output tuple to a raw ``{name: Tensor}`` dict.

    ``typed_outs`` is the tuple of :class:`~neml2.types.TensorWrapper` instances
    a native model returns, in ``output_spec`` order; ``output_names`` is the
    matching list of names. This is the *outbound* half of the eager embed
    boundary -- the raw tensors leave for the C++ ``neml2::eager::Model`` here,
    so this is the one place that legitimately reads ``.data`` (see the module
    docstring + CLAUDE.md "Hard rules / Rule 1").
    """
    return {
        name: wrapper.data  # noqa: data-ok eager embed boundary
        for name, wrapper in zip(output_names, typed_outs, strict=True)
    }


def assemble_jvp_outputs(
    v_out: dict,
    typed_outputs: tuple[TensorWrapper, ...],
    output_names: list[str],
) -> dict[str, torch.Tensor]:
    """Sum directional chain-rule contributions into raw, base-shaped JVP outputs.

    Outbound JVP half of the eager embed boundary. ``v_out`` is the native
    ``ComposedModel`` chain-rule result ``{out_name: {seed_name: block}}`` from a
    K=1 directional seed; each block's data is ``(1, *batch, *out_base)``. For
    each output the K axis is squeezed and the per-input contributions summed,
    giving the unflattened ``(*batch, *out_base)`` -- matching the C++
    ``neml2::aoti::Model::jvp`` contract. An output with no dependence on any
    seeded input is an explicit zero block.

    Reads ``.data`` -- one of the few legitimate raw-tensor sites (see the module
    docstring + CLAUDE.md "Hard rules / Rule 1").
    """
    out: dict[str, torch.Tensor] = {}
    for typed_out, name in zip(typed_outputs, output_names, strict=True):
        contribs = v_out.get(name, {})
        if not contribs:
            # Same (*batch, *out_base) shape as the value output, all zeros.
            out[name] = torch.zeros_like(typed_out.data)  # noqa: data-ok eager embed boundary
            continue
        # (1, *batch, *out_base) -> (*batch, *out_base), summed over seeded inputs.
        blocks = [b.data.squeeze(0) for b in contribs.values()]  # noqa: data-ok eager embed boundary
        out[name] = blocks[0] if len(blocks) == 1 else torch.stack(blocks, 0).sum(0)
    return out


def assemble_jacobian(
    v_out: dict,
    typed_outputs: tuple[TensorWrapper, ...],
    output_names: list[str],
    output_spec: dict[str, type[TensorWrapper]],
    input_names: list[str],
    input_spec: dict[str, type[TensorWrapper]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Assemble per-(out, in) chain-rule blocks into the nested variable-pair Jacobian.

    Outbound Jacobian half of the eager embed boundary. ``v_out`` is the native
    ``ComposedModel`` chain-rule result from a leading-K identity seed per input
    (``{out_name: {in_name: block}}``). Returns the nested
    ``{out_name: {in_name: (*batch, *out_base, *in_base)}}`` matching the C++
    ``neml2::aoti::Model::jacobian`` contract (rows in ``output_spec`` order, cols
    in ``input_spec`` order). A constant ``(out, in)`` pair (absent from
    ``v_out``) is an explicit zero block.

    Reuses :func:`neml2.cli.aoti_export._leading_k_block_to_per_pair` (the AOTI
    export's own block-to-pair converter, lazily imported to avoid import-time
    CLI coupling); the ``.data`` read happens inside that boundary helper.
    """
    from ..cli.aoti_export import _leading_k_block_to_per_pair

    first = typed_outputs[0]
    batch_shape = tuple(first.batch_shape)
    opts = {"dtype": first.dtype, "device": first.device}
    jac: dict[str, dict[str, torch.Tensor]] = {}
    for o_name in output_names:
        o_base = tuple(output_spec[o_name].BASE_SHAPE)
        row: dict[str, torch.Tensor] = {}
        for i_name in input_names:
            i_base = tuple(input_spec[i_name].BASE_SHAPE)
            block = v_out.get(o_name, {}).get(i_name)
            if block is None:
                # Constant (out, in) pair: a structural zero block.
                row[i_name] = torch.zeros(*batch_shape, *o_base, *i_base, **opts)
            else:
                # (*batch, *out_base, *in_base).
                row[i_name] = _leading_k_block_to_per_pair(block, input_spec[i_name]).contiguous()
        jac[o_name] = row
    return jac
