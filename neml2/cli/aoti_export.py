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

"""AOTI export entry point for Python-native NEML2 models.

:func:`export_model_for_aoti` loads a model from a HIT input file (native
registry only — no C++ fallback), detects the model's structure, compiles the
appropriate ``.pt2`` artifacts via :func:`~neml2.export.compile_model`,
and writes a metadata JSON describing the variable layout for the C++ AOTI
wrappers.

Unified on-disk schema: every export emits
``"type": "composed"`` with a per-segment list. Three shapes feed this:

* **Forward** (root is a non-Implicit model, no ``ImplicitUpdate`` children):
  one segment, ``"kind": "forward"`` → one ``.pt2``.
* **Implicit** (root is an ``ImplicitUpdate``): one segment,
  ``"kind": "implicit"`` → two ``.pt2``\\ s (``_rhs`` + ``_step``) plus
  optional ``_predictor.pt2``.
* **Composed** (root is a ``ComposedModel`` whose dependency graph contains an
  ``ImplicitUpdate`` child): partitioned at each ``ImplicitUpdate`` boundary
  into a sequence of segments (alternating forward / implicit / forward / …),
  each emitted in the shapes above.

The C++ Factory dispatches uniformly on ``AOTIModel`` for all three;
single-segment metadata produces a trivial orchestrator that just runs the
one segment, with overhead of a handful of dict touches per call (negligible
vs the AOTI call itself). The previous single-purpose C++ wrappers
(``AOTIForwardModel`` / ``ImplicitAOTIModel``) were removed once
close-out once the unified ``AOTIModel`` covered every shape end-to-end.

CLI
---
``neml2-aoti-export --hit <file.i> --model <name> --output <dir>``
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import torch
from torch import nn

from ..types import TensorWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Metadata schema version. Bumped at every breaking change so the C++ loader
#: can refuse mismatched caches with a clear "wipe and re-export" message
#: rather than failing on a missing field deep in the parser.
#:
#: v2 (current): bake-by-default; promoted parameters listed under top-level
#: ``parameters`` with per-segment ``param_inputs``. No ``buffers`` section.
#: Adds top-level ``device`` + ``dtype`` keys.
AOTI_META_SCHEMA_VERSION = 2


def _var_size(type_cls) -> int:
    return prod(type_cls.BASE_SHAPE) if type_cls.BASE_SHAPE else 1


def _leading_k_identity_seed(
    type_cls: type[TensorWrapper],
    batch_shape: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorWrapper:
    """Identity seed for Phase-14 leading-K typed tangents."""
    n = _var_size(type_cls)
    eye = torch.eye(n, dtype=dtype, device=device)
    mid = (1,) * len(batch_shape)
    eye = eye.reshape(n, *mid, *type_cls.BASE_SHAPE)
    data = eye.expand(n, *tuple(batch_shape), *type_cls.BASE_SHAPE).contiguous()
    return type_cls(data)


def _leading_k_block_to_trailing_k(block: torch.Tensor | TensorWrapper) -> torch.Tensor:
    """Convert a leading-K typed tangent block to ``(*batch, n_out, K)``."""
    if not isinstance(block, TensorWrapper):
        return block
    base_ndim = type(block).BASE_NDIM
    moved = block.data.movedim(0, -1)
    if base_ndim == 0:
        return moved.unsqueeze(-2)
    if base_ndim == 1:
        return moved
    K = moved.shape[-1]
    return moved.reshape(*moved.shape[: moved.ndim - 1 - base_ndim], -1, K)


def _var_infos(
    spec: dict,
    *,
    sub_batch_shapes: dict[str, tuple[int, ...]] | None = None,
    sparsity: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Build the per-variable metadata list.

    Parameters
    ----------
    spec:
        Ordered ``{name: type_cls}`` mapping (usually a ``Model.input_spec``
        or ``Model.output_spec``).
    sub_batch_shapes:
        Optional ``{name: shape_tuple}`` map; absent entries default to ``()``
        and the field is omitted from the output dict to keep the JSON tight.
    sparsity:
        Optional ``{name: {other_name: "diagonal"|"dense"}}`` map. Only the
        non-default ``"dense"`` entries need to appear (the C++ side treats
        absent entries as ``"diagonal"``); however we still emit the full
        per-pair dict when this argument is set so the metadata is
        self-describing.
    """
    infos: list[dict] = []
    sub_batch_shapes = sub_batch_shapes or {}
    sparsity = sparsity or {}
    for k, v in spec.items():
        info: dict = {"name": k, "var_size": _var_size(v), "var_type": v.__name__}
        sb = sub_batch_shapes.get(k, ())
        if sb:
            info["sub_batch_shape"] = list(sb)
        if k in sparsity:
            info["sparsity"] = dict(sparsity[k])
        infos.append(info)
    return infos


def _sparsity_for_outputs(
    model, output_names: tuple[str, ...] | list[str], input_names: tuple[str, ...] | list[str]
) -> dict[str, dict[str, str]]:
    """Project ``model.list_deriv`` to the per-output → input flag map.

    Only outputs that have at least one declared (output, input) entry get an
    entry in the result; outputs whose every (output, input) pair is the
    default ``"diagonal"`` are absent — keeping the JSON quiet for the common
    pre-sub-batch case.
    """
    if not getattr(model, "list_deriv", None):
        return {}
    per_out: dict[str, dict[str, str]] = {}
    for (out_name, in_name), flag in model.list_deriv.items():
        if out_name in output_names and in_name in input_names and flag == "dense":
            per_out.setdefault(out_name, {})[in_name] = flag
    return per_out


def _example_inputs_for(model, device: str) -> tuple:
    """Return a batch-2 example input tuple matching model.input_spec."""
    return tuple(
        torch.zeros(2, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
        for type_cls in model.input_spec.values()
    )


def _seed_implicit_subbatch(
    model,
    driver_inputs: dict[str, torch.Tensor],
    device: str,
) -> None:
    """Call ``system.initialize(u=, g=)`` on every nested
    :class:`~neml2.models.common.ImplicitUpdate` using *driver_inputs* as the
    given-variable values and zero-shaped unknowns. ``initialize`` infers
    per-variable ``sub_batch_shape`` from the input tensor shapes and rebuilds
    the system layouts (D-052 path) without running the Newton solver — the
    cheap, side-effect-only path the Block export needs to know how to size
    its flat ``u_flat`` / ``g_flat`` buffers.

    For each unknown whose matching ``<unknown>~1`` history input is present in
    *driver_inputs*, the unknown is sized the same way; for unknowns without
    such a mate (e.g. predictor-only ``deformation_rate``,
    ``target_cauchy_stress``) the unknown is created with no sub-batch.
    """
    from ..models.common import ImplicitUpdate  # noqa: PLC0415

    seen: set[int] = set()
    for sub in model.modules():
        if not isinstance(sub, ImplicitUpdate):
            continue
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        system = sub.system
        g_dict: dict[str, torch.Tensor] = {}
        for name in system.given_names:
            if name in driver_inputs:
                g_dict[name] = driver_inputs[name]
            else:
                type_cls = system.model.input_spec[name]
                g_dict[name] = torch.zeros(
                    1, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device
                )
        u_dict: dict[str, torch.Tensor] = {}
        for name in system.unknown_names:
            type_cls = system.model.input_spec[name]
            history_name = f"{name}~1"
            if history_name in driver_inputs:
                # Mirror the IC's sub-batch shape: state-at-end-of-step has the
                # same per-site structure as state-at-start-of-step.
                ref = driver_inputs[history_name]
                u_dict[name] = torch.zeros_like(ref)
            else:
                u_dict[name] = torch.zeros(
                    1, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device
                )
        with torch.no_grad():
            system.initialize(u=u_dict, g=g_dict)


def _hit_driver_example_inputs(model, factory, device: str) -> dict[str, torch.Tensor]:
    """Scan the HIT file's ``[Drivers]`` sections for ``TransientDriver``-style
    ``ic_*_names``/``ic_*_values`` and ``force_*_names``/``force_*_values``
    pairs. Map each entry to a per-input-name example tensor:

    * ``ic_<type>_names[i]`` → ``<name>~1`` input, take the IC tensor as-is and
      prepend a unit dynamic-batch dim (ICs are stored without ``*B``).
    * ``force_<type>_names[i]`` → ``<name>`` input, slice the time-zeroth step
      of the force-trajectory tensor (forces are ``(nstep, *B, *base)``).

    The point isn't byte-perfect inputs — only correct *shape* (specifically
    sub-batch dims) so the subsequent ``model.forward(*)`` populates
    ``system._sub_batch_shapes`` on every inner ``ModelNonlinearSystem``.

    Returns a dict ``input_name → torch.Tensor``. Inputs not covered by any
    driver mapping are left out; the caller fills them with the standard
    ``zeros(1, *base)`` no-sub-batch default.
    """
    import nmhit  # noqa: PLC0415

    mapping: dict[str, tuple[str, str]] = {}  # input_name → (kind, tensor_name)
    # kind ∈ {"ic", "force"} so we can pick the right slice/unsqueeze rule.
    type_tags = ("Scalar", "SR2", "WR2", "Rot", "SSR4", "R2", "MillerIndex")
    for top in factory._root.children(nmhit.NodeType.Section):
        if top.path() != "Drivers":
            continue
        for drv in top.children(nmhit.NodeType.Section):
            for tt in type_tags:
                for kind, opt_pair in (
                    ("ic", (f"ic_{tt}_names", f"ic_{tt}_values")),
                    ("force", (f"force_{tt}_names", f"force_{tt}_values")),
                ):
                    names_opt, values_opt = opt_pair
                    names_str = drv.param_optional_str(names_opt, "")
                    values_str = drv.param_optional_str(values_opt, "")
                    if not names_str or not values_str:
                        continue
                    names = names_str.split()
                    values = values_str.split()
                    for n, v in zip(names, values, strict=False):
                        key = f"{n}~1" if kind == "ic" else n
                        mapping[key] = (kind, v)

    examples: dict[str, torch.Tensor] = {}
    input_spec = model.input_spec
    for input_name in input_spec:
        if input_name not in mapping:
            continue
        kind, tname = mapping[input_name]
        try:
            tensor_or_wrapper = factory.get_tensor(tname)
        except Exception:
            continue
        t = tensor_or_wrapper.data if hasattr(tensor_or_wrapper, "data") else tensor_or_wrapper
        if not isinstance(t, torch.Tensor):
            continue
        if kind == "ic":
            # IC has shape (*sub_batch, *base); prepend a unit dyn-batch axis.
            t = t.unsqueeze(0)
        else:  # kind == "force"
            # Force trajectory has shape (nstep, *B, *base) or (nstep, *B,
            # *sub_batch, *base); index step 0.
            if t.ndim >= 1:
                t = t.select(0, 0)
        examples[input_name] = t.to(device=device, dtype=torch.float64)
    return examples


def _freeze_remaining_parameters_to_buffers(module: nn.Module) -> None:
    """Convert every remaining ``nn.Parameter`` in *module* to a persistent buffer.

    ``torch.export`` lifts ``nn.Parameter`` instances as graph inputs by
    default; that's the right behavior for training but is incompatible with
    the bake-by-default contract here (the C++ loader has no slot for them).
    Converting them to buffers makes ``torch.export`` treat them as constants
    that get baked into the lowered graph.

    Call this AFTER :func:`_promote_to_nl_params` has removed any promoted
    entries from ``_parameters``; that way only the non-promoted parameters
    get frozen, and promoted ones still flow through the ``*nl_params``
    machinery as runtime inputs.

    The loaded model object is throwaway in this short-lived export process,
    so mutating it in place is fine.
    """
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is None:
                continue
            del sub._parameters[name]
            sub.register_buffer(name, p.data.detach(), persistent=True)


def _resolve_attr(module: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    """Locate the holder submodule and local attribute name for a qualified
    parameter or buffer name (the same form ``named_parameters(recurse=True)``
    emits). Raises if the path component isn't a registered submodule."""
    if "." in qualified:
        path, local = qualified.rsplit(".", 1)
        holder = module.get_submodule(path)
    else:
        holder = module
        local = qualified
    return holder, local


def _validate_promoted(model: nn.Module, promoted: set[str]) -> tuple[list[str], dict[str, str]]:
    """Verify every name in ``promoted`` resolves to a parameter or buffer in
    *model*, AND that its holding submodule is not inside any
    :class:`ImplicitUpdate`'s ``system.model`` tree (promotion of parameters
    inside an implicit segment is not yet supported -- the Dense/Block
    equation-system wrappers have fixed `(u_flat, g_flat)` signatures).

    Returns ``(sorted_names, origin_map)`` where ``origin_map[name]`` is
    ``"parameter"`` or ``"buffer"`` (diagnostic only; the C++ runtime treats
    both identically). Raises ``ValueError`` with a clear listing if any
    name is unmatched, or ``NotImplementedError`` if any name lives inside
    an implicit segment.
    """
    from ..models.common import ImplicitUpdate  # noqa: PLC0415

    params = dict(model.named_parameters(recurse=True))
    buffers = dict(model.named_buffers(recurse=True))
    available = {**params, **buffers}
    missing = sorted(set(promoted) - set(available))
    if missing:
        avail_str = ", ".join(sorted(available)) or "<none>"
        raise ValueError(f"Promoted name(s) not found in model: {missing}. Available: {avail_str}")

    # Build the set of submodule ids that live inside any ImplicitUpdate's
    # system.model. Promoting into one of these would require threading the
    # NL-param tail through DenseRHS/Step/IFT/predictor; not yet implemented.
    implicit_ids: set[int] = set()
    for sub in model.modules():
        if isinstance(sub, ImplicitUpdate):
            for inner in sub.system.model.modules():
                implicit_ids.add(id(inner))

    in_implicit: list[str] = []
    for qname in promoted:
        holder, _ = _resolve_attr(model, qname)
        if id(holder) in implicit_ids:
            in_implicit.append(qname)
    if in_implicit:
        raise NotImplementedError(
            f"Promotion of parameters inside an ImplicitUpdate segment is not "
            f"yet supported (would-be promoted: {sorted(in_implicit)}). The "
            f"implicit segment's DenseRHS/Step/IFT wrappers have fixed "
            f"(u_flat, g_flat) signatures; a follow-up will thread *nl_params "
            f"through them. Compile without these --parameter flags, or move "
            f"the parameters out of the implicit region."
        )

    origin = {n: ("parameter" if n in params else "buffer") for n in promoted}
    return sorted(promoted), origin


def _snapshot_promoted(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    """Return ``{name: detached tensor}`` for the resolved promoted entries.
    Used to (a) build example inputs for the trace and (b) emit initial values
    into the metadata so the C++ side can populate ``named_parameters()``."""
    params = dict(model.named_parameters(recurse=True))
    buffers = dict(model.named_buffers(recurse=True))
    available = {**params, **buffers}
    return {n: available[n].detach() for n in names}


def _promote_to_nl_params(
    model: nn.Module, promoted_qnames: list[str]
) -> dict[str, tuple[type, nn.Module, str]]:
    """Convert each promoted parameter into an NLParam entry on its holding leaf.

    Routes promotion through the existing native-Model nl-params machinery
    rather than reaching in and mutating attribute storage directly:

    1. Locate the holding submodule for *qname*; the holder must be a native
       :class:`~neml2.model.Model` (i.e. have ``_nl_params`` and
       ``input_spec`` -- which is true of every leaf produced by NEML2's
       schema-driven Models).
    2. Look up the parameter's typed-wrapper class (``Scalar``, ``SR2``, ...)
       from the holder's ``_typed_storage_classes`` registry, populated at
       :meth:`register_typed_parameter` time.
    3. Delete the parameter / buffer slot, append the qname to the holder's
       ``input_spec`` (typed via the looked-up class), and add an ``NLParam``
       entry keyed by the *local* attribute name with ``input_name`` set to
       the qname (qualified naming avoids collisions when two leaves both
       promote a same-local-name parameter).

    Returns ``{qname: (type_cls, holder, local_name)}`` so the caller can
    build example inputs at the right shape and emit metadata.

    After this call, any wrapping :class:`ComposedModel` constructed around
    these leaves automatically picks up the new inputs through its standard
    dependency-resolution + ``_coerce_to_input_type`` wrap on the call
    boundary. The leaf's existing ``self._get_param(local, nl_params,
    type_cls)`` call resolves to ``nl_params[tail_index]``.
    """
    from ..model import NLParam  # noqa: PLC0415

    info: dict[str, tuple[type, nn.Module, str]] = {}
    for qname in promoted_qnames:
        holder, local = _resolve_attr(model, qname)
        # Must be a native Model with the nl-params machinery. Non-Model
        # holders (e.g. plain nn.Module wrappers used internally) lack the
        # `_nl_params`/`input_spec` plumbing.
        if not hasattr(holder, "_nl_params") or not hasattr(holder, "input_spec"):
            raise ValueError(
                f"Cannot promote {qname!r}: its holder {type(holder).__name__} is "
                "not a native NEML2 Model with nl-params support. Promotion is "
                "only supported for parameters declared via "
                "register_typed_parameter (i.e. anywhere _get_param is used)."
            )

        # Look up the typed-wrapper class. Native models record this at
        # register_typed_parameter time in _typed_storage_classes.
        type_cls = getattr(holder, "_typed_storage_classes", {}).get(local)
        if type_cls is None:
            raise ValueError(
                f"Cannot promote {qname!r}: holder {type(holder).__name__} has no "
                f"typed-storage class recorded for {local!r}. The parameter must "
                "have been declared via register_typed_parameter (the canonical "
                "NEML2 pattern)."
            )

        # Drop the static slot.
        if local in holder._parameters:
            del holder._parameters[local]
        elif local in holder._buffers:
            del holder._buffers[local]
        else:
            raise ValueError(
                f"Cannot promote {qname!r}: holder has typed-storage class but no "
                f"_parameters/_buffers entry for {local!r}."
            )

        # Promote class-level input_spec to a per-instance copy on first
        # mutation (mirrors declare_typed_parameter's pattern). Pyright sees
        # holder as nn.Module here; we know it's a NEML2 Model with
        # input_spec / _nl_params / _typed_storage_classes (checked above).
        spec: dict[str, type] = holder.input_spec  # type: ignore[attr-defined,assignment]
        if spec is type(holder).__dict__.get("input_spec"):
            spec = dict(spec)
            holder.input_spec = spec  # type: ignore[attr-defined,assignment]
        if qname in spec and spec[qname] is not type_cls:
            raise ValueError(
                f"Cannot promote {qname!r}: input_spec already has an entry of a "
                f"different type ({spec[qname].__name__})."
            )
        spec[qname] = type_cls

        # Add the NLParam entry. tail_index is the slot in the holder's
        # *nl_params pack -- count current entries so concurrent promotions
        # on the same holder get distinct indices. Same type:ignore reason
        # as above (pyright sees nn.Module here).
        nl_params: dict = holder._nl_params  # type: ignore[attr-defined,assignment]
        nl_params[local] = NLParam(
            input_name=qname,
            tail_index=len(nl_params),
        )

        info[qname] = (type_cls, holder, local)
    return info


def _parameter_infos(snapshots: dict[str, torch.Tensor], origin: dict[str, str]) -> list[dict]:
    """Build the v2 metadata ``parameters`` array entries (sorted by name)."""
    infos = []
    for name in sorted(snapshots):
        t = snapshots[name]
        dtype = str(t.dtype).removeprefix("torch.")
        infos.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(t.shape),
                "device": t.device.type,
                "values": t.detach().to(torch.float64).flatten().tolist(),
                "origin": origin[name],
            }
        )
    return infos


def _write_meta(path: Path, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Forward export
# ---------------------------------------------------------------------------


class _ForwardJacobianModule(nn.Module):
    """Wrap a ComposedModel to also emit a flat ``dout/din`` Jacobian.

    Forward signature ``(*inputs) -> (*outputs, J)`` where $J$ has shape
    ``(*B, n_out_total, n_in_total)`` — block-stacked by output then input,
    each block sized by the corresponding variable's storage size. The flat
    layout is identical to the ``inputs``/``outputs`` order in the master's
    input_spec/output_spec.

    Built on the existing first-order chain-rule machinery: we seed a
    leading-K typed identity tangent for each input (one leaf per input, name
    == input name). The ComposedModel pushforward returns typed blocks with K as
    the leftmost batch dim; this wrapper converts those blocks back to the
    public trailing-K flat matrix layout at the export boundary.

    Used by :func:`_compile_forward_segment` to emit ``<basename>_jvp.pt2``.
    """

    def __init__(self, model, promoted_qnames: set[str] | None = None) -> None:
        super().__init__()
        self.model = model
        self.promoted_qnames = set(promoted_qnames or ())
        self.input_names = tuple(model.input_spec)
        self.output_names = tuple(model.output_spec)
        self.in_types = tuple(model.input_spec.values())
        self.out_types = tuple(model.output_spec.values())
        self.in_sizes = tuple(_var_size(t) for t in self.in_types)
        self.out_sizes = tuple(_var_size(t) for t in self.out_types)
        self.in_ndims = tuple(t.BASE_NDIM for t in self.in_types)
        self.out_ndims = tuple(t.BASE_NDIM for t in self.out_types)
        # Indices of inputs that are STRUCTURAL (i.e. user-facing, not
        # promoted). J's columns correspond only to these; promoted inputs
        # are present in the trace's input signature but contribute no
        # tangent (they're treated as constants from J's perspective).
        self.structural_idx = tuple(
            i for i, n in enumerate(self.input_names) if n not in self.promoted_qnames
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Seed identity tangents only for structural inputs. Promoted-param
        # inputs stay absent from the seed dict; the default chain rule's
        # `v.get(name, {})` returns empty for them, so they contribute no
        # block to J.
        seed = {}
        for i in self.structural_idx:
            raw = inputs[i]
            name = self.input_names[i]
            type_cls = self.in_types[i]
            in_ndim = self.in_ndims[i]
            batch_shape = raw.shape if in_ndim == 0 else raw.shape[:-in_ndim]
            seed[name] = {
                name: _leading_k_identity_seed(
                    type_cls,
                    batch_shape,
                    dtype=raw.dtype,
                    device=raw.device,
                )
            }
        result = self.model(*inputs, v=seed)
        result_tuple = result if isinstance(result, tuple) else (result,)
        *raw_outputs, v_out = result_tuple
        # Determine batch shape from the first output.
        first_out = raw_outputs[0]
        out0_ndim = self.out_ndims[0]
        batch_shape = first_out.shape if out0_ndim == 0 else first_out.shape[:-out0_ndim]
        # Stack blocks: row index = output, col index = STRUCTURAL input.
        # Promoted columns are omitted (J's columns are sum(structural_sizes)).
        row_parts: list[torch.Tensor] = []
        for out_name, out_size in zip(self.output_names, self.out_sizes, strict=True):
            col_parts: list[torch.Tensor] = []
            for i in self.structural_idx:
                in_name = self.input_names[i]
                in_size = self.in_sizes[i]
                block = v_out.get(out_name, {}).get(in_name)
                if block is None:
                    block = first_out.new_zeros(*batch_shape, out_size, in_size)
                block = _leading_k_block_to_trailing_k(block)
                col_parts.append(block)
            row_parts.append(torch.cat(col_parts, dim=-1))
        J = torch.cat(row_parts, dim=-2)
        return (*raw_outputs, J)


def _structural_inputs(spec: dict, promoted_qnames: set[str]) -> dict:
    """Return ``spec`` filtered to entries NOT in *promoted_qnames*.

    After :func:`_promote_to_nl_params` has expanded a leaf's input_spec to
    include the promoted qname, the wrapping ComposedModel sees both the
    structural inputs and the promoted ones as positional inputs. The
    metadata's ``inputs`` (user-facing) and per-segment ``inputs`` should
    list only the structural ones; the promoted set is recorded separately
    under ``parameters`` + per-segment ``param_inputs``.
    """
    return {k: v for k, v in spec.items() if k not in promoted_qnames}


def _segment_param_inputs(seg_spec: dict, promoted_qnames: set[str]) -> list[str]:
    """Return the promoted names that appear in this segment's input_spec,
    preserving the spec's declared order (which is the graph-call order)."""
    return [k for k in seg_spec if k in promoted_qnames]


def _build_example_inputs(
    seg_spec: dict,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
    device: str,
) -> tuple:
    """Build the ordered example-input tuple for a segment's trace.

    Structural inputs get the standard ``(2, *BASE_SHAPE)`` zero tensor (the
    batch=2 forces a true dynamic dim). Promoted inputs use the live snapshot
    at its natural shape -- typically ``()`` for a Scalar parameter -- which
    AOTI then broadcasts against the structural batch at runtime.
    """
    examples: list[torch.Tensor] = []
    for name, type_cls in seg_spec.items():
        if name in promoted_qnames:
            snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
            examples.append(snap)
        else:
            examples.append(
                torch.zeros(2, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
            )
    return tuple(examples)


def _compile_forward_segment(
    model,
    pkg_basename: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str] | None = None,
    promoted_snapshots: dict[str, torch.Tensor] | None = None,
    emit_jvp: bool = True,
) -> tuple[str, list[dict], list[dict], str | None, list[str]]:
    """Compile a single forward-shape model to ``<pkg_basename>.pt2`` plus,
    when ``emit_jvp`` is True, ``<pkg_basename>_jvp.pt2`` carrying the flat
    ``dout/din`` Jacobian.

    Promoted parameters are routed through the leaf's NLParam machinery
    (see :func:`_promote_to_nl_params`), which expanded the leaf's
    ``input_spec``. The fresh ``ComposedModel`` wrap below picks up the new
    inputs automatically via dependency resolution; the call boundary's
    ``_coerce_to_input_type`` wraps the raw tensor in the right
    ``TensorWrapper`` before handing it to the leaf.

    Returns ``(pkg_name, input_infos, output_infos, jvp_pkg_name | None,
    segment_param_inputs)`` -- the per-segment metadata fields the caller
    needs.
    """
    from ..export import compile_model
    from ..models.common import ComposedModel

    promoted_qnames = promoted_qnames or set()
    promoted_snapshots = promoted_snapshots or {}

    # Wrap leaf models in ComposedModel so the compiled boundary is plain
    # tensors. After _promote_to_nl_params has modified the leaves, the
    # fresh ComposedModel below picks up the promoted inputs in its
    # input_spec via dependency resolution.
    exportable = model if isinstance(model, ComposedModel) else ComposedModel([model])

    # Move buffers/parameters onto the target device — torch.export refuses
    # to trace a mixed-device graph.
    exportable = exportable.to(device)

    seg_spec = exportable.input_spec
    seg_param_inputs = _segment_param_inputs(seg_spec, promoted_qnames)
    example_inputs = _build_example_inputs(seg_spec, promoted_qnames, promoted_snapshots, device)

    pkg_name = f"{pkg_basename}.pt2"
    compile_model(exportable, example_inputs, output_dir / pkg_name)

    jvp_pkg_name: str | None = None
    if emit_jvp:
        # JVP wrapper differentiates along structural inputs only -- promoted
        # inputs aren't seeded so they contribute structural zeros to J via
        # the default chain rule's ``v.get(name, {})`` empty fallback.
        jvp_module = _ForwardJacobianModule(exportable, promoted_qnames).to(device)
        jvp_pkg_name = f"{pkg_basename}_jvp.pt2"
        compile_model(jvp_module, example_inputs, output_dir / jvp_pkg_name)

    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    output_sparsity = _sparsity_for_outputs(
        exportable, tuple(model.output_spec), tuple(structural_in)
    )
    return (
        pkg_name,
        _var_infos(structural_in),
        _var_infos(model.output_spec, sparsity=output_sparsity),
        jvp_pkg_name,
        seg_param_inputs,
    )


def _export_forward(
    model,
    model_name: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
) -> dict:
    """Export a forward-shape model as a single-segment composed artifact."""
    pkg_name, in_infos, out_infos, jvp_pkg_name, param_inputs = _compile_forward_segment(
        model,
        model_name,
        output_dir,
        device,
        promoted_qnames=promoted_qnames,
        promoted_snapshots=promoted_snapshots,
    )
    seg = {
        "kind": "forward",
        "package": pkg_name,
        "inputs": in_infos,
        "outputs": out_infos,
        "param_inputs": param_inputs,
    }
    if jvp_pkg_name is not None:
        seg["jvp_package"] = jvp_pkg_name
    return {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": in_infos,
        "outputs": out_infos,
        "segments": [seg],
    }


# ---------------------------------------------------------------------------
# Implicit export
# ---------------------------------------------------------------------------


def _compile_implicit_segment(
    inner,
    pkg_basename: str,
    output_dir: Path,
    device: str,
) -> dict:
    """Compile an ImplicitUpdate to ``<pkg_basename>_rhs.pt2`` + ``_step.pt2``
    (+ optional ``_predictor.pt2``), returning the metadata dict (without the
    outer ``"type"`` key — caller adds it).

    Promotion of parameters inside the implicit segment is rejected earlier
    (in :func:`_validate_promoted`); any promoted-parameter machinery is
    handled entirely in the forward path, so this function takes no
    promoted_* kwargs.
    """
    from ..equation_systems import (
        BlockIFT,
        BlockNewtonStep,
        BlockRHS,
        DenseIFT,
        DenseNewtonStep,
        DenseRHS,
    )
    from ..export import compile_model
    from ..models.common import ComposedModel
    from ..solvers import SchurComplement

    system = inner.system
    solver = inner.solver

    # Solver-type dispatch: SchurComplement → multi-group Block path (handles
    # sub-batch and mixed BLOCK+DENSE 2-group factorisation); anything else →
    # single-group Dense path. The C++ orchestrator sees the same flat
    # (u_flat, g_flat) → (u_new, b_new) contract either way — block-ness is
    # internal to the .pt2.
    use_block = isinstance(solver.linear_solver, SchurComplement)

    # nn.Module.to() moves registered parameters and buffers in place; since
    # rhs, step, and ift all wrap the same `system.model`, moving one moves the
    # underlying buffers for all three.
    if use_block:
        rhs = BlockRHS(system).to(device)
        step = BlockNewtonStep(system, solver.linear_solver).to(device)
        ift = BlockIFT(system, solver.linear_solver).to(device)
    else:
        rhs = DenseRHS(system).to(device)
        step = DenseNewtonStep(system).to(device)
        ift = DenseIFT(system).to(device)

    u_size: int = rhs.u_size
    g_size: int = rhs.g_size

    example_u = torch.zeros(2, u_size, dtype=torch.float64, device=device)
    example_g = torch.zeros(2, g_size, dtype=torch.float64, device=device)

    rhs_name = f"{pkg_basename}_rhs.pt2"
    step_name = f"{pkg_basename}_step.pt2"
    ift_name = f"{pkg_basename}_ift.pt2"

    rhs_inputs = (example_u, example_g)
    compile_model(rhs, rhs_inputs, output_dir / rhs_name)
    compile_model(step, rhs_inputs, output_dir / step_name)
    compile_model(ift, rhs_inputs, output_dir / ift_name)

    # The system's unknown/given names ARE the external names — variable
    # resolution happens at construction time inside _store_schema_values, so
    # there is no separate internal/external split anymore.
    ext_unknowns = list(system.unknown_names)
    ext_givens = list(system.given_names)

    def _var_info(layout, name: str, ext_name) -> dict:
        """Per-variable slot info. ``var_size`` is the total flat storage
        (``prod(sub_batch) * base_size``). ``unflattened_shape`` is the
        ``(*sub_batch, *base)`` shape used to reshape the slot back to its
        natural storage on unpack; empty list means a Scalar with no
        trailing storage dim."""
        type_cls = layout.type_of(name)
        sb = tuple(int(s) for s in layout.sub_batch_shape(name))
        base = tuple(int(s) for s in type_cls.BASE_SHAPE)
        sb_total = 1
        for s in sb:
            sb_total *= s
        return {
            "name": ext_name,
            "var_size": sb_total * _var_size(type_cls),
            "var_type": type_cls.__name__,
            "unflattened_shape": list(sb + base),
        }

    unknown_infos = [
        _var_info(system.ulayout, n, ext_unknowns[i]) for i, n in enumerate(system.unknown_names)
    ]
    given_infos = [
        _var_info(system.glayout, n, ext_givens[i]) for i, n in enumerate(system.given_names)
    ]

    seg: dict = {
        "rhs_package": rhs_name,
        "step_package": step_name,
        "ift_package": ift_name,
        "unknowns": unknown_infos,
        "givens": given_infos,
        "u_size": u_size,
        "g_size": g_size,
        "atol": solver.atol,
        "rtol": solver.rtol,
        "miters": solver.miters,
        # Always empty under this iteration's constraint -- promotion inside
        # implicit segments is rejected above. Recorded for schema uniformity.
        "param_inputs": [],
    }

    # Predictor: compile as an extra graph if present. Promoted tail not
    # threaded here either (same reason as the rhs/step/ift block above; the
    # predictor is part of the implicit segment for promotion purposes).
    if inner.predictor is not None:
        pred = inner.predictor.to(device)
        pred_exportable = ComposedModel([pred])
        pred_inputs = _example_inputs_for(pred_exportable, device)
        pred_name = f"{pkg_basename}_predictor.pt2"
        compile_model(pred_exportable, pred_inputs, output_dir / pred_name)
        seg["predictor_package"] = pred_name
        seg["predictor_inputs"] = _var_infos(pred.input_spec)
        seg["predictor_outputs"] = _var_infos(pred.output_spec)

    return seg


def _export_implicit(
    model,
    inner,
    model_name: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
) -> dict:
    """Export an ImplicitUpdate as a single-segment composed artifact.

    Promotion of parameters inside the implicit segment is rejected up-front
    in :func:`_validate_promoted`, so ``promoted_qnames`` only affects the
    metadata's master ``inputs`` filter here -- the trace itself is identical
    to the no-promotion case.
    """
    seg = _compile_implicit_segment(inner, model_name, output_dir, device)
    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    return {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": _var_infos(structural_in),
        "outputs": _var_infos(model.output_spec),
        "segments": [{"kind": "implicit", **seg}],
    }


# ---------------------------------------------------------------------------
# Composed export — ImplicitUpdate as AOTI breakpoint
# ---------------------------------------------------------------------------


def _flatten_composed(model):
    """Yield the leaves of *model* in dependency order.

    Recursively expands nested :class:`ComposedModel`\\ s so the result is a
    flat list of "atomic" models (anything that is not itself a ComposedModel).
    The order respects each ComposedModel's internal dependency resolution and
    the outer model's resolution across its children.
    """
    from ..models.common import ComposedModel

    inner = model
    if isinstance(inner, ComposedModel):
        for attr, *_ in inner._plan:
            yield from _flatten_composed(getattr(inner, attr))
    else:
        yield model  # keep the remapper wrapper, if any, intact


def _contains_implicit(model) -> bool:
    """True iff *model* (recursively expanded) contains an ImplicitUpdate."""
    from ..models.common import ImplicitUpdate

    return any(isinstance(leaf, ImplicitUpdate) for leaf in _flatten_composed(model))


def _partition_into_segments(model):
    """Split *model* (a ComposedModel containing ImplicitUpdate children) into
    a list of segments, alternating forward / implicit at each ImplicitUpdate.

    Returns ``[(kind, payload), ...]`` where ``kind`` is ``"forward"`` (and
    ``payload`` is a list of consecutive non-implicit leaves) or ``"implicit"``
    (and ``payload`` is the single ImplicitUpdate leaf). The returned list
    never contains empty forward segments.
    """
    from ..models.common import ImplicitUpdate

    segments: list[tuple[str, object]] = []
    current_forward: list = []
    for leaf in _flatten_composed(model):
        if isinstance(leaf, ImplicitUpdate):
            if current_forward:
                segments.append(("forward", current_forward))
                current_forward = []
            segments.append(("implicit", leaf))
        else:
            current_forward.append(leaf)
    if current_forward:
        segments.append(("forward", current_forward))
    return segments


def _export_composed(
    model,
    model_name: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
) -> dict:
    """Export a ComposedModel containing ImplicitUpdate children as multiple
    .pt2 artifacts.

    Each non-implicit run of leaves becomes one forward segment .pt2; each
    ImplicitUpdate becomes the standard rhs/step (+ optional predictor) set.
    The C++ ``AOTIModel`` orchestrates the segments in order.
    """
    from ..models.common import ComposedModel, ImplicitUpdate

    segments = _partition_into_segments(model)

    # Each segment's ComposedModel wrapper would otherwise hide variables that
    # are consumed inside the segment, dropping inter-segment data flow on the
    # floor. Compute per-segment `additional_outputs` so every variable any
    # downstream segment or the master needs survives to the orchestrator's
    # state map.
    def _seg_outputs(payload: object) -> set[str]:
        if isinstance(payload, list):
            produced: set[str] = set()
            for leaf in payload:
                produced.update(leaf.output_spec)
            return produced
        return set(payload.output_spec)  # type: ignore[attr-defined]

    def _seg_inputs(payload: object) -> set[str]:
        if isinstance(payload, list):
            consumed: set[str] = set()
            for leaf in payload:
                consumed.update(leaf.input_spec)
            return consumed
        return set(payload.input_spec)  # type: ignore[attr-defined]

    seg_output_sets = [_seg_outputs(payload) for _, payload in segments]
    downstream_demands: list[set[str]] = [set() for _ in segments]
    for j, (_, payload_j) in enumerate(segments):
        inputs_j = _seg_inputs(payload_j)
        for i in range(j):
            downstream_demands[i] |= inputs_j
    master_outs = set(model.output_spec)

    seg_metas: list[dict] = []
    for i, (kind, payload) in enumerate(segments):
        basename = f"{model_name}_seg{i}"
        if kind == "forward":
            # Wrap the segment's leaves in a fresh ComposedModel so the
            # dependency resolver can derive its input/output specs from the
            # leaves' own specs. Reusing the master's _plan would tangle the
            # segment's spec with the master's outer wiring.
            assert isinstance(payload, list)
            needed = (master_outs | downstream_demands[i]) & seg_output_sets[i]
            extra = sorted(needed)
            seg_model = ComposedModel(payload, additional_outputs=extra).to(device)
            pkg_name, in_infos, out_infos, jvp_pkg_name, param_inputs = _compile_forward_segment(
                seg_model,
                basename,
                output_dir,
                device,
                promoted_qnames=promoted_qnames,
                promoted_snapshots=promoted_snapshots,
            )
            seg_entry = {
                "kind": "forward",
                "package": pkg_name,
                "inputs": in_infos,
                "outputs": out_infos,
                "param_inputs": param_inputs,
            }
            if jvp_pkg_name is not None:
                seg_entry["jvp_package"] = jvp_pkg_name
            seg_metas.append(seg_entry)
        else:
            # payload is the ImplicitUpdate leaf.
            impl_model = payload
            assert isinstance(impl_model, ImplicitUpdate)
            seg = _compile_implicit_segment(impl_model, basename, output_dir, device)
            seg_metas.append({"kind": "implicit", **seg})

    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    return {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": _var_infos(structural_in),
        "outputs": _var_infos(
            model.output_spec,
            sparsity=_sparsity_for_outputs(
                model, tuple(model.output_spec), tuple(model.input_spec)
            ),
        ),
        "segments": seg_metas,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_model_for_aoti(
    hit_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float64",
    promoted: set[str] | list[str] | tuple[str, ...] = (),
    additional_args: tuple[str, ...] = (),
) -> dict:
    """Export a native NEML2 model to ``.pt2`` artifacts for AOTI C++ consumption.

    By default every parameter and buffer in the source model is folded into
    the exported graph as a constant -- the most efficient form for inference
    use. To keep a parameter mutable at runtime, pass its fully-qualified name
    via *promoted*; the exporter traces those entries as additional graph
    inputs and records their initial values in the metadata so the C++ side
    can populate ``aoti::Model::named_parameters()``.

    Parameters
    ----------
    hit_path:
        Path to the HIT ``.i`` file.
    model_name:
        Name of the model in the ``[Models]`` section.
    output_dir:
        Directory for ``.pt2`` artifacts and metadata JSON.  Created if absent.
    device:
        Target device for the resulting artifact (default ``"cpu"``).
    dtype:
        Floating-point dtype for the resulting artifact (``"float64"`` or
        ``"float32"``; default ``"float64"``).
    promoted:
        Set / iterable of fully-qualified parameter or buffer names (the same
        form ``model.named_parameters(recurse=True)`` emits) to promote to
        runtime-flexible status. Empty default = fully baked artifact.

    Returns
    -------
    dict
        The metadata dictionary (same content as the written JSON).
    """
    from ..factory import load_input
    from ..models.common import ImplicitUpdate

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = load_input(hit_path, additional_args=additional_args)
    model = factory.get_model(model_name)

    # Validate / resolve promoted names against the live model BEFORE any
    # ComposedModel wrapping (which would shift the qualified-name namespace).
    # Snapshot the initial values now too -- those go into the metadata so
    # the C++ side can populate `named_parameters()` at construction.
    promoted_set = set(promoted)
    promoted_names, origin = _validate_promoted(model, promoted_set)
    promoted_snapshots = _snapshot_promoted(model, promoted_names)

    # Now route each promoted parameter through the leaf's NLParam machinery
    # (delete the static slot, add `_nl_params[local] = NLParam(input_name=
    # qname, ...)`, append qname to holder's input_spec). Any ComposedModel
    # wrap downstream picks the new inputs up via dependency resolution and
    # the call boundary's `_coerce_to_input_type` wraps raw tensors in the
    # right TensorWrapper before handing them to the leaf's forward, which
    # reads via `_get_param` from the *nl_params pack.
    _promote_to_nl_params(model, promoted_names)
    promoted_qnames = set(promoted_names)

    # Freeze any remaining nn.Parameter to a persistent buffer so torch.export
    # bakes it into the graph instead of lifting it as a graph input. Promoted
    # entries are already gone from _parameters so they're skipped naturally.
    _freeze_remaining_parameters_to_buffers(model)

    # Populate per-variable ``sub_batch_shape`` on every inner
    # ``ModelNonlinearSystem`` from HIT-driver-derived example tensors so the
    # Schur/Block export path traces with the correct per-sub-batch-site
    # layout. Skipped silently when the HIT file has no ``[Drivers]`` section.
    driver_inputs = _hit_driver_example_inputs(model, factory, device)
    if driver_inputs:
        _seed_implicit_subbatch(model, driver_inputs, device)

    inner = model

    if isinstance(inner, ImplicitUpdate):
        meta = _export_implicit(
            model,
            inner,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
        )
    elif _contains_implicit(inner):
        meta = _export_composed(
            model,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            promoted_snapshots=promoted_snapshots,
        )
    else:
        meta = _export_forward(
            model,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            promoted_snapshots=promoted_snapshots,
        )

    # v2 top-level additions: device + dtype are baked into the artifact;
    # parameters records the promoted set with initial values.
    meta["device"] = device
    meta["dtype"] = dtype
    meta["parameters"] = _parameter_infos(promoted_snapshots, origin)

    _write_meta(output_dir / f"{model_name}_meta.json", meta)
    return meta


__all__ = ["export_model_for_aoti", "AOTI_META_SCHEMA_VERSION"]
