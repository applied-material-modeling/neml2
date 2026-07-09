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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ..types import TensorWrapper

if TYPE_CHECKING:
    from ..models.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Metadata schema version -- the wire-format handshake between the exporter and
#: the C++/Python loaders. Bumped on every breaking layout change so a loader
#: refuses a mismatched cache with a clear "regenerate via ``neml2-compile``"
#: message rather than failing on a missing field deep in the parser. The
#: canonical value lives in ``scripts/dependencies.yaml``; bump it there with
#: ``scripts/dep_manager.py`` (which keeps this literal, the C++ loader, and the
#: docs in sync).
# dependencies: aoti.schema_version
AOTI_META_SCHEMA_VERSION = 8


def _var_size(type_cls) -> int:
    return prod(type_cls.BASE_SHAPE) if type_cls.BASE_SHAPE else 1


def _enumerate_group_infos(system) -> dict:
    """Canonical per-group metadata for nonlinear-system segments.

    Single source of truth for the iteration order in which group
    tensors appear in RHS/NewtonStep/IFT forward signatures. The
    matching reader side in :mod:`neml2.es.implicit` walks the same
    order via :func:`~neml2.es.implicit.enumerate_group_var_names`.

    Returns a dict with:

    - ``unknown_groups`` / ``given_groups`` / ``residual_groups``:
      ``list[list[str]]`` of variable names per group.
    - ``group_structures``: per-side list of ``"block"`` / ``"dense"``.
    - ``unknown_group_infos`` / ``given_group_infos`` /
      ``residual_group_infos``: per-group dicts with
      ``structure``, ``sub_batch_shape``, ``var_names``, and
      ``per_var_info`` (a list of per-variable
      ``{name, var_size, base_shape, sub_batch_shape}`` dicts ordered
      as in the group).

    The per-(unknown, given) IFT Jacobian-pair metadata is built
    separately in :func:`_compile_implicit_segment` (the IFT graph emits
    per-pair blocks via ``AssembledMatrix.disassemble``).
    """

    def _var_info(layout, name):
        type_cls = layout.specs[name]
        sb = tuple(int(s) for s in layout.sub_batch_shape(name))
        base = tuple(int(s) for s in type_cls.BASE_SHAPE)
        sb_total = 1
        for s in sb:
            sb_total *= s
        return {
            "name": name,
            "var_size": sb_total * _var_size(type_cls),
            "sub_batch_shape": list(sb),
            "base_shape": list(base),
        }

    def _group_infos(layout):
        infos = []
        for gi, group in enumerate(layout.groups):
            structure = layout.structure[gi]
            sb = list(int(s) for s in layout.sub_batch_shape(group[0])) if group else []
            infos.append(
                {
                    "structure": structure,
                    "sub_batch_shape": sb,
                    "var_names": list(group),
                    "per_var_info": [_var_info(layout, n) for n in group],
                }
            )
        return infos

    ulayout = system.ulayout
    glayout = system.glayout
    blayout = system.blayout

    unknown_group_infos = _group_infos(ulayout)
    given_group_infos = _group_infos(glayout)
    residual_group_infos = _group_infos(blayout)

    return {
        "unknown_groups": [list(g) for g in ulayout.groups],
        "given_groups": [list(g) for g in glayout.groups],
        "residual_groups": [list(g) for g in blayout.groups],
        "group_structures": {
            "unknown": list(ulayout.structure),
            "given": list(glayout.structure),
            "residual": list(blayout.structure),
        },
        "unknown_group_infos": unknown_group_infos,
        "given_group_infos": given_group_infos,
        "residual_group_infos": residual_group_infos,
    }


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
    # Declare the leading axis as a K region (one "full" seed axis of size n)
    # rather than folding it into the batch. Derivative-leaf models (e.g.
    # Normality) emit tangents with k_ndim=1; if the seed leaves K folded
    # (k_ndim=0), the two conventions collide in align_k when both feed the
    # same downstream sum (e.g. a frozen flow_direction passed as a given to an
    # ImplicitUpdate solve). Establishing the K region at the seed keeps the
    # whole chain rule consistent.
    return type_cls(data, k_ndim=1, k_state=("full",), k_pairing=(None,))


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


def _leading_k_block_to_per_pair(
    block: torch.Tensor | TensorWrapper,
    in_type_cls: type,
) -> torch.Tensor:
    """Convert a leading-K typed tangent block to per-(out_var, in_var) Jacobian.

    Output shape: ``(*batch, *sub_batch, *out_base, *in_base)`` -- the
    typed natural shape with K (= prod(in_base)) reshaped back into
    the in_var's base axes at the trailing end. Sub_batch axes the
    tangent threaded through are preserved.

    Raw inputs (non-TensorWrapper) are returned unchanged -- they
    already carry the same trailing-K shape that the JVP module's
    zero-fill produces.
    """
    if not isinstance(block, TensorWrapper):
        return block
    # block.data: (K, *batch, *sub, *out_base)
    moved = block.data.movedim(0, -1)  # (*batch, *sub, *out_base, K)
    in_base = tuple(int(s) for s in in_type_cls.BASE_SHAPE)
    if not in_base:
        # Scalar in_var: K=1, drop trailing axis.
        return moved.squeeze(-1)
    return moved.reshape(*moved.shape[:-1], *in_base)


def _var_infos(
    spec: dict,
    *,
    sub_batch_shapes: dict[str, tuple[int, ...]] | None = None,
) -> list[dict]:
    """Build the master-level per-variable metadata list.

    Emits ``name``, ``var_size``, ``var_type``, optional
    ``sub_batch_shape``, and optional ``sub_batch_labels`` for each
    entry. Used at the top-level ``meta["inputs"]`` /
    ``meta["outputs"]`` -- the Python shim consumes ``var_type`` to
    construct ``input_spec`` / ``output_spec``, reads
    ``sub_batch_shape`` to size the dyn-vs-sub batch split for raw-
    tensor inputs, and reads ``sub_batch_labels`` to thread per-axis
    labels through the typed-wrapper re-wrap at the AOTI load
    boundary. C++ consumes ``var_size`` to allocate the dstate
    row-stack blocks. Per-segment metadata uses
    :func:`_segment_var_infos` instead.

    Parameters
    ----------
    spec:
        Ordered ``{name: type_cls}`` mapping (usually a ``Model.input_spec``
        or ``Model.output_spec``).
    sub_batch_shapes:
        Optional ``{name: shape_tuple}`` map; absent entries default to ``()``
        and the field is omitted from the output dict to keep the JSON tight.
    sub_batch_labels:
        Optional ``{name: labels_tuple}`` map; absent / empty entries are
        omitted. Mirrors ``sub_batch_shapes`` -- the labels are per-axis
        strings naming each sub-batch axis (e.g. ``("grain",)``). The shim
        re-attaches them when wrapping the AOTI raw outputs into typed
        wrappers so the per-axis label dispatch survives the
        export-and-load round-trip.
    """
    infos: list[dict] = []
    sub_batch_shapes = sub_batch_shapes or {}
    for k, v in spec.items():
        info: dict = {
            "name": k,
            "var_size": _var_size(v),
            "var_type": v.__name__,
            # Full base shape (e.g. Scalar -> [], SR2 -> [6], R2 -> [3, 3]). The
            # C++ runtime reads this to report input_base_shapes()/
            # output_base_shapes(), validate canonical (*B, *base) inputs, and
            # unflatten jvp/jacobian outputs into variable-pair blocks. var_size
            # is kept (== prod(base_shape)) for the internal flat offset math.
            "base_shape": [int(s) for s in v.BASE_SHAPE],
        }
        sb = sub_batch_shapes.get(k, ())
        if sb:
            info["sub_batch_shape"] = list(sb)
        infos.append(info)
    return infos


def _segment_var_infos(spec_or_infos) -> list[dict]:
    """Build segment-level per-variable metadata.

    Forward-segment + predictor metadata only needs the name; C++ routes
    ``state[name]`` reads/writes by string key and derives per-output
    flat sizes from the live AOTI tensor shape at runtime. ``var_type`` /
    ``var_size`` / ``sub_batch_shape`` were carried for symmetry with the
    master list but never read off the segment dict; v4 drops them.

    Accepts either a ``{name: type_cls}`` spec mapping or a list of
    already-built master-level info dicts; both are common in the
    caller graph.
    """
    if isinstance(spec_or_infos, dict):
        return [{"name": name} for name in spec_or_infos]
    return [{"name": info["name"]} for info in spec_or_infos]


#: Default per-input batch shape when nothing is declared in HIT [Settings] or
#: on the CLI. ``(2,)`` for the dynamic-batch region, ``()`` for sub-batch.
#:
#: ``2`` is the smallest value that still gets ``torch.export`` to install
#: a real dynamic ``Dim`` (a static ``1`` collapses to a constant). The
#: dynamic-batch value seeds Inductor's per-kernel ``size_hints`` and
#: biases the autotune search toward block sizes that match the example.
#: There is no single "right" default: the autotune-optimal example
#: shape is workload-dependent and not predictable from first principles.
#: Measured on the same machine (idle GPU 1) at B=8192:
#:
#:   * scpcoup (low-K, per-slip-pointwise heavy)
#:       example=2 -> 5253 ms      example=8192 -> 2097 ms      (large wins)
#:   * chaboche6 (high-K=43, cuBLAS-LU heavy)
#:       example=2 -> 6425 ms      example=8192 -> 8155 ms      (small wins)
#:
#: The opposite directions reflect different kernel families dominating
#: each workload (Triton per-slip reductions vs cuBLAS-LU/trsm) and
#: different autotune block-size sweet spots. ``(2,)`` is the historical
#: safe default -- never optimal, but never wildly slow either, and easy
#: to reason about. Users who know their production batch should
#: override via ``example_batch_shape=`` on :func:`export_model_for_aoti`
#: or ``--example-batch-shape`` on the CLI; the benchmark suite does
#: this (see ``benchmark/run_benchmark.py``).
_DEFAULT_EXAMPLE_SHAPE: tuple[tuple[int, ...], tuple[int, ...]] = ((2,), ())


def _parse_example_batch_shape(
    spec: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a shape spec string like ``'(2; 100)'`` into ``(dyn, sub)``.

    Grammar (semicolon delimits dynamic-batch from sub-batch axes):

    * ``'(2,)'``           → ``((2,), ())``
    * ``'(2; 100)'``       → ``((2,), (100,))``
    * ``'(2, 3)'``         → ``((2, 3), ())``
    * ``'(2; 100, 5)'``    → ``((2,), (100, 5))``
    * ``'(; 100)'``        → ``((), (100,))``

    V2P-9: the ``:label`` suffix syntax has been removed (chain rule no
    longer dispatches on labels). A leftover ``:foo`` is rejected with
    a clear error.

    Trailing commas inside each region are tolerated. The outer
    parentheses are required; whitespace is ignored.
    """
    s = spec.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(
            f"example_batch_shape spec {spec!r}: must be parenthesized, e.g. '(2,)' or '(2; 100)'."
        )
    body = s[1:-1].strip()
    if ":" in body:
        raise ValueError(
            f"example_batch_shape spec {spec!r}: the ':label' suffix on sub-batch "
            "extents was removed in v2-parity-chain-rule (V2P-9). Drop the suffix "
            "and use positional ordering instead."
        )

    def _split_ints(region: str) -> tuple[int, ...]:
        region = region.strip()
        if not region:
            return ()
        parts = [p.strip() for p in region.split(",")]
        return tuple(int(p) for p in parts if p)

    if ";" in body:
        dyn_str, sub_str = body.split(";", 1)
        return _split_ints(dyn_str), _split_ints(sub_str)
    return _split_ints(body), ()


def _read_settings(factory) -> tuple[dict[str, str], bool]:
    """Read the AOTI-relevant fields from the input file's ``[Settings]`` block.

    Returns ``(example_batch_shape_map, dynamic_batch)`` where:

    * ``example_batch_shape_map`` maps input variable name → spec string
      (e.g. ``"strain" → "(2; 100)"``). Two HIT forms accepted:

        [Settings]
          example_batch_shape = '(2,)'         # uniform → key '*'

        [Settings]
          [example_batch_shape]                # per-variable → one key per entry
            strain      = '(2; 100)'
            temperature = '(2,)'
          []

    * ``dynamic_batch`` is the boolean ``[Settings]/dynamic_batch`` (default
      ``True``).

    Both default cleanly when no ``[Settings]`` block is present.
    """
    import nmhit  # noqa: PLC0415

    example_shapes: dict[str, str] = {}
    dynamic_batch = True

    settings = None
    for top in factory._root.children(nmhit.NodeType.Section):
        if top.path() == "Settings":
            settings = top
            break
    if settings is None:
        return example_shapes, dynamic_batch

    dyn_str = settings.param_optional_str("dynamic_batch", "")
    if dyn_str:
        if dyn_str.lower() in ("true", "1", "yes", "on"):
            dynamic_batch = True
        elif dyn_str.lower() in ("false", "0", "no", "off"):
            dynamic_batch = False
        else:
            raise ValueError(
                f"[Settings]/dynamic_batch={dyn_str!r}: expected boolean (true|false)."
            )

    # ``example_batch_shape`` can be either a Field (uniform) OR a Section
    # (per-variable map). Probe by node type rather than calling
    # ``param_optional_str`` unconditionally -- the latter throws "node has
    # no value" on the Section case.
    uniform = ""
    ebs_node = settings.find("example_batch_shape")
    if ebs_node is not None and ebs_node.type() == nmhit.NodeType.Field:
        # Field form: example_batch_shape = '(2,)'  → uniform, key '*'.
        uniform = settings.param_optional_str("example_batch_shape", "")
        if uniform:
            example_shapes["*"] = uniform

    # Sub-section form: [Settings/example_batch_shape] [strain] [...]  → per-var.
    for child in settings.children(nmhit.NodeType.Section):
        if child.path().rsplit("/", 1)[-1] != "example_batch_shape":
            continue
        if uniform:
            raise ValueError(
                "[Settings]/example_batch_shape: cannot use both the field "
                "(uniform) and sub-section (per-variable) forms in the same file."
            )
        for entry in child.children(nmhit.NodeType.Field):
            var_name = entry.path().rsplit("/", 1)[-1]
            example_shapes[var_name] = entry.param_str()

    return example_shapes, dynamic_batch


def _resolve_example_shapes(
    input_spec: dict,
    declared: dict[str, str],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Map each input name in *input_spec* to its ``(dyn, sub, labels)`` shape tuple.

    Resolution order:

    1. Per-variable entry in *declared* (e.g. ``declared["strain"]``).
    2. Uniform entry in *declared* (key ``"*"``).
    3. :data:`_DEFAULT_EXAMPLE_SHAPE`.

    Unknown keys in *declared* (not in *input_spec*) raise — almost always a
    typo (e.g. ``"stress"`` written instead of ``"strain"``); silently
    ignoring them would mask the bug.
    """
    uniform_spec = declared.get("*")
    extras = set(declared) - {"*"} - set(input_spec)
    if extras:
        raise ValueError(
            f"example_batch_shape names not in model input_spec: {sorted(extras)}. "
            f"Available: {sorted(input_spec)}."
        )
    resolved: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for name in input_spec:
        if name in declared:
            resolved[name] = _parse_example_batch_shape(declared[name])
        elif uniform_spec is not None:
            resolved[name] = _parse_example_batch_shape(uniform_spec)
        else:
            resolved[name] = _DEFAULT_EXAMPLE_SHAPE
    return resolved


def _shared_dyn_shape(
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    relevant_names,
) -> tuple[int, ...]:
    """Return the single dynamic-batch shape shared by all *relevant_names*.

    ``torch.export`` installs a single ``Dim("batch", ...)`` across all
    structural inputs, so they must agree on the leading dynamic shape.
    Per-variable mode allows different sub-batch dims (those are static,
    each input keeps its own), but not different dyn dims. Raise with the
    conflicting entries if they disagree; return ``_DEFAULT_EXAMPLE_SHAPE[0]``
    when no relevant input has a declared shape.
    """
    seen: dict[tuple[int, ...], list[str]] = {}
    for name in relevant_names:
        if name not in shapes:
            continue
        dyn = shapes[name][0]
        seen.setdefault(tuple(dyn), []).append(name)
    if not seen:
        return _DEFAULT_EXAMPLE_SHAPE[0]
    if len(seen) == 1:
        return next(iter(seen))
    parts = [f"{shape}: {names}" for shape, names in sorted(seen.items())]
    raise ValueError(
        "example_batch_shape: inputs must share the same dynamic-batch shape "
        "(torch.export uses a single Dim across all dynamic inputs). Conflict:\n"
        + "\n".join(f"  {p}" for p in parts)
    )


def _validate_baked_against_shapes(
    model,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    promoted_qnames: set[str],
) -> None:
    """Catch baked parameters whose shape would force ``torch.export`` to
    specialize the dynamic batch dim, BEFORE the export runs.

    Specialization happens when a baked tensor and a dynamic-shape input
    participate in the same op with statically equal dims. The most common
    pattern is a parameter whose full shape matches some input's full
    declared shape — e.g. a ``(2,)`` Scalar parameter against a ``(2,)``
    Scalar input. The user must either promote the param via ``-p``, make
    it scalar, or fall back to ``dynamic_batch=false``.

    Only applied when ``dynamic_batch=True`` — under static-batch export
    every dim is pinned by design, so the check is moot.
    """
    # Build the set of "full input shapes" any baked rank-≥1 param would
    # need to broadcast against. We index by `(*dyn, *sub)` so we catch
    # collisions in the batch region — the base region is owned by the
    # wrapper type and won't change at runtime.
    input_leading_shapes = {tuple(dyn) + tuple(sub) for dyn, sub in shapes.values()}

    conflicts: list[tuple[str, tuple[int, ...]]] = []
    for qname, p in model.named_parameters(recurse=True):
        if qname in promoted_qnames:
            continue  # already a graph input — no baking
        if p.ndim == 0:
            continue  # scalar — broadcasts trivially against any shape
        # Walk every declared input's (dyn, sub) prefix. If the param's
        # shape equals (*declared_leading, *param_base_shape) for any input
        # type sharing the same base, it'll specialize that input's dim.
        for name, type_cls in model.input_spec.items():
            if name not in shapes:
                continue
            dyn, sub = shapes[name]
            expected_full = tuple(dyn) + tuple(sub) + tuple(type_cls.BASE_SHAPE)
            if tuple(p.shape) == expected_full:
                conflicts.append((qname, tuple(p.shape)))
                break
        else:
            # Param doesn't match any input's full declared shape, but may
            # still collide with the leading region alone. Flag too.
            if tuple(p.shape) in input_leading_shapes:
                conflicts.append((qname, tuple(p.shape)))

    if not conflicts:
        return

    lines = [f"  - {qname}: shape={shape}" for qname, shape in conflicts]
    raise ValueError(
        "AOTI export: the following baked parameters have shapes that would "
        "specialize the dynamic batch dim (torch.export would emit a "
        '"marked batch as dynamic but specialized to constant N" error):\n'
        + "\n".join(lines)
        + "\n\nThree ways to resolve each:\n"
        "  1. Promote the parameter to a runtime input via "
        "`neml2-compile -p <qname>`. Its shape then participates in the "
        "graph's dynamic-shape signature instead of being baked.\n"
        "  2. Make the parameter scalar (rank-0) so it broadcasts against "
        "any batch — change the [Tensors] expression that binds it.\n"
        "  3. Compile with `--no-dynamic-batch` (or `[Settings] "
        "dynamic_batch = false`). The artifact is then pinned at the "
        "example batch shape; you compile one .pt2 per batch size.\n"
        "\nNote: promotion of parameters inside an `ImplicitUpdate` segment "
        "is not yet supported, so option (1) only works for parameters in "
        "forward-only segments."
    )


def _example_inputs_for(
    model,
    device: str,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> tuple:
    """Return an example input tuple matching ``model.input_spec``.

    Each entry is a :class:`TensorWrapper` instance of the spec's declared
    class with ``data`` shape ``(*dyn, *sub, *type_cls.BASE_SHAPE)`` and
    ``sub_batch_ndim = len(sub)``. The ``(dyn, sub)`` tuple comes from
    *shapes* (defaulted via :func:`_resolve_example_shapes` when omitted).

    Passing typed wrappers through ``torch.export`` records the wrapper
    class in the resulting ``ExportedProgram``'s pytree call_spec, which is
    preserved across the AOTI compile -> ``.pt2`` -> load round-trip. The
    shim consumes typed wrappers at the call boundary and reads
    ``sub_batch_ndim`` directly off the caller arg, eliminating one piece
    of duplicated metadata.
    """
    if shapes is None:
        shapes = _resolve_example_shapes(model.input_spec, {})
    examples: list = []
    for name, type_cls in model.input_spec.items():
        dyn, sub = shapes[name]
        raw = torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
        examples.append(type_cls(raw, sub_batch_ndim=len(sub)))
    return tuple(examples)


def _seed_implicit_subbatch(
    model,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    device: str,
) -> None:
    """Call ``system.initialize(u=, g=)`` on every nested
    :class:`~neml2.models.common.ImplicitUpdate` using zero-tensor stand-ins
    sized at the user-declared ``(dyn, sub)`` shapes. ``initialize`` infers
    per-variable ``sub_batch_shape`` from the input tensor shapes and rebuilds
    the system layouts (D-052 path) without running the Newton solver — the
    cheap, side-effect-only path the Block export needs to know how to size
    its flat ``u_flat`` / ``g_flat`` buffers.

    For each unknown whose matching ``<unknown>~1`` history input has a
    declared shape, the unknown inherits that shape (state-at-end-of-step
    has the same per-site structure as state-at-start-of-step). Otherwise
    the unknown falls back to the ``_DEFAULT_EXAMPLE_SHAPE``.
    """
    from ..models.common import ImplicitUpdate  # noqa: PLC0415

    def _zero_for(type_cls, dyn, sub):
        # Wrap at construction (rule 1): the raw zeros buffer becomes a typed
        # wrapper with the declared sub_batch_ndim right here, never handed
        # raw across an internal neml2 boundary.
        raw = torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
        return type_cls(raw, sub_batch_ndim=len(sub))

    seen: set[int] = set()
    for sub_module in model.modules():
        if not isinstance(sub_module, ImplicitUpdate):
            continue
        if id(sub_module) in seen:
            continue
        seen.add(id(sub_module))
        system = sub_module.system
        sbn: dict[str, int] = {}
        g_dict: dict[str, TensorWrapper] = {}
        for name in system.given_names:
            type_cls = system.model.input_spec[name]
            # ``given_names`` can include ``name~1`` history entries that
            # don't appear in the outer-input ``shapes`` -- fall back to the
            # bare-name shape so a caller-passed uniform shape applies
            # consistently (see u_dict block below for the same fix).
            bare = name.split("~", 1)[0]
            dyn, sub = shapes.get(name, shapes.get(bare, _DEFAULT_EXAMPLE_SHAPE))
            g_dict[name] = _zero_for(type_cls, dyn, sub)
            sbn[name] = len(sub)
        u_dict: dict[str, TensorWrapper] = {}
        for name in system.unknown_names:
            type_cls = system.model.input_spec[name]
            # Unknowns are internal state -- the bare name (e.g.
            # ``elastic_strain``) typically isn't in the outer input_spec,
            # but its ``~1`` history (the IC) is. Mirror the IC's shape so
            # state-at-end-of-step matches state-at-start-of-step
            # structurally. Fall through to bare-name, then default.
            #
            # Without this lookup chain a caller-passed uniform
            # ``example_batch_shape`` ends up applied to the IC inputs (via
            # input_spec) but the unknowns silently use the library
            # default, producing a symbolic-dim mismatch in the explicit-
            # orientation subsystem at torch.export trace time.
            history_name = f"{name}~1"
            dyn, sub = shapes.get(history_name, shapes.get(name, _DEFAULT_EXAMPLE_SHAPE))
            u_dict[name] = _zero_for(type_cls, dyn, sub)
            sbn[name] = len(sub)
        # ``initialize`` needs typed :class:`SparseVector` inputs whose
        # layouts pin the per-variable sub_batch_shape -- without that
        # commitment the system layouts collapse the sub-batch axes into
        # the dynamic batch and the downstream Block tracer assembles a
        # shape-mismatched Jacobian (typical symptom: ``reshape '[*dyn,
        # base, base]' is invalid for input of size *dyn*base`` inside
        # ``_disassemble_flat``). ``to_sparse`` derives the per-variable
        # sub_batch_shapes from each raw tensor's trailing batch dims.
        with torch.no_grad():
            u_sv, g_sv = system.to_sparse(u_dict, g_dict, sbn)
            system.initialize(u=u_sv, g=g_sv)


def _freeze_remaining_parameters_to_buffers(module: nn.Module) -> None:
    """Convert every remaining ``nn.Parameter`` in *module* to a persistent buffer.

    ``torch.export`` lifts ``nn.Parameter`` instances as graph inputs by
    default; that's the right behavior for training but is incompatible with
    the bake-by-default contract here (the C++ loader has no slot for them).
    Converting them to buffers makes ``torch.export`` treat them as constants
    that get baked into the lowered graph.

    Call this AFTER :func:`_promote_parameters` has removed any promoted
    entries from ``_parameters``; that way only the non-promoted parameters
    get frozen, and promoted ones still flow through the ``*promoted_params``
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
    *model*.

    Promotion is supported everywhere a typed parameter lives -- forward leaves,
    a single :class:`ImplicitUpdate`'s residual (schema v7), and (schema v7+)
    inside an implicit child of a composed model, where the multi-segment
    parameter Jacobian carrier composes ``du/dθ`` through the downstream forward
    segments. The actual promoted-parameter plumbing is validated downstream in
    :func:`_promote_parameters` (the holder must be a native Model with
    ``register_typed_parameter`` storage).

    Returns ``(sorted_names, origin_map)`` where ``origin_map[name]`` is
    ``"parameter"`` or ``"buffer"`` (diagnostic only; the C++ runtime treats
    both identically). Raises ``ValueError`` with a clear listing if any name is
    unmatched.
    """
    params = dict(model.named_parameters(recurse=True))
    buffers = dict(model.named_buffers(recurse=True))
    available = {**params, **buffers}
    missing = sorted(set(promoted) - set(available))
    if missing:
        avail_str = ", ".join(sorted(available)) or "<none>"
        raise ValueError(f"Promoted name(s) not found in model: {missing}. Available: {avail_str}")

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


def _validate_renames(
    renames: dict[str, dict[str, str]] | None,
    *,
    input_names: list[str],
    output_names: list[str],
    param_names: list[str],
) -> dict[str, dict[str, str]]:
    """Validate + normalize the boundary-rename maps for the exported artifact.

    *renames* maps each of ``"inputs"`` / ``"outputs"`` / ``"parameters"`` to an
    ``{original_name: boundary_name}`` sub-map (any namespace omitted => no
    renames there). Renaming is *shallow*: only the names a downstream consumer
    sees at the artifact boundary change; the compiled graphs and every internal
    wiring keep the original authored names.

    Each original name must exist in its namespace -- the structural input
    names, the output names, or the promoted-parameter qualified names -- and the
    resulting boundary names must stay unique within the namespace (a new name
    may not collide with another new name or with an unrenamed sibling).
    Identity entries (boundary == original) are dropped. Raises ``ValueError``
    with a clear listing on any violation.

    Returns a normalized ``{"inputs": {...}, "outputs": {...}, "parameters":
    {...}}`` carrying only the non-identity renames.
    """
    allowed = {
        "inputs": list(input_names),
        "outputs": list(output_names),
        "parameters": list(param_names),
    }
    out: dict[str, dict[str, str]] = {"inputs": {}, "outputs": {}, "parameters": {}}
    if not renames:
        return out

    unknown_ns = sorted(set(renames) - set(allowed))
    if unknown_ns:
        raise ValueError(
            f"Unknown rename namespace(s) {unknown_ns}; expected a subset of {sorted(allowed)}."
        )

    for ns, names in allowed.items():
        mapping = renames.get(ns) or {}
        singular = ns[:-1]  # "inputs" -> "input"
        missing = sorted(set(mapping) - set(names))
        if missing:
            avail = ", ".join(names) or "<none>"
            raise ValueError(
                f"Renamed {singular} name(s) not found in model: {missing}. Available: {avail}."
            )
        clean = {o: n for o, n in mapping.items() if o != n}
        final = [clean.get(name, name) for name in names]
        dupes = sorted({n for n in final if final.count(n) > 1})
        if dupes:
            raise ValueError(
                f"Renaming the model's {ns} would produce duplicate boundary name(s): {dupes} "
                f"(a boundary name must be unique among the {ns}, including unrenamed ones)."
            )
        out[ns] = clean
    return out


def _promote_parameters(
    model: nn.Module, promoted_qnames: list[str]
) -> dict[str, tuple[type, nn.Module, str]]:
    """Convert each promoted parameter into an PromotedParam entry on its holding leaf.

    Routes promotion through the existing native-Model promoted-params machinery
    rather than reaching in and mutating attribute storage directly:

    1. Locate the holding submodule for *qname*; the holder must be a native
       :class:`~neml2.model.Model` (i.e. have ``_promoted_params`` and
       ``input_spec`` -- which is true of every leaf produced by NEML2's
       schema-driven Models).
    2. Look up the parameter's typed-wrapper class (``Scalar``, ``SR2``, ...)
       from the holder's ``_typed_storage_classes`` registry, populated at
       :meth:`register_typed_parameter` time.
    3. Delete the parameter / buffer slot, append the qname to the holder's
       ``input_spec`` (typed via the looked-up class), and add an ``PromotedParam``
       entry keyed by the *local* attribute name with ``input_name`` set to
       the qname (qualified naming avoids collisions when two leaves both
       promote a same-local-name parameter).

    Returns ``{qname: (type_cls, holder, local_name)}`` so the caller can
    build example inputs at the right shape and emit metadata.

    After this call, any wrapping :class:`ComposedModel` constructed around
    these leaves automatically picks up the new inputs through its standard
    dependency-resolution + ``_coerce_to_input_type`` wrap on the call
    boundary. The leaf's existing ``self._get_param(local, promoted_params,
    type_cls)`` call resolves to ``promoted_params[tail_index]``.
    """
    from ..models.model import PromotedParam  # noqa: PLC0415

    info: dict[str, tuple[type, nn.Module, str]] = {}
    for qname in promoted_qnames:
        holder, local = _resolve_attr(model, qname)
        # Must be a native Model with the promoted-params machinery. Non-Model
        # holders (e.g. plain nn.Module wrappers used internally) lack the
        # `_promoted_params`/`input_spec` plumbing.
        if not hasattr(holder, "_promoted_params") or not hasattr(holder, "input_spec"):
            raise ValueError(
                f"Cannot promote {qname!r}: its holder {type(holder).__name__} is "
                "not a native NEML2 Model with promoted-params support. Promotion is "
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
        # input_spec / _promoted_params / _typed_storage_classes (checked above).
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

        # Add the PromotedParam entry. tail_index is the slot in the holder's
        # *promoted_params pack -- count current entries so concurrent promotions
        # on the same holder get distinct indices. Same type:ignore reason
        # as above (pyright sees nn.Module here).
        promoted_params: dict = holder._promoted_params  # type: ignore[attr-defined,assignment]
        promoted_params[local] = PromotedParam(
            input_name=qname,
            tail_index=len(promoted_params),
        )

        info[qname] = (type_cls, holder, local)
    return info


def _rebuild_after_promotion(model: Model) -> Model:
    """Rebuild every spec-caching container after :func:`_promote_parameters`.

    Promotion mutates a leaf's ``input_spec`` in place (adds the promoted
    parameter as a graph input), but every enclosing container caches its
    children's specs at construction:

    * :class:`~neml2.models.common.ComposedModel` caches ``input_spec`` /
      ``output_spec`` / ``_plan`` -- a stale plan never routes the new input to
      the leaf, so the leaf's ``forward`` gets too few ``*promoted_params`` and
      :meth:`Model._get_param` raises ``IndexError`` (the symptom seen for a
      promoted weight in a top-level composed model or a ``ScalarConstantParameter``
      provider).
    * ``Normality`` caches ``self.input_spec = dict(inner.input_spec)`` -- a
      promoted parameter inside the inner model never surfaces to the outer graph.
    * :class:`~neml2.models.common.ImplicitUpdate` reuses its system's residual
      model, so the residual ComposedModel must be rebuilt in place.

    This walks the tree bottom-up and rebuilds each container from its (already
    rebuilt) children so all cached specs reflect the promotion. Leaves are
    returned untouched (their ``input_spec`` was updated by the promotion). The
    forward path's old ``ComposedModel([leaf])`` re-wrap only covered a single
    leaf; this covers arbitrary nesting (composed-of-composed, providers, and
    parameters promoted inside a ``Normality`` inner model).
    """
    from ..models.common import ComposedModel, ImplicitUpdate  # noqa: PLC0415

    # Lazy import: Normality lives in the solid_mechanics leaf library. Tolerate
    # its absence (e.g. a trimmed build) -- only models that actually use it need
    # it, and it's always importable in a full install.
    try:
        from ..models.solid_mechanics.plasticity.Normality import Normality  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        Normality = ()  # type: ignore[assignment]

    if isinstance(model, ComposedModel):
        children = [
            _rebuild_after_promotion(cast("Model", getattr(model, attr)))
            for attr, *_ in model._plan
        ]
        old_outputs = list(model.output_spec)
        natural = ComposedModel(children)
        extra = [o for o in old_outputs if o not in natural.output_spec]
        return ComposedModel(children, additional_outputs=extra) if extra else natural

    if Normality and isinstance(model, Normality):
        inner = _rebuild_after_promotion(model._inner)
        return type(model)(inner, model._function, model._from, model._to)

    if isinstance(model, ImplicitUpdate):
        model.system.model = _rebuild_after_promotion(model.system.model)
        return model

    return model


def _parameter_infos(
    snapshots: dict[str, torch.Tensor],
    origin: dict[str, str],
    base_shapes: dict[str, tuple[int, ...]],
    device: str,
) -> list[dict]:
    """Build the metadata ``parameters`` array entries (sorted by name).

    ``shape`` is the stored snapshot shape (carries any batch dim); ``param_base_shape``
    (schema v7) is the parameter's NATURAL base shape from its typed class. The C++
    runtime uses ``param_base_shape`` to split a batched parameter ``(*pbatch, *base)``
    into batch vs base when broadcasting it to the call batch and when reshaping
    parameter-derivative blocks. For an unbatched parameter the two coincide.

    ``device`` is the artifact's compile-target device (e.g. ``"cuda"``), NOT the
    snapshot's own device: the compiled graphs run on the artifact device and, since
    a promoted parameter now enters the value graph as a per-batch ``(B, *base)``
    buffer, the kernel dereferences it as a device pointer. The snapshot is taken from
    the (cpu) source model, so recording its device would land the parameter on cpu and
    feed a host pointer to a cuda kernel -- an illegal memory access. The runtime
    materializes the inlined ``values`` directly on this device.
    """
    infos = []
    for name in sorted(snapshots):
        t = snapshots[name]
        dtype = str(t.dtype).removeprefix("torch.")
        infos.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(t.shape),
                "param_base_shape": [int(s) for s in base_shapes.get(name, tuple(t.shape))],
                "device": device,
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
    """Wrap a ComposedModel to also emit per-(out_var, in_var) Jacobian blocks.

    Forward signature ``(*inputs) -> (*outputs, *J_pairs)`` where each
    ``J_pair`` is the typed Jacobian block for one ``(out_var, in_var)``
    pair at its natural shape ``(*batch, *sub_batch, *out_base, *in_base)``.
    Pairs are emitted in row-major order (outer: outputs in
    ``output_spec`` order; inner: structural inputs in ``input_spec``
    order, skipping promoted parameters which contribute structural
    zero tangents).

    Built on the existing first-order chain-rule machinery: we seed a
    leading-K typed identity tangent for each structural input (K =
    in_var base size). The ComposedModel pushforward returns typed
    blocks with K as the leftmost batch dim; this wrapper moves K to
    the trailing end and reshapes it back into the in_var's base axes.
    Sub_batch axes that the tangent threaded through are preserved on
    each pair tensor (no fold to dense, no N² densification).

    Used by :func:`_compile_forward_segment` to emit
    ``<basename>_jvp.pt2``. The C++ runtime consumes the per-pair
    tensors via per-pair matmul against ``dstate[in_var]``,
    accumulating into ``dseg_out[out_var]`` -- mirrors the IFT cell
    consumer (`jacobian.cpp:_run_implicit_segment_jacobian`).
    """

    def __init__(
        self,
        model,
        promoted_qnames: set[str] | None = None,
        selected_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.promoted_qnames = set(promoted_qnames or ())
        # Local ``(out_var, in_var)`` pairs to emit. ``None`` means all pairs
        # (legacy / all-pairs export); a set restricts the emitted trailing
        # tensors (and matching ``jacobian_pairs`` metadata) to that subset, in
        # the same row-major order — the C++ consumes them positionally.
        self.selected_pairs = selected_pairs
        self.input_names = tuple(model.input_spec)
        self.output_names = tuple(model.output_spec)
        self.in_types = tuple(model.input_spec.values())
        self.out_types = tuple(model.output_spec.values())
        self.in_sizes = tuple(_var_size(t) for t in self.in_types)
        self.out_sizes = tuple(_var_size(t) for t in self.out_types)
        self.in_ndims = tuple(t.BASE_NDIM for t in self.in_types)
        self.out_ndims = tuple(t.BASE_NDIM for t in self.out_types)
        # Indices of inputs that are STRUCTURAL (not promoted). Only
        # structural inputs get tangents and corresponding J_pairs.
        self.structural_idx = tuple(
            i for i, n in enumerate(self.input_names) if n not in self.promoted_qnames
        )

    def forward(self, *inputs) -> tuple:
        typed_inputs = tuple(
            arg if isinstance(arg, type_cls) else type_cls(arg)
            for type_cls, arg in zip(self.in_types, inputs, strict=True)
        )
        seed = {}
        for i in self.structural_idx:
            ti = typed_inputs[i]
            name = self.input_names[i]
            # Seed with a size-1 dynamic-batch identity ``(K, 1..1, *sub, *base)``
            # rather than expanding to the input's full dynamic batch. A
            # Jacobian block that does not depend on the dynamic batch then
            # stays size-1 along those axes (``(1, ...)``), exposing
            # batch-independence (e.g. a constant elasticity tensor); a block
            # that does depend broadcasts up to the real batch through the
            # ordinary chain-rule arithmetic. Real sub-batch extents are kept
            # so the per-site chain rule is unaffected. ``torch.export``
            # preserves the static size-1 axis under a dynamic batch ``Dim``.
            full_batch = tuple(ti.batch_shape)
            sub_ndim = ti.sub_batch_ndim
            dyn_ndim = len(full_batch) - sub_ndim
            seed_batch = (1,) * dyn_ndim + full_batch[dyn_ndim:]
            seed[name] = {
                name: _leading_k_identity_seed(
                    self.in_types[i],
                    seed_batch,
                    dtype=ti.dtype,
                    device=ti.device,
                )
            }
        result = self.model(*typed_inputs, v=seed)
        result_tuple = result if isinstance(result, tuple) else (result,)
        *typed_outputs, v_out = result_tuple
        first_out = typed_outputs[0]
        batch_shape = first_out.batch_shape
        pairs: list[torch.Tensor] = []
        for out_name, out_type in zip(self.output_names, self.out_types, strict=True):
            out_base = tuple(int(s) for s in out_type.BASE_SHAPE)
            for i in self.structural_idx:
                in_name = self.input_names[i]
                if (
                    self.selected_pairs is not None
                    and (out_name, in_name) not in self.selected_pairs
                ):
                    continue
                in_type = self.in_types[i]
                in_base = tuple(int(s) for s in in_type.BASE_SHAPE)
                block = v_out.get(out_name, {}).get(in_name)
                if block is None:
                    # Zero block at natural per-pair shape. No sub axes
                    # since the tangent didn't survive to this pair.
                    pair = torch.zeros(
                        *batch_shape,
                        *out_base,
                        *in_base,
                        dtype=first_out.dtype,
                        device=first_out.device,
                    )
                else:
                    pair = _leading_k_block_to_per_pair(block, in_type)
                pairs.append(pair)
        return (*typed_outputs, *pairs)


def _model_uses_request_ad(model) -> bool:
    """True iff *model* or any submodule declared ``request_AD`` pairs.

    Such a model's chain rule contains an embedded reverse-mode ``torch.autograd.grad``
    at its request_AD leaves (the analytic leaves keep their hand-written forward-mode
    actions). That autograd lowers only under ``trace_autograd_ops`` + ``strict``, so
    the derivative graphs (forward ``jvp`` and the implicit ``NewtonStep`` / ``IFT``)
    are compiled with :func:`_compile_param_derivative_graph` rather than the plain
    forward :func:`~neml2.models.export.compile_model`.
    """
    return any(getattr(m, "_ad_pairs", None) for m in model.modules())


class _ParamJacobianModule(nn.Module):
    """Wrap a (promoted) ComposedModel to emit per-(out_var, param) Jacobian blocks
    ``d(out)/d(param)`` via reverse-mode autograd.

    Forward signature ``(*structural_inputs, *param_inputs) -> (*param_blocks)``
    where each ``param_block`` is the dense Jacobian for one ``(out_var, param)``
    pair at its natural ``(*batch, *out_base, *param_base)`` shape. Blocks are
    emitted row-major: outputs in ``output_spec`` order (outer), promoted params
    in ``input_spec`` order (inner), restricted to *selected_pairs*.

    Two design points (see DECISION: AOTI reverse-mode AD lowering):

    * Reverse-mode ``torch.autograd.grad`` is the ONLY autodiff that lowers
      through ``torch.export`` -> AOTInductor. Export must run ``strict=True``
      with ``torch._dynamo.config.trace_autograd_ops = True`` (set by the
      compile helper), and grad-connected outputs are ``.detach()``-ed (AOTAutograd
      drops ``requires_grad`` on graph outputs).
    * The promoted parameter enters HERE as a PER-BATCH input
      (``(*dyn, *param_base)``), not the scalar the value/jvp graphs use. A scalar
      expanded to the batch inside the graph is a uniform tensor whose value is the
      symbolic input, which trips Inductor's ``constant_fold_uniform_value`` under
      a dynamic batch. A genuine per-batch input is symbolic and never folded. The
      C++ runtime broadcasts the stored scalar parameter to the runtime batch
      before calling this graph.

    One reverse pass per output base-component (cost ~ n_out, independent of the
    number of parameters).
    """

    def __init__(
        self,
        model,
        promoted_qnames: set[str],
        selected_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_names = tuple(model.input_spec)
        self.in_types = tuple(model.input_spec.values())
        self.output_names = tuple(model.output_spec)
        self.out_types = tuple(model.output_spec.values())
        # Promoted parameter inputs, in input_spec order (the differentiation set).
        self.param_names = tuple(n for n in self.input_names if n in promoted_qnames)
        # Structural inputs supply the batch and pass through unchanged.
        self.structural_idx = tuple(
            i for i, n in enumerate(self.input_names) if n not in promoted_qnames
        )
        self.selected_pairs = selected_pairs

    # This body executes only inside `torch.export`'s Dynamo trace at compile time
    # (transformed bytecode), so coverage.py cannot see these lines; the AOTI
    # parameter-Jacobian tests exercise it end-to-end (compile + run + FD). Hence
    # the coverage exclusion on the def below.
    def forward(self, *inputs) -> tuple:  # pragma: no cover
        s0 = inputs[self.structural_idx[0]]
        s0_type = self.in_types[self.structural_idx[0]]
        s0t = s0 if isinstance(s0, s0_type) else s0_type(s0)
        batch = tuple(s0t.batch_shape)

        call_args: list = []
        leaves: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.input_names):
            arg = inputs[i]
            if name in self.param_names:
                # Per-batch parameter input -> a fresh grad-tracking leaf.
                raw = arg.data if isinstance(arg, self.in_types[i]) else arg
                pe = raw.clone().requires_grad_(True)
                leaves[name] = pe
                call_args.append(pe)  # raw; ComposedModel coerces to typed
            else:
                type_cls = self.in_types[i]
                call_args.append(arg if isinstance(arg, type_cls) else type_cls(arg))

        result = self.model(*call_args)
        typed_outputs = result if isinstance(result, tuple) else (result,)

        blocks: list[torch.Tensor] = []
        for o_typed, o_name, o_type in zip(
            typed_outputs, self.output_names, self.out_types, strict=True
        ):
            odata = o_typed.data
            o_base = tuple(int(s) for s in o_type.BASE_SHAPE)
            n_out = 1
            for s in o_base:
                n_out *= s
            for name in self.param_names:
                if self.selected_pairs is not None and (o_name, name) not in self.selected_pairs:
                    continue
                cols: list[torch.Tensor] = []
                for k in range(n_out):
                    seed_flat = torch.zeros(*batch, n_out, dtype=odata.dtype, device=odata.device)
                    seed_flat[..., k] = 1.0
                    seed = seed_flat.reshape(odata.shape)
                    (g,) = torch.autograd.grad(
                        odata,
                        leaves[name],
                        grad_outputs=seed,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    cols.append(g if g is not None else torch.zeros_like(leaves[name]))
                pbase = tuple(leaves[name].shape[len(batch) :])
                block = torch.stack(cols, dim=len(batch)).reshape(*batch, *o_base, *pbase)
                blocks.append(block.detach())
        return tuple(blocks)


class _ParamVJPModule(nn.Module):
    """Wrap a (promoted) ComposedModel to emit the parameter VJP / adjoint
    ``dL/d(param)`` for a loss ``L = sum_o <cotangent_o, out_o>``.

    Forward signature
    ``(*structural_inputs, *param_inputs, *output_cotangents) -> (*param_grads)``:
    the model inputs (in ``input_spec`` order) followed by one cotangent per
    output (in ``output_spec`` order, each at the output's ``(*batch, *out_base)``
    shape). Returns one gradient per promoted parameter (in ``input_spec`` order,
    restricted to *param_names*) at the parameter's natural ``(*param_base)`` shape.

    This is the adjoint operation PDE-constrained inverse optimization uses; it is
    cheaper than the dense Jacobian when there are many parameters (one reverse
    pass total, vs n_out passes). Like :class:`_ParamJacobianModule` the parameter
    enters as a PER-BATCH leaf (the compile helper feeds a ``(*dyn, *param_base)``
    example), so ``autograd.grad`` returns one gradient per batch element rather
    than the batch-summed scalar-leaf gradient. The C++ runtime collapses that to
    match the parameter's stored shape -- summed over the batch for a global
    (unbatched) parameter, kept per-element for a batched one -- so the result
    matches the eager ``param_vjp``. A genuine per-batch input is also symbolic and
    never tripped by ``constant_fold_uniform_value`` under a dynamic batch. Export
    still runs ``strict=True`` with ``trace_autograd_ops`` enabled (set by the
    compile helper).
    """

    def __init__(self, model, param_names: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.input_names = tuple(model.input_spec)
        self.in_types = tuple(model.input_spec.values())
        self.output_names = tuple(model.output_spec)
        # Promoted parameters to differentiate, in input_spec order.
        self.param_names = tuple(n for n in self.input_names if n in set(param_names))

    # This body executes only inside `torch.export`'s Dynamo trace at compile time
    # (transformed bytecode), so coverage.py cannot see these lines; the AOTI
    # parameter-VJP tests exercise it end-to-end (compile + run + FD). Hence the
    # coverage exclusion on the def below.
    def forward(self, *args) -> tuple:  # pragma: no cover
        n_in = len(self.input_names)
        model_args = args[:n_in]
        cotangents = args[n_in:]  # one per output, in output_spec order

        call_args: list = []
        leaves: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.input_names):
            arg = model_args[i]
            if name in self.param_names:
                raw = arg.data if isinstance(arg, self.in_types[i]) else arg  # scalar param
                pe = raw.clone().requires_grad_(True)
                leaves[name] = pe
                call_args.append(pe)
            else:
                type_cls = self.in_types[i]
                call_args.append(arg if isinstance(arg, type_cls) else type_cls(arg))

        result = self.model(*call_args)
        typed_outputs = result if isinstance(result, tuple) else (result,)

        # L = sum_o <cotangent_o, out_o>.
        loss: torch.Tensor | None = None
        for o_typed, cot in zip(typed_outputs, cotangents, strict=True):
            cw = cot.data if hasattr(cot, "data") else cot
            term = (o_typed.data * cw).sum()
            loss = term if loss is None else loss + term
        assert loss is not None  # at least one output -> at least one term

        grads = torch.autograd.grad(
            loss,
            [leaves[p] for p in self.param_names],
            retain_graph=False,
            allow_unused=True,
        )
        return tuple(
            (g if g is not None else torch.zeros_like(leaves[p])).detach()
            for p, g in zip(self.param_names, grads, strict=True)
        )


def _structural_inputs(spec: dict, promoted_qnames: set[str]) -> dict:
    """Return ``spec`` filtered to entries NOT in *promoted_qnames*.

    After :func:`_promote_parameters` has expanded a leaf's input_spec to
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


def _reorder_params_last(exportable, promoted_qnames: set[str]) -> None:
    """Reorder a forward segment's ``ComposedModel`` input_spec so promoted
    parameters come LAST, matching the C++ runtime's input feed order.

    The C++ forward runners build a graph call as ``[structural inputs] +
    [promoted params]`` (``fwd_inputs`` then ``param_inputs``). The traced graph,
    however, consumes inputs in ``ComposedModel.input_spec`` order
    (``forward`` does ``dict(zip(self._in_names, inputs))``). These agree only
    when promoted parameters happen to sort last -- which is NOT guaranteed: the
    dependency resolver can place a parameter promoted on a ``[Models]``
    *provider* (e.g. a ``ScalarConstantParameter`` feeding several consumers)
    BEFORE the structural inputs, silently swapping arguments at runtime. Reorder
    here so structural-then-param is the canonical graph signature for every
    segment. Only ``input_spec`` / ``_in_names`` (the outer arg map) change; the
    per-child ``_plan`` is independent of the outer order. No-op when there are
    no promoted inputs or they are already last.
    """
    from ..models.common import ComposedModel  # noqa: PLC0415

    if not isinstance(exportable, ComposedModel):
        return
    names = list(exportable.input_spec)
    params = [n for n in names if n in promoted_qnames]
    if not params:
        return
    structural = [n for n in names if n not in promoted_qnames]
    new_order = structural + params
    if new_order == names:
        return
    exportable.input_spec = {n: exportable.input_spec[n] for n in new_order}
    exportable._in_names = new_order


def _resolve_derivative_specs(
    specs: Sequence[str],
    output_names: Sequence[str],
    structural_input_names: Sequence[str],
    param_names: Sequence[str] = (),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Resolve ``-d/--derivative`` ``OUT:IN`` specs into master pairs.

    Returns ``(structural_pairs, param_pairs)`` -- the ``(out, in)`` pairs over
    structural inputs and the ``(out, param)`` pairs over promoted parameters,
    respectively. ``IN`` is matched against the structural inputs first, then the
    promoted-parameter names; a parameter derivative is requested by naming the
    promoted parameter (``stress:E`` with ``E`` promoted via ``-p``).

    Each spec must contain exactly one ``:``. Either side may be empty, meaning
    "all" on that side, with one asymmetry: an empty ``IN`` (``stress:``) selects
    all **structural** inputs only -- parameter derivatives are strictly opt-in by
    name, so the common input-Jacobian request is unchanged and never silently
    pulls in (more expensive) parameter graphs. ``:strain`` is every output w.r.t.
    ``strain``; ``:`` is all structural pairs. An empty *specs* yields two empty
    sets -- no derivative graphs are compiled and ``jvp`` / ``jacobian`` raise.

    Unknown output / input names raise ``ValueError`` listing both the structural
    inputs and the promoted parameters available.
    """
    outputs = list(output_names)
    inputs = list(structural_input_names)
    params = list(param_names)
    struct_pairs: set[tuple[str, str]] = set()
    param_pairs: set[tuple[str, str]] = set()
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--derivative {spec!r}: expected OUT:IN (use ':' for all pairs).")
        out_part, in_part = (s.strip() for s in spec.split(":", 1))
        if out_part == "":
            sel_out = outputs
        elif out_part in outputs:
            sel_out = [out_part]
        else:
            raise ValueError(
                f"--derivative {spec!r}: unknown output {out_part!r}; available outputs: {outputs}."
            )
        if in_part == "":
            sel_struct, sel_param = inputs, []  # all structural; params opt-in by name
        elif in_part in inputs:
            sel_struct, sel_param = [in_part], []
        elif in_part in params:
            sel_struct, sel_param = [], [in_part]
        else:
            raise ValueError(
                f"--derivative {spec!r}: unknown input {in_part!r}; available structural "
                f"inputs: {inputs}; promoted parameters: {params or '<none; promote with -p>'}."
            )
        for o in sel_out:
            for i in sel_struct:
                struct_pairs.add((o, i))
            for p in sel_param:
                param_pairs.add((o, p))
    return struct_pairs, param_pairs


def _build_example_inputs(
    seg_spec: dict,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
    device: str,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> tuple:
    """Build the ordered example-input tuple for a segment's trace.

    Structural inputs get a zero tensor of shape ``(*dyn, *sub, *BASE_SHAPE)``
    where ``(dyn, sub)`` comes from *shapes* (defaults to
    :data:`_DEFAULT_EXAMPLE_SHAPE` when omitted). Promoted inputs use the
    live snapshot at its natural shape -- typically ``()`` for a Scalar
    parameter -- which AOTI then broadcasts against the structural batch at
    runtime.

    Critical detail: every example tensor in a segment must use the **same**
    dynamic batch shape. ``torch.export`` installs a single ``Dim("batch")``
    across all inputs and unifies their batch ``SymInt`` *only when the
    example sizes agree at trace time*. If a downstream-state input falls
    back to ``_DEFAULT_EXAMPLE_SHAPE = ((2,), ())`` while the user-supplied
    structural inputs use ``((4,), ())``, pytorch sees two different sizes
    and assigns independent ``SymInt``s. A later binary op that broadcasts
    across them fires ``RuntimeError: tensor a (s21) must match tensor b
    (s39) at non-singleton dimension 0``. So when ``shapes`` provides
    *any* dyn shape, fall back to that shared dyn (via
    :func:`_shared_dyn_shape`) instead of the library default — keeping
    every segment input uniform on the batch axis.
    """
    if shapes is None:
        shapes = {}
    fallback_dyn = (
        _shared_dyn_shape(shapes, list(seg_spec.keys())) if shapes else _DEFAULT_EXAMPLE_SHAPE[0]
    )
    examples: list = []
    for name, type_cls in seg_spec.items():
        if name in promoted_qnames:
            # Promoted parameters arrive as raw ``torch.Tensor`` inputs routed
            # through the per-segment ``param_inputs`` tail; ``ComposedModel.forward``
            # auto-wraps via ``_coerce_to_input_type`` so the leaf still sees a typed
            # value. Feed a genuine PER-BATCH example ``(*dyn, *param_base)`` so
            # ``torch.export`` marks the parameter's leading (batch) dim dynamic --
            # a batched parameter (e.g. a per-batch-element Scalar set via
            # ``set_parameter``) then flows through the compiled value graph, and the
            # C++ runtime broadcasts a stored scalar parameter up to the call batch.
            # The example value is symbolic at runtime, so the snapshot broadcast is
            # only a representative operating point; it lives on the artifact's target
            # ``device`` (matching the recorded parameter device in the metadata).
            base = tuple(int(s) for s in type_cls.BASE_SHAPE)
            snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
            examples.append(snap.broadcast_to((*fallback_dyn, *base)).contiguous())
        else:
            # Segments can have names that the outer input_spec doesn't
            # surface, in two flavours:
            #
            # * ``name~N`` history inputs (e.g. ``elastic_strain~1``) where
            #   only the bare name ``elastic_strain`` is exposed -- strip
            #   the suffix and try the bare key.
            # * "downstream" state inputs (e.g. ``elastic_strain``) that
            #   flow from an earlier subsystem and aren't outer inputs, but
            #   whose lagged ``~1`` history IS in the outer input_spec --
            #   add the ``~1`` suffix and try that key.
            #
            # Either direction lets a caller-passed uniform shape apply
            # consistently; otherwise downstream segments end up using the
            # library default and the trace sees a symbolic-dim mismatch
            # against the structural inputs.
            bare = name.split("~", 1)[0]
            dyn, sub = shapes.get(
                name,
                shapes.get(
                    bare,
                    shapes.get(f"{bare}~1", (fallback_dyn, ())),
                ),
            )
            raw = torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
            # Wrap with the declared ``sub_batch_ndim`` so the exported
            # graph sees a structurally-typed input -- the wrapper class
            # and per-input sub_batch_ndim flow into the .pt2's TreeSpec
            # call signature.
            examples.append(type_cls(raw, sub_batch_ndim=len(sub)))
    return tuple(examples)


def _report(progress_cb: Callable[[str], None] | None, name: str) -> None:
    """Fire the per-file progress callback for a just-generated artifact, if set.

    *name* is the bare filename (``<basename>.pt2`` / ``<model>_meta.json`` /
    ``<model>_aoti.i``) the caller just wrote. The single-argument callback is
    threaded from :func:`export_model_for_aoti` (and the CLI) so every generated
    file -- ``.pt2``, ``.json``, and ``.i`` -- is reported through one channel.
    ``None`` (the default everywhere) is a no-op, preserving the silent library
    behavior.
    """
    if progress_cb is not None:
        progress_cb(name)


def _compile_forward_segment(
    model,
    pkg_basename: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str] | None = None,
    promoted_snapshots: dict[str, torch.Tensor] | None = None,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    selected_pairs: set[tuple[str, str]] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[str, list[dict], list[dict], str | None, list[str], list[dict] | None]:
    """Compile a single forward-shape model to ``<pkg_basename>.pt2`` plus,
    when derivatives are requested, ``<pkg_basename>_jvp.pt2`` carrying the
    per-(out, in) Jacobian blocks.

    *selected_pairs* selects which local ``(out_var, in_var)`` blocks to emit:
    ``None`` emits all pairs (legacy / all-pairs export); an empty set emits no
    derivative graph at all (forward value only); a non-empty set restricts the
    emitted blocks (and matching ``jacobian_pairs`` metadata) to that subset.

    Promoted parameters are routed through the leaf's PromotedParam machinery
    (see :func:`_promote_parameters`), which expanded the leaf's
    ``input_spec``. The fresh ``ComposedModel`` wrap below picks up the new
    inputs automatically via dependency resolution; the call boundary's
    ``_coerce_to_input_type`` wraps the raw tensor in the right
    ``TensorWrapper`` before handing it to the leaf.

    Returns ``(pkg_name, input_infos, output_infos, jvp_pkg_name | None,
    segment_param_inputs)`` -- the per-segment metadata fields the caller
    needs.
    """
    from ..models.common import ComposedModel
    from ..models.export import compile_model

    promoted_qnames = promoted_qnames or set()
    promoted_snapshots = promoted_snapshots or {}

    # Wrap leaf models in ComposedModel so the compiled boundary is plain
    # tensors. After _promote_parameters has modified the leaves, the
    # fresh ComposedModel below picks up the promoted inputs in its
    # input_spec via dependency resolution.
    exportable = model if isinstance(model, ComposedModel) else ComposedModel([model])
    # Canonical structural-then-param graph signature (matches the C++ feed).
    _reorder_params_last(exportable, promoted_qnames)

    # Move buffers/parameters onto the target device — torch.export refuses
    # to trace a mixed-device graph.
    exportable = exportable.to(device)

    seg_spec = exportable.input_spec
    seg_param_inputs = _segment_param_inputs(seg_spec, promoted_qnames)
    example_inputs = _build_example_inputs(
        seg_spec, promoted_qnames, promoted_snapshots, device, shapes=shapes
    )
    dynamic_dim = 0 if dynamic_batch else None

    pkg_name = f"{pkg_basename}.pt2"
    compile_model(exportable, example_inputs, output_dir / pkg_name, dynamic_batch_dim=dynamic_dim)
    _report(progress_cb, pkg_name)

    jvp_pkg_name: str | None = None
    jacobian_pairs: list[dict] | None = None
    do_jvp = selected_pairs is None or len(selected_pairs) > 0
    if do_jvp:
        # JVP wrapper differentiates along structural inputs only -- promoted
        # inputs aren't seeded so they contribute structural zeros via the default
        # chain rule's ``v.get(name, {})`` empty fallback. The SAME forward-mode
        # chain-rule module is used whether or not the model uses request_AD: the
        # analytic leaves emit their hand-written actions, and a request_AD leaf's
        # reverse-mode local Jacobian traces inline (the guard is inert while
        # compiling -- see _guard._armed). A request_AD model's chain rule contains
        # that embedded ``torch.autograd.grad``, which lowers only under
        # ``trace_autograd_ops`` + ``strict``, so its JVP graph uses the
        # parameter-derivative compile helper; an all-analytic model uses the plain
        # forward compile.
        uses_ad = _model_uses_request_ad(exportable)
        jvp_module = _ForwardJacobianModule(
            exportable, promoted_qnames, selected_pairs=selected_pairs
        ).to(device)
        # Probe eagerly first (example dynamic batch is >=2 by construction, see
        # _DEFAULT_EXAMPLE_SHAPE) to classify each emitted pair: a block whose
        # dynamic-batch axes stay size-1 does not depend on the runtime batch
        # (e.g. a constant elasticity tensor) and is recorded as
        # ``batch_independent`` so the runtime can carry / return it unbatched. The
        # eager request_AD pushforward re-enables grad itself, so the no_grad probe
        # is fine even when an AD leaf is present.
        with torch.no_grad():
            probe = jvp_module(*example_inputs)
        n_outs = len(jvp_module.output_names)
        probe_pairs = list(probe[n_outs:])
        jvp_pkg_name = f"{pkg_basename}_jvp.pt2"
        if uses_ad:
            _compile_param_derivative_graph(
                jvp_module, example_inputs, output_dir / jvp_pkg_name, dynamic_batch_dim=dynamic_dim
            )
        else:
            compile_model(
                jvp_module, example_inputs, output_dir / jvp_pkg_name, dynamic_batch_dim=dynamic_dim
            )
        _report(progress_cb, jvp_pkg_name)
        # Per-(out_var, in_var) pair metadata in the SAME order the JVP
        # module emits trailing tensors: rows-outer (output_spec order),
        # cols-inner (structural inputs in input_spec order).
        structural_in_spec = _structural_inputs(model.input_spec, promoted_qnames)
        shapes_local = shapes or {}
        jacobian_pairs = []
        k = 0
        for out_name, out_type in model.output_spec.items():
            out_base_ndim = len(out_type.BASE_SHAPE)
            for in_name, in_type in structural_in_spec.items():
                if selected_pairs is not None and (out_name, in_name) not in selected_pairs:
                    continue
                in_sub = shapes_local.get(in_name, ((), ()))[1]
                in_base_ndim = len(in_type.BASE_SHAPE)
                pair_t = probe_pairs[k]
                # Pair shape is (*dyn, *sub, *out_base, *in_base); the leading
                # dyn axes are size-1 iff this block is batch-independent.
                dyn_ndim = max(pair_t.dim() - len(in_sub) - out_base_ndim - in_base_ndim, 0)
                batch_independent = bool(dynamic_batch) and all(
                    int(pair_t.shape[d]) == 1 for d in range(dyn_ndim)
                )
                jacobian_pairs.append(
                    {
                        "out_var": out_name,
                        "in_var": in_name,
                        "out_base_shape": [int(s) for s in out_type.BASE_SHAPE],
                        "in_base_shape": [int(s) for s in in_type.BASE_SHAPE],
                        "in_sub_batch_shape": [int(s) for s in in_sub],
                        "batch_independent": batch_independent,
                    }
                )
                k += 1

    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    in_sb = {n: sub for n, (_, sub) in (shapes or {}).items() if sub}
    return (
        pkg_name,
        _var_infos(structural_in, sub_batch_shapes=in_sb),
        _var_infos(model.output_spec),
        jvp_pkg_name,
        seg_param_inputs,
        jacobian_pairs,
    )


#: AOTInductor cannot lower a reverse-mode-autograd graph through an operator
#: whose backward SAVES ITS OUTPUT (``sqrt`` / ``reciprocal`` / ``exp`` / ``tanh``
#: / ``sigmoid`` ..., hence a von-Mises stress / norm) under a dynamic batch +
#: strict export: AOTAutograd materialises that saved-output activation as a
#: lifted constant with a symbolic (batch) shape, which Inductor can neither
#: inline nor serialise. (Ops that save their INPUT -- ``log``, ``pow``, ``x*x``
#: -- reference the placeholder instead and lower fine, so the trigger is the
#: saved output, not a division in the backward: ``exp`` has no division yet
#: fails, ``log`` divides yet succeeds.) Filed upstream as
#: https://github.com/pytorch/pytorch/issues/187907. These are the substrings of
#: the (InductorError-wrapped) RuntimeError messages that failure produces --
#: ``storage_offset`` (example batch <= 8, inline path), ``size_bytes_is_heap_
#: allocated_`` (example batch > 8), and ``data is not allocated`` (static batch).
#: Matching the message (not the torch version) makes the guard self-disable the
#: moment the upstream fix stops producing the error -- we cannot know the fix
#: version in advance. The COMMON saved-output ops are already worked around at
#: the typed-wrapper level (``neml2/types/functions.py`` ``_recompute_unary`` +
#: the division-dunder routing), so this guard is a BACKSTOP: it fires only when
#: a model's differentiated path uses a saved-output op that is not yet wrapped.
_AOTI_PARAM_DERIV_BUG_SIGNATURES = (
    "storage_offset() on tensor with symbolic",
    "size_bytes_is_heap_allocated_",
    "tensor has a non-zero number of elements, but its data is not allocated",
)

_AOTI_PARAM_DERIV_BUG_HINT = (
    "neml2-compile: AOTInductor could not lower the parameter-derivative graph "
    "'{graph}'. This model's differentiated path contains an operator whose "
    "reverse-mode backward saves its output. This is a known upstream PyTorch "
    "limitation, not a NEML2 bug: https://github.com/pytorch/pytorch/issues/187907."
    "\n\n"
    "NEML2 already works around the COMMON saved-output ops (sqrt / exp / tanh / "
    "reciprocal and division by a differentiated value) via input-recompute "
    "autograd Functions in neml2/types/functions.py, so most models compile. "
    "Hitting this means the differentiated path uses a saved-output op that is "
    "NOT yet wrapped (e.g. a raw torch.<op> call inside a leaf, or an op such as "
    "sigmoid with no typed wrapper). Fixes, in order of preference:\n"
    "  1. Route the offending op through its AOTI-safe wrapper in "
    "neml2/types/functions.py (add one via `_recompute_unary` if missing); or\n"
    "  2. Compute parameter derivatives for this model through the EAGER route, "
    "which never lowers a graph and is always safe:\n"
    "       - Python:  neml2.eager (param_jacobian / param_vjp)\n"
    "       - C++:     the cpp-eager runtime\n"
    "  3. Recompile without `-d <out>:<param>` for the affected parameters if you "
    "only need the value / input Jacobian.\n\n"
    "The original AOTInductor error is chained below for reference."
)


def _reverse_ad_aoti_unsupported_reason() -> str | None:
    """Why reverse-mode-AD AOTI graphs (parameter derivatives / request_AD) cannot
    be lowered on the running torch, or ``None`` if they can.

    Two torch versions are excluded:

    * torch < 2.11 lacks ``torch._dynamo.config.trace_autograd_ops`` -- the only
      configuration in which ``torch.autograd.grad`` lowers through AOTInductor.
    * torch 2.11.x has that config but its ``torch._dynamo`` rejects
      ``Tensor.requires_grad_()`` under strict export
      (``torch._dynamo.exc.Unsupported``), which every reverse-AD graph hits;
      fixed in torch 2.12.

    Forward / jvp / jacobian compilation is unaffected by either. The tests that
    exercise parameter-derivative / request_AD AOTI graphs skip on the same
    predicate, so the runtime guard and the test gate can't drift.
    """
    import torch._dynamo  # noqa: PLC0415

    if not hasattr(torch._dynamo.config, "trace_autograd_ops"):
        return (
            "requires torch >= 2.11 (this torch lacks "
            "torch._dynamo.config.trace_autograd_ops, the only configuration in "
            "which torch.autograd.grad lowers through AOTInductor)"
        )
    parts = str(torch.__version__).split("+", 1)[0].split(".")
    major, minor = int(parts[0]), int(parts[1])
    if (major, minor) == (2, 11):
        return (
            "is not supported on torch 2.11.x (torch._dynamo rejects "
            "Tensor.requires_grad_() under strict export; fixed in torch 2.12)"
        )
    return None


def _compile_param_derivative_graph(module, example_inputs, path, *, dynamic_batch_dim) -> None:
    """``compile_model`` for a parameter-derivative graph (strict + reverse-mode
    autograd), with a clear error on the known upstream AOTInductor lowering bug.

    Sets ``trace_autograd_ops`` (the only configuration in which
    ``torch.autograd.grad`` lowers), compiles ``strict=True``, and -- if the
    compile hits the documented reverse-mode-AD constant-handling failure
    (pytorch/pytorch#187907) -- re-raises a ``NotImplementedError`` that names the
    workaround instead of a cryptic Inductor traceback. Any OTHER error
    propagates unchanged, and the original error is always chained.
    """
    import torch._dynamo  # noqa: PLC0415

    from ..models.export import compile_model  # noqa: PLC0415

    reason = _reverse_ad_aoti_unsupported_reason()
    if reason is not None:
        raise NotImplementedError(
            f"Parameter-derivative / request_AD AOTI compilation {reason}. Forward / "
            "jvp / jacobian compilation is unaffected."
        )

    prev = torch._dynamo.config.trace_autograd_ops
    torch._dynamo.config.trace_autograd_ops = True
    try:
        compile_model(
            module, example_inputs, path, dynamic_batch_dim=dynamic_batch_dim, strict=True
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(sig in msg for sig in _AOTI_PARAM_DERIV_BUG_SIGNATURES):
            raise NotImplementedError(
                _AOTI_PARAM_DERIV_BUG_HINT.format(graph=Path(path).name)
            ) from exc
        raise
    finally:
        torch._dynamo.config.trace_autograd_ops = prev


def _compile_param_jacobian(
    model,
    pkg_basename: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
    selected_param_pairs: set[tuple[str, str]],
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    """Compile the parameter-Jacobian graph to ``<pkg_basename>_pjac.pt2``.

    Returns ``(pkg_name, param_jacobian_pairs)``. The graph emits dense
    ``d(out)/d(param)`` blocks (one per requested ``(out, param)`` pair) via
    reverse-mode autograd in :class:`_ParamJacobianModule`. Promoted parameters
    are fed as PER-BATCH inputs (``(*dyn, *param_base)``) so the export does not
    materialize a uniform-from-scalar tensor under a dynamic batch; the value /
    input-jvp graphs keep them scalar. Export runs ``strict=True`` with
    ``torch._dynamo.config.trace_autograd_ops`` enabled -- the only configuration
    in which ``torch.autograd.grad`` lowers through AOTInductor (torch 2.12).
    """
    from ..models.common import ComposedModel  # noqa: PLC0415

    exportable = model if isinstance(model, ComposedModel) else ComposedModel([model])
    _reorder_params_last(exportable, promoted_qnames)
    exportable = exportable.to(device)
    seg_spec = exportable.input_spec

    shapes = shapes or {}
    fallback_dyn = (
        _shared_dyn_shape(shapes, list(seg_spec.keys())) if shapes else _DEFAULT_EXAMPLE_SHAPE[0]
    )
    # Example inputs: structural at (*dyn, *sub, *base); promoted parameters at
    # (*dyn, *param_base) -- the per-batch form the derivative graph needs.
    examples: list = []
    for name, type_cls in seg_spec.items():
        if name in promoted_qnames:
            # Per-batch example ``(*dyn, *param_base)`` at the parameter's NATURAL
            # base (from the typed class). The example value is the parameter's actual
            # SNAPSHOT broadcast to the batch -- the reverse-mode param-derivative
            # graph BAKES THE OPERATING POINT, so the example must be a representative
            # parameter value (a random example yields a wrong derivative). Broadcasting
            # the snapshot also keeps the build deterministic / reproducible.
            base = tuple(int(s) for s in type_cls.BASE_SHAPE)
            snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
            examples.append(snap.broadcast_to((*fallback_dyn, *base)).contiguous())
        else:
            bare = name.split("~", 1)[0]
            dyn, sub = shapes.get(
                name, shapes.get(bare, shapes.get(f"{bare}~1", (fallback_dyn, ())))
            )
            raw = torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
            examples.append(type_cls(raw, sub_batch_ndim=len(sub)))
    example_inputs = tuple(examples)

    module = _ParamJacobianModule(
        exportable, promoted_qnames, selected_pairs=selected_param_pairs
    ).to(device)

    pkg_name = f"{pkg_basename}_pjac.pt2"
    dynamic_dim = 0 if dynamic_batch else None
    _compile_param_derivative_graph(
        module, example_inputs, output_dir / pkg_name, dynamic_batch_dim=dynamic_dim
    )
    _report(progress_cb, pkg_name)

    # Per-(out_var, param) metadata in the SAME order the module emits blocks:
    # outputs outer (output_spec order), promoted params inner (input_spec order).
    param_pairs: list[dict] = []
    for out_name, out_type in exportable.output_spec.items():
        for p_name in seg_spec:
            if p_name not in promoted_qnames or (out_name, p_name) not in selected_param_pairs:
                continue
            param_pairs.append(
                {
                    "out_var": out_name,
                    "param": p_name,
                    "out_base_shape": [int(s) for s in out_type.BASE_SHAPE],
                    "param_base_shape": [int(s) for s in seg_spec[p_name].BASE_SHAPE],
                }
            )
    return pkg_name, param_pairs


def _compile_param_vjp(
    model,
    pkg_basename: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
    selected_params: set[str],
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Compile the parameter VJP / adjoint graph to ``<pkg_basename>_pvjp.pt2``.

    Returns ``(pkg_name, param_vjp_params, param_vjp_outputs)`` -- the parameters
    differentiated (in ``input_spec`` order, the grad-output order) and the
    outputs whose cotangents the graph consumes (in ``output_spec`` order, the
    cotangent-input order). The graph computes the per-batch-element adjoint
    ``d(<cotangents, outputs>)/d(param)`` for ``L = sum_o <cotangent_o, out_o>``
    via :class:`_ParamVJPModule`. Promoted parameters are fed PER-BATCH
    (``(*dyn, *param_base)``, like the param-Jacobian graph), so ``autograd.grad``
    returns one gradient per batch element ``(*dyn, *param_base)`` -- the C++
    runtime then sums over the batch for an unbatched (global) stored parameter
    and keeps the per-element gradient for a batched one, matching the eager
    ``param_vjp``. Export runs ``strict=True`` with ``trace_autograd_ops`` enabled.
    """
    from ..models.common import ComposedModel  # noqa: PLC0415

    exportable = model if isinstance(model, ComposedModel) else ComposedModel([model])
    _reorder_params_last(exportable, promoted_qnames)
    exportable = exportable.to(device)
    seg_spec = exportable.input_spec
    out_spec = exportable.output_spec

    shapes = shapes or {}
    fallback_dyn = (
        _shared_dyn_shape(shapes, list(seg_spec.keys())) if shapes else _DEFAULT_EXAMPLE_SHAPE[0]
    )
    # Model inputs (structural typed + promoted parameters PER-BATCH, so the
    # adjoint is per-batch-element; a genuine per-batch input is also symbolic and
    # never constant-folded under a dynamic batch), then one cotangent per output
    # at (*dyn, *out_base). The parameter example is the SNAPSHOT broadcast to the
    # batch -- the param-derivative graph bakes the operating point.
    examples: list = []
    for name, type_cls in seg_spec.items():
        if name in promoted_qnames:
            base = tuple(int(s) for s in type_cls.BASE_SHAPE)
            snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
            examples.append(snap.broadcast_to((*fallback_dyn, *base)).contiguous())
        else:
            bare = name.split("~", 1)[0]
            dyn, sub = shapes.get(
                name, shapes.get(bare, shapes.get(f"{bare}~1", (fallback_dyn, ())))
            )
            raw = torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)
            examples.append(type_cls(raw, sub_batch_ndim=len(sub)))
    # Output cotangents must be TYPED wrappers (matching each output's class), not
    # raw tensors: a raw cotangent input makes Inductor mis-serialize a baked
    # constant ("size_bytes_is_heap_allocated_") at AOTI codegen; a typed
    # (pytree-registered) cotangent compiles cleanly.
    for o_type in out_spec.values():
        cot = torch.ones(*fallback_dyn, *o_type.BASE_SHAPE, dtype=torch.float64, device=device)
        examples.append(o_type(cot))
    example_inputs = tuple(examples)

    module = _ParamVJPModule(exportable, tuple(selected_params)).to(device)

    pkg_name = f"{pkg_basename}_pvjp.pt2"
    dynamic_dim = 0 if dynamic_batch else None
    _compile_param_derivative_graph(
        module, example_inputs, output_dir / pkg_name, dynamic_batch_dim=dynamic_dim
    )
    _report(progress_cb, pkg_name)

    return pkg_name, list(module.param_names), list(out_spec)


# ---------------------------------------------------------------------------
# Implicit export
# ---------------------------------------------------------------------------


def _compile_implicit_segment(
    inner,
    pkg_basename: str,
    output_dir: Path,
    device: str,
    *,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    emit_ift: bool = True,
    selected_ift_pairs: set[tuple[str, str]] | None = None,
    promoted_qnames: set[str] | None = None,
    promoted_snapshots: dict[str, torch.Tensor] | None = None,
    selected_param_pairs: set[tuple[str, str]] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Compile an ImplicitUpdate to ``<pkg_basename>_rhs.pt2`` + ``_step.pt2``
    (+ ``_ift.pt2`` when *emit_ift*) (+ ``_pift.pt2`` when *selected_param_pairs*)
    (+ optional ``_predictor.pt2``), returning the metadata dict (without the
    outer ``"type"`` key — caller adds it).

    *selected_ift_pairs* restricts the IFT graph to the requested
    ``(unknown, given)`` pairs (``None`` = all). A requested pair whose unknown
    or given is **sub-batched** (per-grain, e.g. crystal plasticity) is rejected
    here with a clear compile-time error: the per-pair IFT consumer only supports
    plain-batch (DENSE) pairs today, so e.g. a global-output / global-input
    derivative of a crystal-plasticity model compiles, while a per-grain pair
    fails fast at ``neml2-compile`` rather than at runtime.

    ``rhs`` + ``step`` drive the forward Newton solve and are always compiled.
    ``ift`` is the user-facing input-derivative graph (the implicit-function-
    theorem sensitivity ``du/dg``); it is compiled only when *emit_ift* is True
    (some ``-d`` pair targets a structural input). ``pift`` is the parameter
    sensitivity graph (:class:`~neml2.es.implicit.ParamIFT`, ``du/dθ``); it is
    compiled only when *selected_param_pairs* is non-empty (some ``-d`` pair
    targets a promoted parameter inside the residual). When a graph is not
    compiled its package key is omitted and the runtime leaves that loader null.

    Per-variable I/O at the AOTI graph boundary (v5+): each unknown,
    given, and residual is its own positional tensor in the segment
    signature. Preserved-label per-group sub_batch storage stays
    heterogeneous-ndim end-to-end; no cross-group cat or fold-to-flat
    inside the graph (except IFT's once-per-solve flat ``du/dg`` slab).

    Promoted parameters living inside the implicit residual (schema v7) are
    threaded as a positional tail through every graph. ``rhs`` / ``step`` / ``ift``
    take the stored SCALAR (constant across the solve / input-Jacobian); the C++
    runtime passes ``_gather_params(seg.param_inputs)`` for them. ``pift`` takes
    the parameter PER-BATCH (the reverse pass must be per batch element); the C++
    runtime broadcasts the stored scalar to the runtime batch before calling it.
    """
    from ..es import IFT, RHS, AssembledVector, NewtonStep, ParamIFT
    from ..models.common import ComposedModel
    from ..models.export import compile_model

    system = inner.system
    solver = inner.solver

    promoted_qnames = promoted_qnames or set()
    promoted_snapshots = promoted_snapshots or {}
    selected_param_pairs = selected_param_pairs or set()
    # Promoted parameters that live inside this segment's residual model, in
    # model.input_spec order (the canonical graph-call tail order). After the
    # _promote_parameters + _rebuild_after_promotion pass these appear
    # in system.model.input_spec but are neither unknowns nor givens.
    seg_param_inputs = [
        n
        for n in system.model.input_spec
        if n in promoted_qnames and n not in system.unknown_names and n not in system.given_names
    ]
    param_tail = tuple(seg_param_inputs)
    emit_pift = bool(selected_param_pairs)

    rhs = RHS(system, param_names=param_tail).to(device)
    step = NewtonStep(system, solver.linear_solver, param_names=param_tail).to(device)
    ift = IFT(
        system, solver.linear_solver, selected_pairs=selected_ift_pairs, param_names=param_tail
    ).to(device)
    pift = ParamIFT(
        system, solver.linear_solver, param_tail, selected_pairs=selected_param_pairs
    ).to(device)

    # Compile-time guard: the per-pair IFT consumer only supports plain-batch
    # (DENSE) variable pairs. A requested pair touching a sub-batched (per-grain)
    # unknown or given -- e.g. crystal plasticity's per-grain state -- is not yet
    # supported; fail fast here with a clear message instead of at runtime. A
    # global-output / global-input pair of the same model compiles fine.
    if emit_ift:
        for u, g in ift.emitted_pairs():
            u_sub = tuple(system.ulayout.sub_batch_shape(u))
            g_sub = tuple(system.glayout.sub_batch_shape(g))
            if u_sub or g_sub:
                bad = f"{u!r} (sub_batch {u_sub})" if u_sub else f"{g!r} (sub_batch {g_sub})"
                raise NotImplementedError(
                    f"neml2-compile: derivative ({u}, {g}) of implicit model involves the "
                    f"sub-batched variable {bad}; compiled Jacobian/JVP for sub-batched "
                    f"(per-grain) implicit variables is not yet supported. Request only "
                    f"non-sub-batched output:input pairs (e.g. a global-to-global "
                    f"derivative), or use eager mode for the per-grain sensitivities."
                )

    # Build per-variable example inputs at the user-declared natural
    # shapes first, then pack them into per-group tensors via
    # AssembledVector.from_dict -- the same path the segment forwards
    # use at runtime, so the export-time tracing shape matches the
    # C++ runtime input shape exactly.
    dyn_shape = _shared_dyn_shape(shapes or {}, system.given_names + system.unknown_names)
    shapes = shapes or {}

    def _example_for_unknown(name: str) -> torch.Tensor:
        history_name = f"{name}~1"
        dyn, sub = shapes.get(history_name, shapes.get(name, (dyn_shape, ())))
        type_cls = system.model.input_spec[name]
        return torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)

    def _example_for_given(name: str) -> torch.Tensor:
        bare = name.split("~", 1)[0]
        dyn, sub = shapes.get(name, shapes.get(bare, (dyn_shape, ())))
        type_cls = system.model.input_spec[name]
        return torch.zeros(*dyn, *sub, *type_cls.BASE_SHAPE, dtype=torch.float64, device=device)

    def _per_group_examples(layout, example_for_var) -> list[torch.Tensor]:
        typed_values: dict = {}
        for group in layout.groups:
            for name in group:
                raw = example_for_var(name)
                type_cls = layout.specs[name]
                sb = layout.sub_batch_shape(name)
                typed_values[name] = type_cls(raw, sub_batch_ndim=len(sb))
        vec = AssembledVector.from_dict(layout, typed_values)
        return [t.data for t in vec.tensors]  # data-ok AOTI

    u_group_examples = _per_group_examples(system.ulayout, _example_for_unknown)
    g_group_examples = _per_group_examples(system.glayout, _example_for_given)
    # Promoted-parameter tail. rhs/step/ift take the stored SCALAR (natural
    # shape -- typically ()); ParamIFT takes the parameter PER-BATCH
    # ((*dyn, *param_base)) so its reverse pass is per batch element (the same
    # scalar-vs-per-batch split the forward value vs param-Jacobian graphs use).
    scalar_param_examples = [
        promoted_snapshots[n].to(device=device, dtype=torch.float64) for n in seg_param_inputs
    ]

    def _per_batch_param(name: str) -> torch.Tensor:
        snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
        pbase = tuple(snap.shape)
        return snap.reshape((1,) * len(dyn_shape) + pbase).expand(*dyn_shape, *pbase).contiguous()

    per_batch_param_examples = [_per_batch_param(n) for n in seg_param_inputs]

    base_examples = tuple(u_group_examples) + tuple(g_group_examples)
    example_inputs = base_examples + tuple(scalar_param_examples)
    pift_example_inputs = base_examples + tuple(per_batch_param_examples)

    rhs_name = f"{pkg_basename}_rhs.pt2"
    step_name = f"{pkg_basename}_step.pt2"
    ift_name = f"{pkg_basename}_ift.pt2"
    pift_name = f"{pkg_basename}_pift.pt2"

    dynamic_dim = 0 if dynamic_batch else None
    # A request_AD leaf inside the residual makes the residual Jacobian (``A=∂r/∂u``
    # in ``step``, ``∂r/∂g`` in ``ift``) carry an embedded reverse-mode
    # ``torch.autograd.grad``, which lowers only under ``trace_autograd_ops`` +
    # ``strict``. The equation-system assembly that both graphs use was made
    # strict-export-friendly (no in-``forward`` generators -- the static layout math
    # is precomputed), so the SAME graphs lower with the strict helper. ``rhs`` is
    # the residual VALUE (no autograd), always plain compile.
    uses_ad = _model_uses_request_ad(system.model)
    _compile_jac = _compile_param_derivative_graph if uses_ad else compile_model
    compile_model(rhs, example_inputs, output_dir / rhs_name, dynamic_batch_dim=dynamic_dim)
    _report(progress_cb, rhs_name)
    _compile_jac(step, example_inputs, output_dir / step_name, dynamic_batch_dim=dynamic_dim)
    _report(progress_cb, step_name)
    # Per-(unknown, given) pair metadata for the IFT graph. The IFT emits one
    # block per variable pair (via AssembledMatrix.disassemble), in
    # ift.emitted_pairs() order, so the C++ runtime composes them against
    # dg_dmaster exactly like a forward segment's per-pair Jacobian blocks. An
    # IFT block is always batch-dependent (it is a function of the converged
    # solution + the residual Jacobians), so ``batch_independent`` is False -- no
    # eager probe is run (the IFT solve would be singular at the zero example
    # point anyway). The sub-batch (BLOCK) guard above guarantees every emitted
    # pair is plain-batch, so ``in_sub_batch_shape`` is empty.
    ift_jacobian_pairs: list[dict] | None = None
    if emit_ift:
        ift_jacobian_pairs = [
            {
                "out_var": u,
                "in_var": g,
                "out_base_shape": [int(s) for s in system.ulayout.specs[u].BASE_SHAPE],
                "in_base_shape": [int(s) for s in system.glayout.specs[g].BASE_SHAPE],
                "in_sub_batch_shape": [],
                "batch_independent": False,
            }
            for (u, g) in ift.emitted_pairs()
        ]
        _compile_jac(ift, example_inputs, output_dir / ift_name, dynamic_batch_dim=dynamic_dim)
        _report(progress_cb, ift_name)

    # Per-(unknown, param) parameter-Jacobian graph (ParamIFT). Emits one dense
    # ``du/dθ`` block per requested ``(unknown, param)`` pair, in
    # pift.emitted_param_pairs() order (unknowns outer, params inner). The
    # promoted parameter enters PER-BATCH (pift_example_inputs); the C++ runtime
    # broadcasts the stored scalar to the runtime batch before the call. Compiled
    # with the same strict + trace_autograd_ops recipe the forward parameter
    # graphs use (reverse-mode autograd is the only AD that lowers).
    param_jacobian_pairs: list[dict] | None = None
    if emit_pift:
        param_jacobian_pairs = [
            {
                "out_var": u,
                "param": p,
                "out_base_shape": [int(s) for s in system.ulayout.specs[u].BASE_SHAPE],
                "param_base_shape": [int(s) for s in promoted_snapshots[p].shape],
            }
            for (u, p) in pift.emitted_param_pairs()
        ]
        _compile_param_derivative_graph(
            pift, pift_example_inputs, output_dir / pift_name, dynamic_batch_dim=dynamic_dim
        )
        _report(progress_cb, pift_name)

    # Per-variable infos retained at the segment level as the source of
    # truth for the C++ runtime's per-variable state map (predictor
    # routing, downstream forward composition). Per-group / per-cell
    # metadata sits alongside via _enumerate_group_infos.
    def _var_info(layout, name: str) -> dict:
        type_cls = layout.type_of(name)
        sb = tuple(int(s) for s in layout.sub_batch_shape(name))
        base = tuple(int(s) for s in type_cls.BASE_SHAPE)
        sb_total = 1
        for s in sb:
            sb_total *= s
        return {
            "name": name,
            "var_size": sb_total * _var_size(type_cls),
            "sub_batch_shape": list(sb),
            "base_shape": list(base),
        }

    unknown_infos = [_var_info(system.ulayout, n) for n in system.unknown_names]
    given_infos = [_var_info(system.glayout, n) for n in system.given_names]
    residual_infos = [_var_info(system.blayout, n) for n in system.residual_names]

    group_meta = _enumerate_group_infos(system)

    # NB: solver convergence / line-search configuration (atol / rtol / miters /
    # linesearch) is NOT baked here (schema v4). The generated stub `.i` carries
    # a [Solvers] block and the AOTIModel shim forwards it to the C++ runtime at
    # load time. Only the linear solver is baked -- it lives inside the compiled
    # step/ift graphs above.
    seg: dict = {
        "rhs_package": rhs_name,
        "step_package": step_name,
        "unknowns": unknown_infos,
        "givens": given_infos,
        "residuals": residual_infos,
        # Per-group metadata drives the per-group pack/unpack at solve
        # start/end and the per-group Newton loop arithmetic (rhs/step).
        "unknown_group_infos": group_meta["unknown_group_infos"],
        "given_group_infos": group_meta["given_group_infos"],
        "residual_group_infos": group_meta["residual_group_infos"],
        # Promoted parameters threaded into this segment's graphs as the trailing
        # positional tail (graph-call order). Empty in the common fully-baked
        # case; non-empty (schema v7) when a parameter inside the residual was
        # promoted. The C++ runtime passes the stored scalars to rhs/step/ift and
        # broadcasts them per-batch for the param-IFT graph.
        "param_inputs": list(seg_param_inputs),
    }
    if emit_ift:
        seg["ift_package"] = ift_name
        # Per-(unknown, given) pair metadata for the IFT loader's output tuple,
        # in the same order the IFT graph emits blocks. Consumed via the
        # per-pair Jacobian composition (same path as a forward segment).
        seg["jacobian_pairs"] = ift_jacobian_pairs
    if emit_pift:
        seg["param_ift_package"] = pift_name
        # Per-(unknown, param) pair metadata for the ParamIFT loader's output
        # tuple, in pift.emitted_param_pairs() order (unknowns outer, params
        # inner). Consumed by the C++ implicit parameter-Jacobian path.
        seg["param_jacobian_pairs"] = param_jacobian_pairs

    # Predictor: compile as an extra graph if present. Promoted tail not
    # threaded here either (same reason as the rhs/step/ift block above; the
    # predictor is part of the implicit segment for promotion purposes).
    if inner.predictor is not None:
        pred = inner.predictor.to(device)
        pred_exportable = ComposedModel([pred])
        pred_inputs = _example_inputs_for(pred_exportable, device, shapes=shapes)
        pred_name = f"{pkg_basename}_predictor.pt2"
        compile_model(
            pred_exportable, pred_inputs, output_dir / pred_name, dynamic_batch_dim=dynamic_dim
        )
        _report(progress_cb, pred_name)
        seg["predictor_package"] = pred_name
        seg["predictor_inputs"] = _segment_var_infos(pred.input_spec)
        seg["predictor_outputs"] = _segment_var_infos(pred.output_spec)

    return seg


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


@dataclass
class _Reachability:
    """Per-segment dependency reachability for the derivative-graph prune.

    ``reach[V]`` = master (structural) inputs that can flow into V; ``canreach
    [V]`` = master outputs V can reach; ``param_reach[V]`` = requested promoted
    params whose sensitivity reaches V. Each segment is modelled as "every
    output depends on every input" (the all-pairs internal Jacobian; an implicit
    segment's IFT couples every given to every unknown) -- the correct
    conservative data-flow model. The ``keep_*`` predicates decide, per local
    ``(out, in)`` / ``(out, param)`` pair, whether that block must be compiled.
    Built once by :func:`_segment_reachability` and shared by both the planner
    (artifact prediction) and the executor (which pairs to emit) so the two can
    never drift.
    """

    segments: list
    seg_input_sets: list[set[str]]
    seg_output_sets: list[set[str]]
    downstream_demands: list[set[str]]
    seg_direct_params: list[set[str]]
    master_outs: set[str]
    reach: dict[str, set[str]]
    canreach: dict[str, set[str]]
    param_reach: dict[str, set[str]]
    master_pairs: set[tuple[str, str]]
    param_pairs: set[tuple[str, str]]

    def keep_local_pair(self, o: str, i: str) -> bool:
        """A local pair ``(o, i)`` carries a requested master sensitivity iff its
        column is reachable from a requested input AND its row reaches a
        requested output."""
        return any(
            (i_m in self.reach.get(i, ())) and (o_m in self.canreach.get(o, ()))
            for (o_m, i_m) in self.master_pairs
        )

    def keep_param_prop(self, o: str, i: str) -> bool:
        """A jvp / IFT local pair ``(o, i)`` must be compiled when its column
        ``i`` carries a requested parameter sensitivity needed at a requested
        output reachable from ``o``."""
        return any(
            (p in self.param_reach.get(i, ())) and (o_m in self.canreach.get(o, ()))
            for (o_m, p) in self.param_pairs
        )

    def keep_param_pair(self, o: str, p: str) -> bool:
        """A direct param-graph pair ``(o, p)``: keep when ``o`` reaches a
        requested master output paired with parameter ``p``."""
        return any(o_m in self.canreach.get(o, ()) for (o_m, pp) in self.param_pairs if pp == p)


def _segment_reachability(
    model,
    segments,
    promoted_qnames: set[str],
    master_pairs: set[tuple[str, str]],
    param_pairs: set[tuple[str, str]],
) -> _Reachability:
    """Compute the :class:`_Reachability` bundle for a partitioned composed model.

    Factored out of the old ``_export_composed`` so the planner and executor
    share one copy of the reachability math (single source of truth -- no drift
    between predicted and produced derivative graphs).
    """
    requested_params = {p for (_, p) in param_pairs}

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
    seg_input_sets = [_seg_inputs(payload) for _, payload in segments]
    downstream_demands: list[set[str]] = [set() for _ in segments]
    for j, (_, payload_j) in enumerate(segments):
        inputs_j = _seg_inputs(payload_j)
        for i in range(j):
            downstream_demands[i] |= inputs_j
    master_outs = set(model.output_spec)

    structural_master_in = set(_structural_inputs(model.input_spec, promoted_qnames))
    reach: dict[str, set[str]] = {i: {i} for i in structural_master_in}
    for sin, sout in zip(seg_input_sets, seg_output_sets, strict=True):
        flowed: set[str] = set()
        for x in sin:
            flowed |= reach.get(x, set())
        for o in sout:
            reach[o] = reach.get(o, set()) | flowed
    canreach: dict[str, set[str]] = {o: {o} for o in master_outs}
    for sin, sout in zip(reversed(seg_input_sets), reversed(seg_output_sets), strict=True):
        reachable: set[str] = set()
        for o in sout:
            reachable |= canreach.get(o, set())
        for x in sin:
            canreach[x] = canreach.get(x, set()) | reachable

    # Which requested promoted parameters live directly in each segment (after
    # promotion + _rebuild_after_promotion, the qname appears in that segment's
    # model.input_spec).
    def _seg_direct_params(payload: object) -> set[str]:
        if isinstance(payload, list):
            return {p for p in requested_params for leaf in payload if p in leaf.input_spec}
        return {p for p in requested_params if p in payload.system.model.input_spec}  # type: ignore[attr-defined]

    seg_direct_params = [_seg_direct_params(payload) for _, payload in segments]

    # ``param_reach[V]`` = requested promoted params whose sensitivity reaches V.
    # Forward pass mirroring ``reach``: a segment's outputs inherit the union of
    # its inputs' param_reach (indirect) plus the params promoted directly in
    # that segment (direct, conservative all-outputs model).
    param_reach: dict[str, set[str]] = {}
    for sin, sout, direct in zip(seg_input_sets, seg_output_sets, seg_direct_params, strict=True):
        flowed = set(direct)
        for x in sin:
            flowed |= param_reach.get(x, set())
        for o in sout:
            param_reach[o] = param_reach.get(o, set()) | flowed

    return _Reachability(
        segments=segments,
        seg_input_sets=seg_input_sets,
        seg_output_sets=seg_output_sets,
        downstream_demands=downstream_demands,
        seg_direct_params=seg_direct_params,
        master_outs=master_outs,
        reach=reach,
        canreach=canreach,
        param_reach=param_reach,
        master_pairs=master_pairs,
        param_pairs=param_pairs,
    )


# ---------------------------------------------------------------------------
# Model preparation (the shared setup prelude)
# ---------------------------------------------------------------------------


@dataclass
class _PreparedExport:
    """The result of :func:`_prepare_export` -- a model made ready to compile,
    plus every resolved option a segment compile / plan needs.

    Deterministic in ``(input file, options, device)``: the same inputs always
    yield the same bundle, which is what lets a spawned worker re-derive its
    assigned segment from scratch (:func:`_compile_segment_worker`) instead of
    receiving an unpicklable live ``nn.Module`` from the parent.
    """

    model: Model
    promoted_qnames: set[str]
    promoted_snapshots: dict[str, torch.Tensor]
    origin: dict[str, str]
    promoted_base_shapes: dict[str, tuple[int, ...]]
    resolved_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    dynamic_batch: bool
    master_pairs: set[tuple[str, str]]
    param_pairs: set[tuple[str, str]]
    boundary_aliases: dict[str, dict[str, str]]
    structural_input_names: list[str]


def _prepare_export(
    hit_path: str | Path,
    model_name: str,
    *,
    device: str = "cpu",
    promoted: set[str] | list[str] | tuple[str, ...] = (),
    example_batch_shape: dict[str, str] | str | None = None,
    dynamic_batch: bool | None = None,
    derivatives: Sequence[str] = (),
    renames: dict[str, dict[str, str]] | None = None,
    pre: Sequence[str] = (),
    additional_args: tuple[str, ...] = (),
) -> _PreparedExport:
    """Load a model and run the full pre-compile prelude, returning a
    :class:`_PreparedExport`.

    This is the setup shared by every export path: it loads the model, wraps a
    bare forward leaf in a ``ComposedModel`` (so promoted names live in the eager
    namespace), resolves example shapes, promotes parameters + rebuilds the
    spec-caching containers, resolves ``-d`` derivative specs, validates boundary
    renames, freezes leftover parameters to buffers, checks baked-shape
    conflicts, and seeds every implicit sub-batch layout. No ``.pt2`` is
    produced -- the expensive ``compile_model`` calls happen later, in
    :func:`_execute_segment_plan`.

    ``device`` is threaded through because :func:`_seed_implicit_subbatch` builds
    device-resident stand-in tensors; the bundle is therefore device-specific and
    must be rebuilt per device (never cached across devices).
    """
    from ..factory import load_input
    from ..models.common import ComposedModel

    factory = load_input(hit_path, pre=pre, additional_args=additional_args)
    model = factory.get_model(model_name)

    # Wrap a bare (non-composed) forward leaf up-front -- BEFORE promotion -- so
    # promoted parameter names live in the SAME namespace the eager runtime
    # reports. ImplicitUpdate / composed-with-implicit models keep their dedicated
    # export path + namespace; an already-composed model is left untouched.
    if not isinstance(model, ComposedModel) and not _contains_implicit(model):
        model = ComposedModel([model])

    # Resolve example-batch-shape declarations: kwarg wins, then HIT [Settings],
    # then the (2,)/uniform default.
    settings_shapes, settings_dyn = _read_settings(factory)
    if isinstance(example_batch_shape, str):
        declared = {"*": example_batch_shape}
    elif isinstance(example_batch_shape, dict):
        declared = dict(example_batch_shape)
    else:
        declared = settings_shapes
    if dynamic_batch is None:
        dynamic_batch = settings_dyn
    resolved_shapes = _resolve_example_shapes(model.input_spec, declared)

    # Validate / resolve promoted names against the live model BEFORE any
    # ComposedModel wrapping. Snapshot the initial values now too.
    promoted_set = set(promoted)
    promoted_names, origin = _validate_promoted(model, promoted_set)
    promoted_snapshots = _snapshot_promoted(model, promoted_names)

    # Route each promoted parameter through the leaf's PromotedParam machinery.
    promo_info = _promote_parameters(model, promoted_names)
    promoted_qnames = set(promoted_names)
    promoted_base_shapes = {q: tuple(cls.BASE_SHAPE) for q, (cls, _h, _l) in promo_info.items()}

    # Rebuild every spec-caching container so promoted parameters route correctly.
    if promoted_names:
        model = _rebuild_after_promotion(model)

    # Resolve -d/--derivative specs into the master (out, in) / (out, param) pairs.
    structural_input_names = list(_structural_inputs(model.input_spec, promoted_qnames))
    master_pairs, param_pairs = _resolve_derivative_specs(
        derivatives,
        list(model.output_spec),
        structural_input_names,
        param_names=sorted(promoted_qnames),
    )

    # Validate + normalize boundary renames against the post-promotion namespaces.
    boundary_aliases = _validate_renames(
        renames,
        input_names=structural_input_names,
        output_names=list(model.output_spec),
        param_names=sorted(promoted_qnames),
    )

    # Freeze any remaining nn.Parameter to a persistent buffer so torch.export
    # bakes it into the graph instead of lifting it as a graph input.
    _freeze_remaining_parameters_to_buffers(model)

    # Catch baked-parameter shape conflicts before torch.export does.
    if dynamic_batch:
        _validate_baked_against_shapes(model, resolved_shapes, promoted_qnames)

    # Populate per-variable sub_batch_shape on every inner ModelNonlinearSystem.
    _seed_implicit_subbatch(model, resolved_shapes, device)

    return _PreparedExport(
        model=model,
        promoted_qnames=promoted_qnames,
        promoted_snapshots=promoted_snapshots,
        origin=origin,
        promoted_base_shapes=promoted_base_shapes,
        resolved_shapes=resolved_shapes,
        dynamic_batch=dynamic_batch,
        master_pairs=master_pairs,
        param_pairs=param_pairs,
        boundary_aliases=boundary_aliases,
        structural_input_names=structural_input_names,
    )


# ---------------------------------------------------------------------------
# Segment planning + execution
# ---------------------------------------------------------------------------


@dataclass
class _SegmentPlan:
    """One compilable segment: what it is, what it will emit, and how to build it.

    ``predicted_artifacts`` (ordered ``.pt2`` filenames) is picklable and is the
    only field a spawned worker needs from the parent (for the progress
    denominator). The execution fields (``seg_model`` / ``impl_model`` /
    ``selected_*``) hold live objects used only in-process by
    :func:`_execute_segment_plan`; a worker rebuilds them by re-running the
    planner on its own re-derived model.
    """

    index: int
    kind: str  # "forward" | "implicit"
    basename: str
    predicted_artifacts: list[str]
    single_forward_pvjp: bool = False
    seg_model: object = None
    impl_model: object = None
    selected_pairs: set[tuple[str, str]] | None = None
    selected_param_pairs: set[tuple[str, str]] | None = None
    selected_ift_pairs: set[tuple[str, str]] | None = None


def _predict_forward_artifacts(
    basename: str,
    selected_pairs: set[tuple[str, str]] | None,
    selected_param_pairs: set[tuple[str, str]] | None,
    *,
    single_forward_pvjp: bool,
) -> list[str]:
    """Ordered ``.pt2`` names a forward segment emits -- mirrors, in emission
    order, the ``compile_model`` calls in :func:`_compile_forward_segment` (+ the
    param-Jacobian / VJP helpers). The pvjp graph is emitted ONLY for the lone
    forward segment of a forward-only model (``single_forward_pvjp``)."""
    arts = [f"{basename}.pt2"]
    # jvp: emitted when pairs are None (all) or a non-empty selection.
    if selected_pairs is None or len(selected_pairs) > 0:
        arts.append(f"{basename}_jvp.pt2")
    if selected_param_pairs:
        arts.append(f"{basename}_pjac.pt2")
        if single_forward_pvjp:
            arts.append(f"{basename}_pvjp.pt2")
    return arts


def _predict_implicit_artifacts(
    basename: str,
    *,
    emit_ift: bool,
    emit_pift: bool,
    has_predictor: bool,
) -> list[str]:
    """Ordered ``.pt2`` names an implicit segment emits -- mirrors, in emission
    order, the ``compile_model`` calls in :func:`_compile_implicit_segment`."""
    arts = [f"{basename}_rhs.pt2", f"{basename}_step.pt2"]
    if emit_ift:
        arts.append(f"{basename}_ift.pt2")
    if emit_pift:
        arts.append(f"{basename}_pift.pt2")
    if has_predictor:
        arts.append(f"{basename}_predictor.pt2")
    return arts


def _plan_segments(prepared: _PreparedExport, model_name: str) -> list[_SegmentPlan]:
    """Partition *prepared*'s model into an ordered list of :class:`_SegmentPlan`.

    Unifies all three model shapes -- forward-only (one forward segment named
    ``model_name``), a single ``ImplicitUpdate`` (one implicit segment named
    ``model_name``), and a ``ComposedModel`` containing implicit children (one
    segment per partition, named ``model_name_seg{i}``). The derivative-graph
    selection uses the shared :func:`_segment_reachability`, so the predicted
    artifacts exactly match what :func:`_execute_segment_plan` will produce.
    """
    from ..models.common import ComposedModel, ImplicitUpdate

    model = prepared.model
    promoted_qnames = prepared.promoted_qnames
    master_pairs = prepared.master_pairs
    param_pairs = prepared.param_pairs
    inner = model

    # Single ImplicitUpdate: one implicit segment. For a single ImplicitUpdate
    # the outputs are the unknowns, so master pairs map to local pairs directly.
    if isinstance(inner, ImplicitUpdate):
        isys = inner.system
        selected_ift_pairs = {
            (o, i) for (o, i) in master_pairs if o in isys.unknown_names and i in isys.given_names
        }
        selected_param_pairs = {
            (o, p) for (o, p) in param_pairs if o in isys.unknown_names and p in promoted_qnames
        }
        arts = _predict_implicit_artifacts(
            model_name,
            emit_ift=bool(selected_ift_pairs),
            emit_pift=bool(selected_param_pairs),
            has_predictor=inner.predictor is not None,
        )
        return [
            _SegmentPlan(
                index=0,
                kind="implicit",
                basename=model_name,
                predicted_artifacts=arts,
                impl_model=inner,
                selected_ift_pairs=selected_ift_pairs,
                selected_param_pairs=selected_param_pairs,
            )
        ]

    # ComposedModel containing ImplicitUpdate children: one segment per partition.
    if _contains_implicit(inner):
        segments = _partition_into_segments(model)
        reach = _segment_reachability(model, segments, promoted_qnames, master_pairs, param_pairs)
        plans: list[_SegmentPlan] = []
        for i, (kind, payload) in enumerate(segments):
            basename = f"{model_name}_seg{i}"
            if kind == "forward":
                # Wrap the segment's leaves in a fresh ComposedModel so the
                # dependency resolver derives its specs from the leaves' own
                # specs. (.to(device) is deferred to _compile_forward_segment.)
                assert isinstance(payload, list)
                needed = (reach.master_outs | reach.downstream_demands[i]) & reach.seg_output_sets[
                    i
                ]
                extra = sorted(needed)
                seg_model = ComposedModel(payload, additional_outputs=extra)
                seg_struct_in = _structural_inputs(seg_model.input_spec, promoted_qnames)
                seg_selected = {
                    (o, si)
                    for o in seg_model.output_spec
                    for si in seg_struct_in
                    if reach.keep_local_pair(o, si) or reach.keep_param_prop(o, si)
                }
                seg_param_pairs = {
                    (o, p)
                    for o in seg_model.output_spec
                    for p in reach.seg_direct_params[i]
                    if reach.keep_param_pair(o, p)
                }
                arts = _predict_forward_artifacts(
                    basename, seg_selected, seg_param_pairs, single_forward_pvjp=False
                )
                plans.append(
                    _SegmentPlan(
                        index=i,
                        kind="forward",
                        basename=basename,
                        predicted_artifacts=arts,
                        seg_model=seg_model,
                        selected_pairs=seg_selected,
                        selected_param_pairs=seg_param_pairs,
                    )
                )
            else:
                impl_model = payload
                assert isinstance(impl_model, ImplicitUpdate)
                isys = impl_model.system
                selected_ift_pairs = {
                    (u, g)
                    for u in isys.unknown_names
                    for g in isys.given_names
                    if reach.keep_local_pair(u, g) or reach.keep_param_prop(u, g)
                }
                selected_param_pairs = {
                    (u, p)
                    for u in isys.unknown_names
                    for p in reach.seg_direct_params[i]
                    if reach.keep_param_pair(u, p)
                }
                arts = _predict_implicit_artifacts(
                    basename,
                    emit_ift=bool(selected_ift_pairs),
                    emit_pift=bool(selected_param_pairs),
                    has_predictor=impl_model.predictor is not None,
                )
                plans.append(
                    _SegmentPlan(
                        index=i,
                        kind="implicit",
                        basename=basename,
                        predicted_artifacts=arts,
                        impl_model=impl_model,
                        selected_ift_pairs=selected_ift_pairs,
                        selected_param_pairs=selected_param_pairs,
                    )
                )
        return plans

    # Forward-only: a single forward segment. For one forward segment the master
    # pairs ARE the segment-local pairs, so they pass straight through.
    arts = _predict_forward_artifacts(
        model_name, master_pairs, param_pairs, single_forward_pvjp=True
    )
    return [
        _SegmentPlan(
            index=0,
            kind="forward",
            basename=model_name,
            predicted_artifacts=arts,
            seg_model=model,
            selected_pairs=master_pairs,
            selected_param_pairs=param_pairs,
            single_forward_pvjp=True,
        )
    ]


def _execute_segment_plan(
    plan: _SegmentPlan,
    prepared: _PreparedExport,
    output_dir: Path,
    device: str,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Compile one segment's ``.pt2`` graphs and return its metadata dict.

    The single execution path shared by the serial loop and the process worker.
    Mirrors the per-segment bodies of the former ``_export_forward`` /
    ``_export_implicit`` / ``_export_composed``.
    """
    promoted_qnames = prepared.promoted_qnames
    promoted_snapshots = prepared.promoted_snapshots
    shapes = prepared.resolved_shapes
    dynamic_batch = prepared.dynamic_batch

    if plan.kind == "forward":
        (
            pkg_name,
            in_infos,
            out_infos,
            jvp_pkg_name,
            param_inputs,
            jacobian_pairs,
        ) = _compile_forward_segment(
            plan.seg_model,
            plan.basename,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            promoted_snapshots=promoted_snapshots,
            shapes=shapes,
            dynamic_batch=dynamic_batch,
            selected_pairs=plan.selected_pairs,
            progress_cb=progress_cb,
        )
        seg_entry: dict = {
            "kind": "forward",
            "package": pkg_name,
            "inputs": _segment_var_infos(in_infos),
            "outputs": _segment_var_infos(out_infos),
            "param_inputs": param_inputs,
        }
        if jvp_pkg_name is not None:
            seg_entry["jvp_package"] = jvp_pkg_name
            seg_entry["jacobian_pairs"] = jacobian_pairs
        if plan.selected_param_pairs:
            pjac_pkg, pjac_pairs = _compile_param_jacobian(
                plan.seg_model,
                plan.basename,
                output_dir,
                device,
                promoted_qnames=promoted_qnames,
                promoted_snapshots=promoted_snapshots,
                selected_param_pairs=plan.selected_param_pairs,
                shapes=shapes,
                dynamic_batch=dynamic_batch,
                progress_cb=progress_cb,
            )
            seg_entry["param_jacobian_package"] = pjac_pkg
            seg_entry["param_jacobian_pairs"] = pjac_pairs
            # The single-forward path also emits the adjoint (VJP) graph -- the
            # cheaper form for many-parameter optimization. Composed forward
            # segments do NOT (only the direct param-Jacobian propagates).
            if plan.single_forward_pvjp:
                pvjp_pkg, pvjp_params, pvjp_outputs = _compile_param_vjp(
                    plan.seg_model,
                    plan.basename,
                    output_dir,
                    device,
                    promoted_qnames=promoted_qnames,
                    promoted_snapshots=promoted_snapshots,
                    selected_params={p for (_, p) in plan.selected_param_pairs},
                    shapes=shapes,
                    dynamic_batch=dynamic_batch,
                    progress_cb=progress_cb,
                )
                seg_entry["param_vjp_package"] = pvjp_pkg
                seg_entry["param_vjp_params"] = pvjp_params
                seg_entry["param_vjp_outputs"] = pvjp_outputs
        return seg_entry

    # implicit
    seg = _compile_implicit_segment(
        plan.impl_model,
        plan.basename,
        output_dir,
        device,
        shapes=shapes,
        dynamic_batch=dynamic_batch,
        emit_ift=bool(plan.selected_ift_pairs),
        selected_ift_pairs=plan.selected_ift_pairs,
        promoted_qnames=promoted_qnames,
        promoted_snapshots=promoted_snapshots,
        selected_param_pairs=plan.selected_param_pairs,
        progress_cb=progress_cb,
    )
    return {"kind": "implicit", **seg}


@dataclass
class _ExportPlan:
    """What a compile of ``(hit_path, model_name, options)`` on one device will
    generate, without compiling.

    ``artifacts`` is the flat, ordered list of every generated filename for the
    device -- the ``.pt2`` graphs of each segment followed by the
    ``<model>_meta.json``. ``.total`` is the count the CLI uses for the ``[k/N]``
    progress denominator (the standalone ``<model>_aoti.i`` stub is a CLI-level
    concern added on top).
    """

    artifacts: list[str]
    segments: list[_SegmentPlan]

    @property
    def total(self) -> int:
        return len(self.artifacts)


def plan_export_artifacts(
    hit_path: str | Path,
    model_name: str,
    *,
    device: str = "cpu",
    promoted: set[str] | list[str] | tuple[str, ...] = (),
    example_batch_shape: dict[str, str] | str | None = None,
    dynamic_batch: bool | None = None,
    derivatives: Sequence[str] = (),
    renames: dict[str, dict[str, str]] | None = None,
    pre: Sequence[str] = (),
    additional_args: tuple[str, ...] = (),
) -> _ExportPlan:
    """Enumerate the files a compile will generate, WITHOUT compiling.

    Runs the same prelude + segment planning :func:`export_model_for_aoti` uses,
    then returns the ordered list of every generated filename for *device* (each
    segment's ``.pt2`` graphs, then ``<model_name>_meta.json``). No
    ``compile_model`` is called, so this is cheap (one model load + structural
    analysis). Primarily drives the ``[k/N]`` progress denominator; also useful
    for programmatic inspection of a model's segment structure. The artifact set
    is device-independent, so a single call (e.g. ``device="cpu"``) suffices to
    size a multi-device compile.
    """
    prepared = _prepare_export(
        hit_path,
        model_name,
        device=device,
        promoted=promoted,
        example_batch_shape=example_batch_shape,
        dynamic_batch=dynamic_batch,
        derivatives=derivatives,
        renames=renames,
        pre=pre,
        additional_args=additional_args,
    )
    plans = _plan_segments(prepared, model_name)
    artifacts: list[str] = []
    for plan in plans:
        artifacts.extend(plan.predicted_artifacts)
    artifacts.append(f"{model_name}_meta.json")
    return _ExportPlan(artifacts=artifacts, segments=plans)


# ---------------------------------------------------------------------------
# Parallel segment compilation (process pool)
# ---------------------------------------------------------------------------

#: Sentinel enqueued by the parent to tell the progress-drainer thread to exit.
_PROGRESS_SENTINEL = "__neml2_progress_done__"

#: Per-worker handle to the shared progress queue (set by :func:`_worker_init`).
#: A module global because ``ProcessPoolExecutor`` cannot pass a Manager queue as
#: a per-task argument -- it must arrive via the pool ``initializer``.
_worker_progress_queue = None


def _worker_init(queue) -> None:
    """Pool ``initializer``: stash the shared progress queue in the worker."""
    global _worker_progress_queue
    _worker_progress_queue = queue


def _compile_segment_worker(args) -> tuple[int, dict]:
    """Worker entry: re-derive the model from the input file and compile ONE
    segment. Everything in *args* is picklable (paths, strings, an options dict,
    and the segment index); no live ``nn.Module`` crosses the process boundary.
    Per-``.pt2`` progress is streamed back to the parent through the shared queue.
    """
    hit_path, model_name, output_dir, device, export_opts, seg_index = args
    prepared = _prepare_export(hit_path, model_name, device=device, **export_opts)
    plans = _plan_segments(prepared, model_name)
    plan = plans[seg_index]
    queue = _worker_progress_queue
    cb = (lambda name: queue.put(name)) if queue is not None else None
    seg_meta = _execute_segment_plan(plan, prepared, Path(output_dir), device, progress_cb=cb)
    return seg_index, seg_meta


def _compile_segments_parallel(
    hit_path: str | Path,
    model_name: str,
    output_dir: Path,
    device: str,
    export_opts: dict,
    plans: list[_SegmentPlan],
    jobs: int,
    progress_cb: Callable[[str], None] | None,
) -> list[dict]:
    """Compile independent segments concurrently in a spawn process pool.

    Each segment is compiled by a worker that re-derives it from *hit_path* +
    *export_opts* (workers can't receive live modules). Segment metadata is
    reassembled in SEGMENT ORDER regardless of completion order, so the resulting
    ``_meta.json`` is identical to a serial (``jobs=1``) run. Per-``.pt2``
    progress from workers is forwarded to *progress_cb* by a drainer thread.

    The worker pool is sized at ``min(jobs, len(plans))`` -- there is one task per
    segment, so requesting more processes than segments would only spawn idle
    interpreters. Granularity is per-segment: a worker compiles all of one
    segment's ``.pt2`` graphs sequentially.
    """
    import concurrent.futures as cf
    import multiprocessing as mp
    import threading

    ctx = mp.get_context("spawn")
    # One task per segment -> never spawn more workers than segments.
    workers = max(1, min(jobs, len(plans)))

    manager = None
    queue = None
    drainer = None
    initializer = None
    initargs: tuple = ()
    if progress_cb is not None:
        manager = ctx.Manager()
        queue = manager.Queue()
        initializer = _worker_init
        initargs = (queue,)

        def _drain() -> None:
            while True:
                item = queue.get()
                if item == _PROGRESS_SENTINEL:
                    break
                progress_cb(item)

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

    n = len(plans)
    results: dict[int, dict] = {}
    hit_s = str(hit_path)
    out_s = str(output_dir)
    try:
        with cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=initializer, initargs=initargs
        ) as ex:
            futs = {
                ex.submit(
                    _compile_segment_worker,
                    (hit_s, model_name, out_s, device, export_opts, i),
                ): i
                for i in range(n)
            }
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    idx, seg_meta = fut.result()
                except Exception as exc:  # noqa: BLE001
                    ex.shutdown(cancel_futures=True)
                    plan = plans[i]
                    raise RuntimeError(
                        f"neml2-compile: segment {plan.index} ({plan.kind}, "
                        f"{plan.basename}) failed: {exc}"
                    ) from exc
                results[idx] = seg_meta
    finally:
        if queue is not None:
            queue.put(_PROGRESS_SENTINEL)
        if drainer is not None:
            drainer.join()
        if manager is not None:
            manager.shutdown()

    return [results[i] for i in range(n)]


def _finalize_device_meta(
    prepared: _PreparedExport,
    model_name: str,
    device: str,
    dtype: str,
    seg_metas: list[dict],
    output_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Assemble one device's top-level metadata envelope, write it, and report it.

    The envelope is device-independent except for the recorded ``device`` (the
    top-level field and each promoted parameter's ``device``), so a single
    cpu-side *prepared* can finalize every device -- which is what lets the
    multi-device grid path (:func:`export_model_multidevice`) prepare once on cpu
    and still emit per-device metadata identical to a per-device compile.
    """
    model = prepared.model
    structural_in = _structural_inputs(model.input_spec, prepared.promoted_qnames)
    in_sb = {n: sub for n, (_, sub) in (prepared.resolved_shapes or {}).items() if sub}
    meta: dict = {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": _var_infos(structural_in, sub_batch_shapes=in_sb),
        "outputs": _var_infos(model.output_spec),
        "segments": seg_metas,
    }
    # v2 top-level additions: device + dtype are baked into the artifact;
    # parameters records the promoted set with initial values.
    meta["device"] = device
    meta["dtype"] = dtype
    meta["parameters"] = _parameter_infos(
        prepared.promoted_snapshots, prepared.origin, prepared.promoted_base_shapes, device
    )
    # Master (out, in) derivative pairs, in deterministic (output, input) order.
    out_order = {n: k for k, n in enumerate(model.output_spec)}
    in_order = {n: k for k, n in enumerate(prepared.structural_input_names)}
    meta["derivatives"] = [
        [o, i]
        for (o, i) in sorted(
            prepared.master_pairs, key=lambda p: (out_order.get(p[0], 0), in_order.get(p[1], 0))
        )
    ]
    # Master (out, param) parameter-derivative pairs, in (output, param) order.
    meta["parameter_derivatives"] = [
        [o, p]
        for (o, p) in sorted(prepared.param_pairs, key=lambda x: (out_order.get(x[0], 0), x[1]))
    ]
    # Boundary renames (shallow): only namespaces with an actual rename are written.
    active_aliases = {ns: m for ns, m in prepared.boundary_aliases.items() if m}
    if active_aliases:
        meta["boundary_aliases"] = active_aliases

    meta_name = f"{model_name}_meta.json"
    _write_meta(output_dir / meta_name, meta)
    _report(progress_cb, meta_name)
    return meta


def _compile_grid_worker(args) -> tuple[str, int, dict]:
    """Worker for the multi-device grid: compile ONE ``(device, segment)`` cell.

    Like :func:`_compile_segment_worker` but keyed by device -- it writes to that
    device's artifact subfolder and tags its progress events with the device so
    the parent can distinguish the same segment across devices. Everything in
    *args* is picklable; the worker re-derives its segment from the input file.
    """
    hit_path, model_name, output_dir, device, export_opts, seg_index = args
    prepared = _prepare_export(hit_path, model_name, device=device, **export_opts)
    plans = _plan_segments(prepared, model_name)
    plan = plans[seg_index]
    queue = _worker_progress_queue
    cb = (lambda name: queue.put(f"{device}/{name}")) if queue is not None else None
    seg_meta = _execute_segment_plan(plan, prepared, Path(output_dir), device, progress_cb=cb)
    return device, seg_index, seg_meta


def _run_grid_pool(
    hit_path: str | Path,
    model_name: str,
    artifact_dir: Path,
    devices: list[str],
    n_segments: int,
    export_opts: dict,
    jobs: int,
    progress_cb: Callable[[str], None] | None,
) -> dict[str, list[dict]]:
    """Compile the full ``(device × segment)`` grid concurrently in a spawn pool.

    Up to ``min(jobs, len(devices) * n_segments)`` workers run at once, so the two
    parallelism axes (devices and segments) are flattened into one pool -- e.g.
    two devices with two segments each fully use ``-j 4``. Returns
    ``{device: [seg_meta in segment order]}``; per-``.pt2`` progress (device-tagged)
    is forwarded to *progress_cb* by a drainer thread.
    """
    import concurrent.futures as cf
    import multiprocessing as mp
    import threading

    ctx = mp.get_context("spawn")
    tasks = [(device, i) for device in devices for i in range(n_segments)]
    workers = max(1, min(jobs, len(tasks)))

    manager = None
    queue = None
    drainer = None
    initializer = None
    initargs: tuple = ()
    if progress_cb is not None:
        manager = ctx.Manager()
        queue = manager.Queue()
        initializer = _worker_init
        initargs = (queue,)

        def _drain() -> None:
            while True:
                item = queue.get()
                if item == _PROGRESS_SENTINEL:
                    break
                progress_cb(item)

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

    results: dict[tuple[str, int], dict] = {}
    hit_s = str(hit_path)
    try:
        with cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=initializer, initargs=initargs
        ) as ex:
            futs = {}
            for device, i in tasks:
                out_s = str(Path(artifact_dir) / device)
                futs[
                    ex.submit(
                        _compile_grid_worker,
                        (hit_s, model_name, out_s, device, export_opts, i),
                    )
                ] = (device, i)
            for fut in cf.as_completed(futs):
                device, i = futs[fut]
                try:
                    d, idx, seg_meta = fut.result()
                except Exception as exc:  # noqa: BLE001
                    ex.shutdown(cancel_futures=True)
                    raise RuntimeError(
                        f"neml2-compile: segment {i} on device {device!r} failed: {exc}"
                    ) from exc
                results[(d, idx)] = seg_meta
    finally:
        if queue is not None:
            queue.put(_PROGRESS_SENTINEL)
        if drainer is not None:
            drainer.join()
        if manager is not None:
            manager.shutdown()

    return {device: [results[(device, i)] for i in range(n_segments)] for device in devices}


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
    example_batch_shape: dict[str, str] | str | None = None,
    dynamic_batch: bool | None = None,
    derivatives: Sequence[str] = (),
    renames: dict[str, dict[str, str]] | None = None,
    pre: Sequence[str] = (),
    additional_args: tuple[str, ...] = (),
    jobs: int = 1,
    progress_cb: Callable[[str], None] | None = None,
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
    example_batch_shape:
        Override for the per-input example shape used at trace time. Accepts
        either a single spec string (uniform across all inputs) or a dict
        mapping input variable name to spec string (per-variable). Each spec
        uses the ``(dyn; sub)`` grammar (see
        :func:`_parse_example_batch_shape`). Overrides any
        ``[Settings]/example_batch_shape`` in the HIT file. When ``None``
        (default), the HIT setting -- or :data:`_DEFAULT_EXAMPLE_SHAPE` if
        unset -- applies.
    dynamic_batch:
        Override for ``[Settings]/dynamic_batch``. ``True`` lets the leading
        batch axis vary at runtime (the default); ``False`` produces a
        static-batch artifact pinned at the example shape. Use ``False`` when
        a baked parameter has a rank ≥ 1 shape that would specialize the
        dynamic dim. When ``None`` (default), the HIT setting -- or ``True``
        if unset -- applies.
    derivatives:
        Sequence of ``OUT:IN`` derivative specs (the ``-d/--derivative`` CLI
        surface). Each requests a Jacobian/JVP block for that output-input
        pair; omitting a side selects all on that side (``stress:`` = all
        inputs of stress, ``:strain`` = all outputs wrt strain, ``:`` = all
        pairs). Empty (the default) compiles **no** derivative graphs, and the
        runtime ``jvp`` / ``jacobian`` raise until recompiled with ``-d``.
        Resolved against the model's outputs + structural inputs after
        promotion (see :func:`_resolve_derivative_specs`).
    renames:
        Optional shallow boundary renames (the ``--rename-input`` /
        ``--rename-output`` / ``--rename-parameter`` CLI surface). A dict with
        any of ``"inputs"`` / ``"outputs"`` / ``"parameters"`` keys, each an
        ``{original_name: boundary_name}`` sub-map. Only the names reported at
        the compiled artifact's interface change; the graphs and every internal
        wiring keep the original authored names. Validated against the
        post-promotion boundary namespaces (see :func:`_validate_renames`) and
        recorded under ``boundary_aliases`` in the metadata. ``None`` (default)
        keeps the original names.
    pre:
        HIT snippets prepended before parsing (same semantics as
        ``nmhit.parse_file``'s ``pre`` arg). Use to bind ``${var}``
        substitution variables that the input file expects -- e.g.
        ``pre=["nbatch=2", "device=cpu"]``. Post-``:=`` overrides via
        *additional_args* do NOT propagate into ``${...}`` substitution.
    additional_args:
        Trailing HIT override tokens appended to the parser's *post* list
        (e.g. ``"Models/elasticity/E:=210000"``). Path-style overrides only;
        for variable substitution use *pre*.
    jobs:
        Number of worker processes for parallel segment compilation. ``1``
        (default) compiles serially (behavior-identical to before this option).
        ``> 1`` compiles independent segments concurrently in a spawn process
        pool -- effective only for a multi-segment model (a ``ComposedModel``
        containing an ``ImplicitUpdate``); single-segment models ignore it. The
        resulting ``_meta.json`` is identical to a serial run (segment metadata
        is reassembled in segment order regardless of completion order).
    progress_cb:
        Optional callback invoked with the bare filename of each generated file
        as it is written -- every ``.pt2`` graph and the ``<model>_meta.json``.
        The CLI passes a printer that renders ``[k/N] <name>`` progress. ``None``
        (default) is silent. Under ``jobs > 1`` the callback still fires per
        ``.pt2`` (forwarded from workers), then once for the metadata.

    Returns
    -------
    dict
        The metadata dictionary (same content as the written JSON).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the full pre-compile prelude (load, wrap, promote, resolve derivatives /
    # renames, freeze, seed) once. Deterministic in (input, options, device) so a
    # spawned worker can reproduce the exact same segments.
    prepared = _prepare_export(
        hit_path,
        model_name,
        device=device,
        promoted=promoted,
        example_batch_shape=example_batch_shape,
        dynamic_batch=dynamic_batch,
        derivatives=derivatives,
        renames=renames,
        pre=pre,
        additional_args=additional_args,
    )

    # Plan the segments (cheap, structural). All three model shapes -- forward,
    # single ImplicitUpdate, composed-with-implicit -- flow through one planner so
    # the artifacts predicted for progress exactly match what is produced.
    plans = _plan_segments(prepared, model_name)

    # Execute: serial (default) or, for a multi-segment model with jobs > 1, in a
    # spawn process pool where each worker re-derives its segment from the file.
    if jobs > 1 and len(plans) > 1:
        # Pass the ORIGINAL options (not the post-prepare derived values) so each
        # worker's _prepare_export re-derives from byte-identical inputs.
        export_opts = {
            "promoted": sorted(set(promoted)),
            "example_batch_shape": example_batch_shape,
            "dynamic_batch": dynamic_batch,
            "derivatives": tuple(derivatives),
            "renames": renames,
            "pre": tuple(pre),
            "additional_args": tuple(additional_args),
        }
        seg_metas = _compile_segments_parallel(
            hit_path, model_name, output_dir, device, export_opts, plans, jobs, progress_cb
        )
    else:
        seg_metas = [
            _execute_segment_plan(plan, prepared, output_dir, device, progress_cb=progress_cb)
            for plan in plans
        ]

    return _finalize_device_meta(
        prepared, model_name, device, dtype, seg_metas, output_dir, progress_cb=progress_cb
    )


def export_model_multidevice(
    hit_path: str | Path,
    model_name: str,
    artifact_dir: str | Path,
    devices: Sequence[str],
    *,
    dtype: str = "float64",
    promoted: set[str] | list[str] | tuple[str, ...] = (),
    example_batch_shape: dict[str, str] | str | None = None,
    dynamic_batch: bool | None = None,
    derivatives: Sequence[str] = (),
    renames: dict[str, dict[str, str]] | None = None,
    pre: Sequence[str] = (),
    additional_args: tuple[str, ...] = (),
    jobs: int = 1,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Compile *model_name* for every device in *devices*, parallelizing across
    the full ``(device × segment)`` grid.

    Each device's artifacts land in ``artifact_dir/<device>/`` with its own
    ``<model_name>_meta.json``; returns ``{device: meta}``. This is the CLI's
    multi-device entry point: unlike calling :func:`export_model_for_aoti` once
    per device (which parallelizes only that device's segments and runs devices
    sequentially), *jobs* here bounds the workers across ALL ``(device, segment)``
    pairs at once -- so e.g. two devices with two segments each saturate ``-j 4``,
    and the devices compile concurrently.

    *progress_cb* is invoked with a device-tagged ``"<device>/<filename>"`` for
    every generated ``.pt2`` and ``_meta.json`` (so the same segment on different
    devices is distinguishable). ``jobs`` is capped at the number of grid cells.
    Single-device / ``jobs=1`` runs are behavior-identical to a per-device
    :func:`export_model_for_aoti`.
    """
    artifact_dir = Path(artifact_dir).resolve()
    devices = list(dict.fromkeys(devices))  # de-dupe, preserve order

    def _dev_cb(device: str) -> Callable[[str], None] | None:
        if progress_cb is None:
            return None
        cb = progress_cb  # bind non-None for the closure
        return lambda name: cb(f"{device}/{name}")

    def _prepare_on(device: str) -> _PreparedExport:
        return _prepare_export(
            hit_path,
            model_name,
            device=device,
            promoted=promoted,
            example_batch_shape=example_batch_shape,
            dynamic_batch=dynamic_batch,
            derivatives=derivatives,
            renames=renames,
            pre=pre,
            additional_args=additional_args,
        )

    # Plan once on cpu -- the segment structure is device-independent, and this
    # keeps the parent from initializing CUDA (the workers own that).
    prepared_cpu = _prepare_on("cpu")
    plans = _plan_segments(prepared_cpu, model_name)
    n = len(plans)
    for device in devices:
        (artifact_dir / device).mkdir(parents=True, exist_ok=True)

    metas: dict[str, dict] = {}
    if jobs > 1 and len(devices) * n > 1:
        # Grid pool: workers re-derive each (device, segment) from picklable opts.
        export_opts = {
            "promoted": sorted(set(promoted)),
            "example_batch_shape": example_batch_shape,
            "dynamic_batch": dynamic_batch,
            "derivatives": tuple(derivatives),
            "renames": renames,
            "pre": tuple(pre),
            "additional_args": tuple(additional_args),
        }
        seg_by_dev = _run_grid_pool(
            hit_path, model_name, artifact_dir, devices, n, export_opts, jobs, progress_cb
        )
        # The cpu-side prepared finalizes every device (envelope is device-
        # independent apart from the recorded device fields).
        for device in devices:
            metas[device] = _finalize_device_meta(
                prepared_cpu,
                model_name,
                device,
                dtype,
                seg_by_dev[device],
                artifact_dir / device,
                progress_cb=_dev_cb(device),
            )
    else:
        # Serial: prepare per device (device-correct) and compile in order --
        # behavior-identical to a per-device export_model_for_aoti.
        for device in devices:
            dev_dir = artifact_dir / device
            prepared_d = _prepare_on(device)
            plans_d = _plan_segments(prepared_d, model_name)
            seg_metas = [
                _execute_segment_plan(p, prepared_d, dev_dir, device, progress_cb=_dev_cb(device))
                for p in plans_d
            ]
            metas[device] = _finalize_device_meta(
                prepared_d,
                model_name,
                device,
                dtype,
                seg_metas,
                dev_dir,
                progress_cb=_dev_cb(device),
            )
    return metas


__all__ = [
    "export_model_for_aoti",
    "export_model_multidevice",
    "plan_export_artifacts",
    "AOTI_META_SCHEMA_VERSION",
]
