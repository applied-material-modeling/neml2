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
from collections.abc import Sequence
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
#: v2: bake-by-default; promoted parameters listed under top-level
#: ``parameters`` with per-segment ``param_inputs``. No ``buffers`` section.
#: Adds top-level ``device`` + ``dtype`` keys.
#:
#: v3: implicit-segment Newton step graph split into a step-direction
#: artifact (cheap A + solve, no residual recompute) plus a C++-side
#: backtracking line search using the existing rhs graph; forward-segment
#: per-input/output metadata collapsed to ``{"name": ...}``; and
#: implicit-segment I/O moved from a single packed ``(*dyn, u_size)`` slab
#: to per-variable tensors at their natural ``(*dyn, *sub_batch, *base)``
#: shape (per-variable ``sub_batch_shape`` / ``sub_batch_labels`` /
#: ``base_shape`` recorded so the C++ side can reshape per-variable slots).
#:
#: v4: solver convergence / line-search configuration is no longer
#: baked into the metadata. The implicit-segment ``atol`` / ``rtol`` /
#: ``miters`` / ``linesearch`` keys are gone -- the generated stub ``.i``
#: carries a minimal ``[Solvers]`` block (the honored knobs only; the linear
#: solver, which is baked into the step/IFT graphs, is omitted) and the
#: ``AOTIModel`` shim forwards it to the C++ runtime at load time. The
#: predictor is unchanged: it still lowers to its own ``_predictor.pt2``
#: graph with ``predictor_package`` / ``predictor_inputs`` /
#: ``predictor_outputs`` metadata.
#:
#: v5: per-group / per-cell metadata for implicit segments
#: (``unknown_group_infos`` / ``given_group_infos`` / ``residual_group_infos``
#: / ``ift_cells``) so the C++ Newton loop + IFT composition runs per group.
#:
#: v6 (current): derivative graphs are opt-in. A new top-level ``derivatives``
#: array lists the master ``[out, in]`` pairs the artifact supports (empty =>
#: none; ``jvp`` / ``jacobian`` raise). A forward segment's ``jvp_package`` /
#: ``jacobian_pairs`` and an implicit segment's ``ift_package`` are present
#: only when some requested pair needs them, and each ``jacobian_pairs`` entry
#: gains a ``batch_independent`` flag (the block does not depend on the dynamic
#: batch, e.g. a constant elasticity tensor, so the runtime may carry / return
#: it unbatched).
# dependencies: aoti.schema_version
AOTI_META_SCHEMA_VERSION = 6


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
    # NL-param tail through RHS/NewtonStep/IFT/predictor; not yet
    # implemented.
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
            f"implicit segment's RHS/NewtonStep/IFT wrappers have fixed "
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
    from ..models.model import NLParam  # noqa: PLC0415

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


def _resolve_derivative_specs(
    specs: Sequence[str],
    output_names: Sequence[str],
    structural_input_names: Sequence[str],
) -> set[tuple[str, str]]:
    """Resolve ``-d/--derivative`` ``OUT:IN`` specs into master (out, in) pairs.

    Each spec must contain exactly one ``:``. Either side may be empty, meaning
    "all" on that side: ``stress:strain`` is one pair; ``stress:`` is every
    structural input of ``stress``; ``:strain`` is every output w.r.t.
    ``strain``; ``:`` is all pairs. An empty *specs* yields an empty set — no
    derivative graphs are compiled and the runtime ``jvp`` / ``jacobian`` raise.

    Unknown output / input names raise ``ValueError`` listing the available
    names. A promoted-parameter name on the input side is rejected the same way
    (it is not a structural input and never appears in the Jacobian).
    """
    outputs = list(output_names)
    inputs = list(structural_input_names)
    pairs: set[tuple[str, str]] = set()
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
            sel_in = inputs
        elif in_part in inputs:
            sel_in = [in_part]
        else:
            raise ValueError(
                f"--derivative {spec!r}: unknown input {in_part!r}; "
                f"available (structural) inputs: {inputs}."
            )
        for o in sel_out:
            for i in sel_in:
                pairs.add((o, i))
    return pairs


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
            # Promoted parameters arrive as raw ``torch.Tensor`` snapshots;
            # they're treated as constants by the trace (AOTI bakes them
            # or routes them through the per-segment ``param_inputs`` tail).
            # ``ComposedModel.forward`` auto-wraps via ``_coerce_to_input_type``
            # so the leaf still sees a typed value when promoted parameters
            # flow into it.
            snap = promoted_snapshots[name].to(device=device, dtype=torch.float64)
            examples.append(snap)
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
) -> tuple[str, list[dict], list[dict], str | None, list[str], list[dict] | None]:
    """Compile a single forward-shape model to ``<pkg_basename>.pt2`` plus,
    when derivatives are requested, ``<pkg_basename>_jvp.pt2`` carrying the
    per-(out, in) Jacobian blocks.

    *selected_pairs* selects which local ``(out_var, in_var)`` blocks to emit:
    ``None`` emits all pairs (legacy / all-pairs export); an empty set emits no
    derivative graph at all (forward value only); a non-empty set restricts the
    emitted blocks (and matching ``jacobian_pairs`` metadata) to that subset.

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
    from ..models.common import ComposedModel
    from ..models.export import compile_model

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
    example_inputs = _build_example_inputs(
        seg_spec, promoted_qnames, promoted_snapshots, device, shapes=shapes
    )
    dynamic_dim = 0 if dynamic_batch else None

    pkg_name = f"{pkg_basename}.pt2"
    compile_model(exportable, example_inputs, output_dir / pkg_name, dynamic_batch_dim=dynamic_dim)

    jvp_pkg_name: str | None = None
    jacobian_pairs: list[dict] | None = None
    do_jvp = selected_pairs is None or len(selected_pairs) > 0
    if do_jvp:
        # JVP wrapper differentiates along structural inputs only -- promoted
        # inputs aren't seeded so they contribute structural zeros via
        # the default chain rule's ``v.get(name, {})`` empty fallback.
        jvp_module = _ForwardJacobianModule(
            exportable, promoted_qnames, selected_pairs=selected_pairs
        ).to(device)
        # Probe eagerly first (example dynamic batch is >=2 by construction, see
        # _DEFAULT_EXAMPLE_SHAPE) to classify each emitted pair: a block whose
        # dynamic-batch axes stay size-1 does not depend on the runtime batch
        # (e.g. a constant elasticity tensor) and is recorded as
        # ``batch_independent`` so the runtime can carry / return it unbatched.
        with torch.no_grad():
            probe = jvp_module(*example_inputs)
        n_outs = len(jvp_module.output_names)
        probe_pairs = list(probe[n_outs:])
        jvp_pkg_name = f"{pkg_basename}_jvp.pt2"
        compile_model(
            jvp_module, example_inputs, output_dir / jvp_pkg_name, dynamic_batch_dim=dynamic_dim
        )
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


def _export_forward(
    model,
    model_name: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    promoted_snapshots: dict[str, torch.Tensor],
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    derivatives: set[tuple[str, str]] | None = None,
) -> dict:
    """Export a forward-shape model as a single-segment composed artifact.

    *derivatives* is the resolved master ``(out, in)`` pair set (empty = no
    derivative graph). For a single forward segment the master pairs *are* the
    segment-local pairs, so they pass straight through as ``selected_pairs``.
    """
    (
        pkg_name,
        in_infos,
        out_infos,
        jvp_pkg_name,
        param_inputs,
        jacobian_pairs,
    ) = _compile_forward_segment(
        model,
        model_name,
        output_dir,
        device,
        promoted_qnames=promoted_qnames,
        promoted_snapshots=promoted_snapshots,
        shapes=shapes,
        dynamic_batch=dynamic_batch,
        selected_pairs=derivatives if derivatives is not None else set(),
    )
    seg = {
        "kind": "forward",
        "package": pkg_name,
        "inputs": _segment_var_infos(in_infos),
        "outputs": _segment_var_infos(out_infos),
        "param_inputs": param_inputs,
    }
    if jvp_pkg_name is not None:
        seg["jvp_package"] = jvp_pkg_name
        seg["jacobian_pairs"] = jacobian_pairs
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
    *,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    emit_ift: bool = True,
    selected_ift_pairs: set[tuple[str, str]] | None = None,
) -> dict:
    """Compile an ImplicitUpdate to ``<pkg_basename>_rhs.pt2`` + ``_step.pt2``
    (+ ``_ift.pt2`` when *emit_ift*) (+ optional ``_predictor.pt2``), returning
    the metadata dict (without the outer ``"type"`` key — caller adds it).

    *selected_ift_pairs* restricts the IFT graph to the requested
    ``(unknown, given)`` pairs (``None`` = all). A requested pair whose unknown
    or given is **sub-batched** (per-grain, e.g. crystal plasticity) is rejected
    here with a clear compile-time error: the per-pair IFT consumer only supports
    plain-batch (DENSE) pairs today, so e.g. a global-output / global-input
    derivative of a crystal-plasticity model compiles, while a per-grain pair
    fails fast at ``neml2-compile`` rather than at runtime.

    ``rhs`` + ``step`` drive the forward Newton solve and are always compiled.
    ``ift`` is the user-facing derivative graph (the implicit-function-theorem
    sensitivity of the converged solution); it is compiled only when *emit_ift*
    is True (i.e. some derivative was requested via ``-d``). When False, the
    ``ift_package`` key is omitted and the runtime leaves the segment's IFT
    loader null.

    Per-variable I/O at the AOTI graph boundary (v5+): each unknown,
    given, and residual is its own positional tensor in the segment
    signature. Preserved-label per-group sub_batch storage stays
    heterogeneous-ndim end-to-end; no cross-group cat or fold-to-flat
    inside the graph (except IFT's once-per-solve flat ``du/dg`` slab).

    Promotion of parameters inside the implicit segment is rejected
    earlier (in :func:`_validate_promoted`); any promoted-parameter
    machinery is handled entirely in the forward path, so this function
    takes no promoted_* kwargs.
    """
    from ..es import IFT, RHS, AssembledVector, NewtonStep
    from ..models.common import ComposedModel
    from ..models.export import compile_model

    system = inner.system
    solver = inner.solver

    rhs = RHS(system).to(device)
    step = NewtonStep(system, solver.linear_solver).to(device)
    ift = IFT(system, solver.linear_solver, selected_pairs=selected_ift_pairs).to(device)

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
        return [t.data for t in vec.tensors]  # noqa: data-ok AOTI

    u_group_examples = _per_group_examples(system.ulayout, _example_for_unknown)
    g_group_examples = _per_group_examples(system.glayout, _example_for_given)
    example_inputs = tuple(u_group_examples) + tuple(g_group_examples)

    rhs_name = f"{pkg_basename}_rhs.pt2"
    step_name = f"{pkg_basename}_step.pt2"
    ift_name = f"{pkg_basename}_ift.pt2"

    dynamic_dim = 0 if dynamic_batch else None
    compile_model(rhs, example_inputs, output_dir / rhs_name, dynamic_batch_dim=dynamic_dim)
    compile_model(step, example_inputs, output_dir / step_name, dynamic_batch_dim=dynamic_dim)
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
        compile_model(ift, example_inputs, output_dir / ift_name, dynamic_batch_dim=dynamic_dim)

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
        # Always empty under this iteration's constraint -- promotion inside
        # implicit segments is rejected above. Recorded for schema uniformity.
        "param_inputs": [],
    }
    if emit_ift:
        seg["ift_package"] = ift_name
        # Per-(unknown, given) pair metadata for the IFT loader's output tuple,
        # in the same order the IFT graph emits blocks. Consumed via the
        # per-pair Jacobian composition (same path as a forward segment).
        seg["jacobian_pairs"] = ift_jacobian_pairs

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
        seg["predictor_package"] = pred_name
        seg["predictor_inputs"] = _segment_var_infos(pred.input_spec)
        seg["predictor_outputs"] = _segment_var_infos(pred.output_spec)

    return seg


def _export_implicit(
    model,
    inner,
    model_name: str,
    output_dir: Path,
    device: str,
    *,
    promoted_qnames: set[str],
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    derivatives: set[tuple[str, str]] | None = None,
) -> dict:
    """Export an ImplicitUpdate as a single-segment composed artifact.

    Promotion of parameters inside the implicit segment is rejected up-front
    in :func:`_validate_promoted`, so ``promoted_qnames`` only affects the
    metadata's master ``inputs`` filter here -- the trace itself is identical
    to the no-promotion case.

    *derivatives* is the resolved master pair set; the single implicit segment
    emits its IFT graph iff any pair was requested (the IFT couples every given
    to every unknown, so a single requested pair pulls in the whole graph).
    """
    # Map the requested master (out, in) pairs to local (unknown, given) IFT
    # pairs. For a single ImplicitUpdate the outputs are the unknowns and the
    # structural inputs are givens, so this is a direct filter.
    master_pairs = derivatives or set()
    sys = inner.system
    selected_ift_pairs = {
        (o, i) for (o, i) in master_pairs if o in sys.unknown_names and i in sys.given_names
    }
    seg = _compile_implicit_segment(
        inner,
        model_name,
        output_dir,
        device,
        shapes=shapes,
        dynamic_batch=dynamic_batch,
        emit_ift=bool(selected_ift_pairs),
        selected_ift_pairs=selected_ift_pairs,
    )
    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    in_sb = {n: sub for n, (_, sub) in (shapes or {}).items() if sub}
    # Output labels mirror unknowns' labels (each output is an unknown that
    # inherits sub_batch_labels from its history input via _solve).
    return {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": _var_infos(structural_in, sub_batch_shapes=in_sb),
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
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    dynamic_batch: bool = True,
    derivatives: set[tuple[str, str]] | None = None,
) -> dict:
    """Export a ComposedModel containing ImplicitUpdate children as multiple
    .pt2 artifacts.

    Each non-implicit run of leaves becomes one forward segment .pt2; each
    ImplicitUpdate becomes the standard rhs/step (+ optional predictor) set.
    The C++ ``AOTIModel`` orchestrates the segments in order.

    *derivatives* is the resolved master ``(out, in)`` pair set. Forward
    segments emit only the local pairs that lie on a dependency path between a
    requested master input and a requested master output (a forward segment with
    no such pair emits no derivative graph); an implicit segment emits its IFT
    graph iff it lies on any requested path. Correctness gate: a local pair is
    kept whenever its row reaches a requested output AND its column is reachable
    from a requested input — never dropping a pair that carries a requested
    sensitivity. Empty *derivatives* → no derivative graphs anywhere.
    """
    from ..models.common import ComposedModel, ImplicitUpdate

    segments = _partition_into_segments(model)
    master_pairs = derivatives or set()

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

    # --- Dependency reachability for the per-segment derivative prune ---
    # ``reach[V]`` = master (structural) inputs that can flow into V; ``canreach
    # [V]`` = master outputs V can reach. Each segment is modelled as "every
    # output depends on every input" (the all-pairs internal Jacobian; an
    # implicit segment's IFT couples every given to every unknown), which is the
    # correct conservative data-flow model. A local pair (o, i) carries a
    # requested master sensitivity (o_m, i_m) iff ``i_m ∈ reach[i]`` and
    # ``o_m ∈ canreach[o]`` — keep it iff that holds for some requested pair.
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

    def _keep_local_pair(o: str, i: str) -> bool:
        return any(
            (i_m in reach.get(i, ())) and (o_m in canreach.get(o, ()))
            for (o_m, i_m) in master_pairs
        )

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
            # Local pairs to emit: those on a dependency path between a requested
            # master input and a requested master output (empty -> no jvp graph).
            seg_struct_in = _structural_inputs(seg_model.input_spec, promoted_qnames)
            seg_selected = {
                (o, i)
                for o in seg_model.output_spec
                for i in seg_struct_in
                if _keep_local_pair(o, i)
            }
            (
                pkg_name,
                in_infos,
                out_infos,
                jvp_pkg_name,
                param_inputs,
                jacobian_pairs,
            ) = _compile_forward_segment(
                seg_model,
                basename,
                output_dir,
                device,
                promoted_qnames=promoted_qnames,
                promoted_snapshots=promoted_snapshots,
                shapes=shapes,
                dynamic_batch=dynamic_batch,
                selected_pairs=seg_selected,
            )
            seg_entry = {
                "kind": "forward",
                "package": pkg_name,
                "inputs": _segment_var_infos(in_infos),
                "outputs": _segment_var_infos(out_infos),
                "param_inputs": param_inputs,
            }
            if jvp_pkg_name is not None:
                seg_entry["jvp_package"] = jvp_pkg_name
                seg_entry["jacobian_pairs"] = jacobian_pairs
            seg_metas.append(seg_entry)
        else:
            # payload is the ImplicitUpdate leaf.
            impl_model = payload
            assert isinstance(impl_model, ImplicitUpdate)
            # Local (unknown, given) IFT pairs on a requested dependency path:
            # the given reachable from a requested master input AND the unknown
            # reaching a requested master output. Drives both whether to emit the
            # IFT graph and which per-pair blocks it emits.
            isys = impl_model.system
            selected_ift_pairs = {
                (u, g)
                for u in isys.unknown_names
                for g in isys.given_names
                if _keep_local_pair(u, g)
            }
            seg = _compile_implicit_segment(
                impl_model,
                basename,
                output_dir,
                device,
                shapes=shapes,
                dynamic_batch=dynamic_batch,
                emit_ift=bool(selected_ift_pairs),
                selected_ift_pairs=selected_ift_pairs,
            )
            seg_metas.append({"kind": "implicit", **seg})

    structural_in = _structural_inputs(model.input_spec, promoted_qnames)
    in_sb = {n: sub for n, (_, sub) in (shapes or {}).items() if sub}
    return {
        "schema_version": AOTI_META_SCHEMA_VERSION,
        "type": "composed",
        "inputs": _var_infos(structural_in, sub_batch_shapes=in_sb),
        "outputs": _var_infos(model.output_spec),
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
    example_batch_shape: dict[str, str] | str | None = None,
    dynamic_batch: bool | None = None,
    derivatives: Sequence[str] = (),
    pre: Sequence[str] = (),
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

    Returns
    -------
    dict
        The metadata dictionary (same content as the written JSON).
    """
    from ..factory import load_input
    from ..models.common import ImplicitUpdate

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = load_input(hit_path, pre=pre, additional_args=additional_args)
    model = factory.get_model(model_name)

    # Resolve example-batch-shape declarations: CLI/Python kwarg wins, then
    # HIT [Settings], then the (2,)/uniform default. The full per-input map
    # is what every downstream helper consumes.
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

    # Resolve -d/--derivative specs into the master (out, in) pair set, against
    # the post-promotion outputs + structural inputs (promoted parameters never
    # appear in the Jacobian). Empty => no derivative graphs are compiled.
    structural_input_names = list(_structural_inputs(model.input_spec, promoted_qnames))
    master_pairs = _resolve_derivative_specs(
        derivatives, list(model.output_spec), structural_input_names
    )

    # Freeze any remaining nn.Parameter to a persistent buffer so torch.export
    # bakes it into the graph instead of lifting it as a graph input. Promoted
    # entries are already gone from _parameters so they're skipped naturally.
    _freeze_remaining_parameters_to_buffers(model)

    # Catch baked-parameter shape conflicts before torch.export does — the
    # NEML2-side message names the param, the conflicting input, and the
    # three available resolutions; torch.export's "you marked batch as
    # dynamic but specialized it to N" message names neither.
    if dynamic_batch:
        _validate_baked_against_shapes(model, resolved_shapes, promoted_qnames)

    # Populate per-variable ``sub_batch_shape`` on every inner
    # ``ModelNonlinearSystem`` from the resolved per-variable shapes so the
    # Schur/Block export path traces with the correct per-sub-batch-site
    # layout. The compile path no longer reaches into [Drivers] for this.
    _seed_implicit_subbatch(model, resolved_shapes, device)

    inner = model

    if isinstance(inner, ImplicitUpdate):
        meta = _export_implicit(
            model,
            inner,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            shapes=resolved_shapes,
            dynamic_batch=dynamic_batch,
            derivatives=master_pairs,
        )
    elif _contains_implicit(inner):
        meta = _export_composed(
            model,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            promoted_snapshots=promoted_snapshots,
            shapes=resolved_shapes,
            dynamic_batch=dynamic_batch,
            derivatives=master_pairs,
        )
    else:
        meta = _export_forward(
            model,
            model_name,
            output_dir,
            device,
            promoted_qnames=promoted_qnames,
            promoted_snapshots=promoted_snapshots,
            shapes=resolved_shapes,
            dynamic_batch=dynamic_batch,
            derivatives=master_pairs,
        )

    # v2 top-level additions: device + dtype are baked into the artifact;
    # parameters records the promoted set with initial values.
    meta["device"] = device
    meta["dtype"] = dtype
    meta["parameters"] = _parameter_infos(promoted_snapshots, origin)
    # Master (out, in) derivative pairs the artifact supports, in deterministic
    # (output-order, input-order) order so the runtime iterates rows/cols
    # consistently. Empty => no derivative graphs (jvp/jacobian raise).
    out_order = {n: k for k, n in enumerate(model.output_spec)}
    in_order = {n: k for k, n in enumerate(structural_input_names)}
    meta["derivatives"] = [
        [o, i]
        for (o, i) in sorted(
            master_pairs, key=lambda p: (out_order.get(p[0], 0), in_order.get(p[1], 0))
        )
    ]

    _write_meta(output_dir / f"{model_name}_meta.json", meta)
    return meta


__all__ = ["export_model_for_aoti", "AOTI_META_SCHEMA_VERSION"]
