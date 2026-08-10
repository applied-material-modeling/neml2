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

"""Route-neutral reader for the HIT ``[Settings]`` block.

``[Settings]/example_batch_shape`` declares each variable's batch structure as
two regions separated by ``;`` -- ``'(2; 100)'`` means a dynamic-batch shape of
``(2,)`` and a sub-batch shape of ``(100,)``. The two regions are read by
different audiences:

* The **dynamic** region is a nominal ``torch.export`` trace hint. It seeds
  Inductor's ``size_hints`` and nothing else -- at run time the leading batch
  axis is whatever the caller passes. Only the AOTI export path reads it.
* The **sub-batch** region is a real, static property of the variable: a
  per-slip-system or per-bin axis that every route must agree on. It is the
  only way to know a sub-batched variable's shape *before* a tensor for it
  exists -- which is exactly the case for an implicit unknown with no initial
  condition and no predictor.

This module therefore lives outside :mod:`neml2.cli`, so the eager path
(:class:`~neml2.drivers.TransientDriver`,
:class:`~neml2.models.common.ImplicitUpdate`,
:class:`~neml2.es.ModelNonlinearSystem`, the pyzag adapter) resolves the same
declaration through the same code as ``neml2-compile``. Parity is an invariant
(see ``CLAUDE.md``): a declaration that means one thing to the exporter and
another to the driver is a bug in this module, not in either consumer.

Reach it via :attr:`neml2.factory._NativeInputFile.settings` rather than calling
:func:`read_settings` directly -- every ``from_hit`` already receives that handle.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nmhit

#: Key under which the uniform (non-per-variable) declaration is stored.
UNIFORM_KEY = "*"

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
#: override via ``example_batch_shape=`` on
#: :func:`~neml2.cli.aoti_export.export_model_for_aoti` or
#: ``--example-batch-shape`` on the CLI; the benchmark suite does
#: this (see ``benchmark/run_benchmark.py``).
DEFAULT_EXAMPLE_SHAPE: tuple[tuple[int, ...], tuple[int, ...]] = ((2,), ())


def parse_example_batch_shape(
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


def validate_declared_names(
    declared: Mapping[str, str],
    known: Collection[str],
    *,
    universe: str = "model input_spec",
) -> None:
    """Raise if *declared* names a variable absent from *known*.

    An unknown key is almost always a typo (e.g. ``"stress"`` written instead
    of ``"strain"``); silently ignoring it would mask the bug and leave the
    variable it was meant to describe at its inferred shape.
    """
    extras = set(declared) - {UNIFORM_KEY} - set(known)
    if extras:
        raise ValueError(
            f"example_batch_shape names not in {universe}: {sorted(extras)}. "
            f"Available: {sorted(known)}."
        )


def resolve_example_shapes(
    input_spec: Iterable[str],
    declared: Mapping[str, str],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Map each name in *input_spec* to its ``(dyn, sub)`` shape tuple.

    Resolution order:

    1. Per-variable entry in *declared* (e.g. ``declared["strain"]``).
    2. Uniform entry in *declared* (key :data:`UNIFORM_KEY`).
    3. :data:`DEFAULT_EXAMPLE_SHAPE`.

    This is the *export-time* resolution: it assigns every input a shape,
    falling back to the default. The eager path wants to distinguish "declared
    as un-sub-batched" from "not declared at all" and uses
    :meth:`Settings.sub_batch_shape` instead.

    Unknown keys in *declared* raise -- see :func:`validate_declared_names`.
    """
    names = list(input_spec)
    validate_declared_names(declared, names)
    uniform_spec = declared.get(UNIFORM_KEY)
    resolved: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for name in names:
        if name in declared:
            resolved[name] = parse_example_batch_shape(declared[name])
        elif uniform_spec is not None:
            resolved[name] = parse_example_batch_shape(uniform_spec)
        else:
            resolved[name] = DEFAULT_EXAMPLE_SHAPE
    return resolved


@dataclass(frozen=True)
class Settings:
    """The resolved contents of a HIT ``[Settings]`` block.

    ``example_batch_shape`` holds the *raw* spec strings keyed by variable
    name, with the uniform form under :data:`UNIFORM_KEY`. Consumers should go
    through :meth:`sub_batch_shape` / :meth:`sub_batch_ndim` (eager) or
    :meth:`resolve` (export) rather than reading the dict.
    """

    example_batch_shape: Mapping[str, str] = field(default_factory=dict)
    dynamic_batch: bool = True

    def __post_init__(self) -> None:
        self._check_lag_agreement()

    # ── declaration lookup ────────────────────────────────────────────────────

    def sub_batch_shape(self, name: str) -> tuple[int, ...] | None:
        """*name*'s declared sub-batch extent, or ``None`` if undeclared.

        A spec declares a sub-batch extent only where it *writes* one -- that
        is, only if it carries the ``;`` separator. ``'(2,)'`` names a dynamic
        batch and says nothing at all about sub-batch, so it reads as ``None``
        here even though :func:`resolve_example_shapes` (which must assign
        every input a full ``(dyn, sub)`` pair for the trace) reports its sub
        region as ``()``. That distinction is what lets a file declare a
        production batch size uniformly -- ``example_batch_shape =
        '(${nbatch},)'``, as every benchmark input does -- without thereby
        asserting that no variable in the file has a sub-batch axis.

        ``()`` is therefore a deliberate claim ("written, and empty":
        ``'(2; )'``) and conflicts with a sub-batched value, while ``None``
        means nothing was said and the consumer falls back to inference.

        Lookup runs over the entries that write a sub-batch region, in order:
        the exact key, the bare variable name, any other lag of the same
        variable (they agree by :meth:`_check_lag_agreement`), the uniform
        entry.
        """
        declared = self.example_batch_shape
        base = _base_name(name)
        candidates = [declared.get(name), declared.get(base)]
        candidates += [
            spec for key, spec in declared.items() if key != UNIFORM_KEY and _base_name(key) == base
        ]
        candidates.append(declared.get(UNIFORM_KEY))
        for spec in candidates:
            if spec is not None and _declares_sub_batch(spec):
                return parse_example_batch_shape(spec)[1]
        return None

    def sub_batch_ndim(self, name: str) -> int | None:
        """``len(self.sub_batch_shape(name))``, or ``None`` if undeclared."""
        sub = self.sub_batch_shape(name)
        return None if sub is None else len(sub)

    def resolve(
        self, input_spec: Iterable[str]
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        """Export-time ``(dyn, sub)`` per input -- see :func:`resolve_example_shapes`."""
        return resolve_example_shapes(input_spec, self.example_batch_shape)

    def validate_names(self, known: Collection[str], *, universe: str) -> None:
        """Raise if a declared name is absent from *known* -- see
        :func:`validate_declared_names`."""
        validate_declared_names(self.example_batch_shape, known, universe=universe)

    # ── internal ──────────────────────────────────────────────────────────────

    def _check_lag_agreement(self) -> None:
        """A sub-batch axis belongs to the variable, not to the lag.

        ``elastic_strain`` and ``elastic_strain~1`` are the same quantity one
        step apart, so a per-slip axis on one is a per-slip axis on the other.
        Declaring them differently would let the driver and the residual model
        disagree about the same variable's layout, so it is rejected here --
        once, at parse time -- rather than surfacing as a shape mismatch deep
        in Newton. The *dynamic* regions may still differ per lag, and a lag
        that writes no sub-batch region at all is silent rather than in
        conflict (see :meth:`sub_batch_shape`).
        """
        by_base: dict[str, dict[str, tuple[int, ...]]] = {}
        for key, spec in self.example_batch_shape.items():
            if key == UNIFORM_KEY or not _declares_sub_batch(spec):
                continue
            by_base.setdefault(_base_name(key), {})[key] = parse_example_batch_shape(spec)[1]
        for base, entries in by_base.items():
            if len(set(entries.values())) > 1:
                detail = ", ".join(f"{k}={v}" for k, v in sorted(entries.items()))
                raise ValueError(
                    f"[Settings]/example_batch_shape: {base!r} and its history lags must "
                    f"declare the same sub-batch extent -- a sub-batch axis belongs to the "
                    f"variable, not to the lag. Got {detail}."
                )


def read_settings(root: nmhit.Root) -> Settings:
    """Read the ``[Settings]`` block off a parsed HIT *root*.

    Two HIT forms are accepted for ``example_batch_shape``::

        [Settings]
          example_batch_shape = '(2,)'         # uniform → key '*'
        []

        [Settings]
          [example_batch_shape]                # per-variable → one key per entry
            strain      = '(2; 100)'
            temperature = '(2,)'
          []
        []

    Everything defaults cleanly when no ``[Settings]`` block is present.
    """
    import nmhit  # noqa: PLC0415

    example_shapes: dict[str, str] = {}
    dynamic_batch = True

    settings = None
    for top in root.children(nmhit.NodeType.Section):
        if top.path() == "Settings":
            settings = top
            break
    if settings is None:
        return Settings(example_shapes, dynamic_batch)

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
            example_shapes[UNIFORM_KEY] = uniform

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

    return Settings(example_shapes, dynamic_batch)


def _declares_sub_batch(spec: str) -> bool:
    """True iff *spec* writes a sub-batch region -- i.e. carries the ``;``."""
    return ";" in spec


def _base_name(name: str) -> str:
    """``'x~1'`` → ``'x'``. Imported lazily: :mod:`neml2.es` pulls in
    :mod:`neml2.factory`, which imports this module."""
    from .es._helpers import lag_order  # noqa: PLC0415

    return lag_order(name)[0]


__all__ = [
    "DEFAULT_EXAMPLE_SHAPE",
    "UNIFORM_KEY",
    "Settings",
    "parse_example_batch_shape",
    "read_settings",
    "resolve_example_shapes",
    "validate_declared_names",
]
