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

"""Declarative HIT syntax for Python-native objects.

``HitSchema`` is the native counterpart of the C++ ``expected_options`` pattern,
but intentionally smaller: it records fixed model inputs/outputs and the HIT
options needed to construct common leaf models.  The :class:`Model` base class
uses it to derive ``input_spec`` / ``output_spec`` and to provide a default
``from_hit`` implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import TensorWrapper

if TYPE_CHECKING:
    import nmhit

    from .factory import _NativeInputFile

_MISSING = object()

#: Sentinel ``default`` for :func:`var_output`: the variable name defaults to the
#: model's HIT block name (mirrors C++ ``Interpolation`` whose output variable
#: defaults to the object name). Resolved against ``node.path()`` at parse time.
BLOCK_NAME = object()

#: Field kinds whose HIT-option *value* names one or more model variables.
_VAR_KINDS = frozenset({"var_inputs"})


@dataclass(frozen=True)
class HitField:
    kind: str
    name: str
    value_type: Any = None
    doc: str = ""
    default: Any = _MISSING
    attr: str | None = None
    reader: Callable[[nmhit.Node, str], Any] | None = None
    optional_reader: Callable[[nmhit.Node, str, Any], Any] | None = None
    factory_getter: str | None = None
    allow_promotion: bool = False
    #: For ``input``/``output`` fields: a literal string appended to the
    #: HIT-resolved variable name, used to derive the conventional NEML2
    #: secondary-variable names (``~1`` for history, ``_rate`` for rates,
    #: ``_residual`` for implicit residuals). Multiple input/output fields can
    #: share the same ``name`` (HIT option) with different ``suffix``es to
    #: derive several related variables from one user-supplied base name.
    suffix: str | None = None
    #: For ``input``/``output`` fields: the name of another HIT option whose
    #: non-empty value takes precedence over ``name + suffix`` for the
    #: resolved variable name. Used by ``rate`` overrides — when the user
    #: supplies an explicit ``rate = 'my_rate_var'`` they bypass the
    #: ``<variable>_rate`` derivation.
    override: str | None = None
    #: For ``output`` fields: an explicit producer priority used by the
    #: :class:`DependencyResolver` when more than one model provides the
    #: same variable name. ``"high"`` = "my output supersedes any other
    #: producer; run me last." ``"low"`` = "my output is overwritten by
    #: any other producer; run me first." ``None`` (default) = "I'm the
    #: sole producer — duplicate-provider error if not."
    priority: str | None = None

    @property
    def ctor_name(self) -> str:
        return self.attr or self.name

    @property
    def required(self) -> bool:
        return self.default is _MISSING

    @property
    def var_type(self) -> type[TensorWrapper] | None:
        """The variable ``TensorWrapper`` type, for ``var_*`` / ``input`` / ``output``."""
        if self.kind in _VAR_KINDS or self.kind in {"input", "output"}:
            return self.value_type if isinstance(self.value_type, type) else None
        return None

    @property
    def is_input(self) -> bool:
        return self.kind in {"input", "var_input", "var_inputs"}

    @property
    def is_output(self) -> bool:
        return self.kind in {"output", "var_output"}

    @property
    def implicit_override_name(self) -> str | None:
        """HIT option that renames this field's derived variable, or ``None``.

        A ``derived_output``'s default name — the referenced option name plus
        ``suffix`` (e.g. ``back_stress`` + ``_rate`` = ``back_stress_rate``) —
        doubles as an optional rename knob (see :func:`_read_derived_var_name`);
        this returns that knob's name. Only ``derived_output`` fields with a
        ``suffix`` expose one — ``derived_input`` does not. Single source of
        truth for the resolver, the unknown-field validator, and the syntax
        catalog.
        """
        if self.kind == "derived_output" and self.suffix:
            return f"{self.name}{self.suffix}"
        return None


class HitSchema:
    """Ordered declaration of a native object's HIT surface."""

    def __init__(self, *fields: HitField) -> None:
        self.fields = tuple(fields)
        for field in self.fields:
            # ``derived_*`` fields don't appear in the syntax catalog (their
            # name and behavior are derived from the referenced field), so
            # their doc would never be rendered — don't burden authors with a
            # required-but-invisible docstring.
            if field.kind in {"derived_input", "derived_output"}:
                continue
            _validate_doc(field.name, field.doc)

    @property
    def input_spec(self) -> dict[str, type[TensorWrapper]]:
        return {
            f.name: f.value_type
            for f in self.fields
            if f.kind == "input" and isinstance(f.value_type, type)
        }

    @property
    def output_spec(self) -> dict[str, type[TensorWrapper]]:
        return {
            f.name: f.value_type
            for f in self.fields
            if f.kind == "output" and isinstance(f.value_type, type)
        }

    def kwargs_from_hit(self, node: nmhit.Node, factory: _NativeInputFile) -> dict[str, Any]:
        """Parse constructor kwargs for every schema field.

        ``input`` / ``output`` fields resolve their canonical variable name from
        HIT (with the option name as the default, or :data:`BLOCK_NAME` /
        ``None`` per :func:`input` / :func:`output`); the resolved name is
        passed under :attr:`HitField.ctor_name` (``attr`` or option name) so
        ``Model._store_schema_values`` can record per-instance ``input_spec`` /
        ``output_spec`` keys without going through a separate remapping pass.

        ``derived_input`` / ``derived_output`` fields reference another option
        by name and reuse that option's resolution (default included). A
        precomputed ``option_values`` map lets the derived resolver see e.g.
        ``time → "t"`` when the option declared ``default="t"`` and HIT was
        silent.
        """
        # Precompute resolved string values for every name-bearing field
        # (input / output / string option) so derived_* fields can reuse them
        # by referencing the field's ``name``. This is what lets
        # ``derived_input("variable", ...)`` use the value of the
        # ``input("variable", ...)`` field — the user declares the primary
        # variable once and the secondary suffixed names fall out of it.
        name_values: dict[str, str | None] = {}
        for field in self.fields:
            if field.kind in {"input", "output"}:
                name_values[field.name] = _read_var_name(node, field)
            elif field.kind == "option" and field.value_type is str:
                name_values[field.name] = _read_str(node, field.name, field.default)

        kwargs: dict[str, Any] = {}
        for field in self.fields:
            if field.kind == "dependency":
                option_value = _read_str(node, field.name, field.default)
                getter = getattr(factory, field.factory_getter or "")
                kwargs[field.ctor_name] = getter(option_value)
                continue
            if field.kind == "var_inputs":
                kwargs[field.ctor_name] = _read_var_names(node, field)
                continue
            if field.kind in {"parameters", "buffers"}:
                kwargs[field.ctor_name] = _read_parameter_list(node, field)
                continue
            if field.kind in {"input", "output"}:
                kwargs[field.ctor_name] = name_values[field.name]
                continue
            if field.kind in {"derived_input", "derived_output"}:
                if field.name not in name_values:
                    raise ValueError(
                        f"{field.kind} {field.attr or field.name!r} references "
                        f"{field.name!r}, but no input/output/option field of that "
                        "name is declared in the schema."
                    )
                kwargs[field.ctor_name] = _read_derived_var_name(node, field, name_values)
                continue
            kwargs[field.ctor_name] = _read_field(node, field)

        return kwargs

    def reject_unknown_fields(self, node: nmhit.Node) -> None:
        """Raise if *node* carries any HIT field not declared in the schema.

        Mirrors the C++ parser's behaviour: an unrecognised option is a hard
        error, not a silent skip. Silently ignored options are how stale
        knobs (e.g. ``priority`` on a ``ComposedModel`` after the resolver
        stopped supporting it) survive across refactors.

        ``type`` is always allowed -- it's read by the factory before the
        schema-driven kwargs pass to dispatch the registered class.
        Override option names (an :func:`option`'s ``override=`` target) are
        also allowed; they substitute for the primary option's value when
        non-empty. A :func:`derived_output`'s default name (referenced name +
        suffix) is likewise allowed: it doubles as an implicit rename knob (see
        :func:`_read_derived_var_name`).
        """
        allowed: set[str] = {"type"}
        for field in self.fields:
            if field.kind in {"derived_input", "derived_output"}:
                # Purely virtual — a derived field introduces no HIT option of
                # its own, except that a derived_output's default name doubles
                # as an implicit rename knob (see _read_derived_var_name).
                knob = field.implicit_override_name
                if knob is not None:
                    allowed.add(knob)
                continue
            allowed.add(field.name)
            if field.override:
                allowed.add(field.override)
        import nmhit  # noqa: PLC0415

        seen: list[str] = [c.path().rsplit("/", 1)[-1] for c in node.children(nmhit.NodeType.Field)]
        unknown = sorted(set(seen) - allowed)
        if not unknown:
            return
        type_name = node.param_optional_str("type", "?")
        block_path = node.path()
        accepted_str = ", ".join(sorted(allowed - {"type"})) or "<none>"
        raise ValueError(
            f"[{block_path}] (type={type_name!r}): unknown option(s) {unknown}. "
            f"Accepted: {accepted_str}. Remove or rename the offending "
            "entries -- silently ignored options have masked real bugs in the past."
        )


def input(  # noqa: A001
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    default: Any = _MISSING,
    attr: str | None = None,
) -> HitField:
    """Declare an input variable.

    The canonical variable name resolves as ``HIT[name]`` if specified, else
    *default* (which itself defaults to the option name). Pass
    ``default=BLOCK_NAME`` to fall back to the model's HIT block name, or
    ``default=None`` to keep the resolved name optional (yielding ``None``
    when HIT is silent — used by leaves that compute a derived default).
    When *attr* is given, ``Model.__init__`` stores the resolved name on the
    instance under that attribute (for leaves that reference the variable name
    explicitly inside ``forward``, e.g. as a chain-rule action key).

    To derive a *secondary* variable name from another option (time-integration
    pattern: history ``~1``, ``_rate``, ``_residual``), use :func:`derived_input`
    / :func:`derived_output` instead — those declare a variable without
    introducing a new HIT option.
    """
    return HitField("input", name, type_cls, doc=doc, default=default, attr=attr)


def output(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    default: Any = _MISSING,
    attr: str | None = None,
    priority: str | None = None,
) -> HitField:
    """Declare an output variable.

    Same resolution rules as :func:`input` (HIT override → *default* → option
    name); pass ``default=BLOCK_NAME`` for "default to the HIT block name"
    (the C++ ``Interpolation`` convention) or ``default=None`` for a derived
    default the leaf computes itself. For secondary names derived from another
    option, see :func:`derived_output`.

    Set *priority* to disambiguate composition when multiple sibling models
    provide the same variable name:

    * ``"high"`` — "my output supersedes any other producer; run me last."
      The composed model returns this leaf's value.
    * ``"low"``  — "my output is overwritten by any other producer; run me
      first." The composed model returns the other (default-priority) leaf's
      value.
    * ``None`` (default) — claim the name exclusively. Duplicate-provider
      error if any sibling also provides it.

    Used by post-processors like ``FixOrientation`` that mutate a sibling's
    output in place (``input='orientation'`` / ``output='orientation'``).
    """
    if priority not in (None, "high", "low"):
        raise ValueError(
            f"output({name!r}): priority must be None, 'high', or 'low'; got {priority!r}."
        )
    return HitField(
        "output", name, type_cls, doc=doc, default=default, attr=attr, priority=priority
    )


def derived_input(
    referenced: str,
    type_cls: type[TensorWrapper],
    *,
    attr: str | None = None,
    suffix: str | None = None,
    override: str | None = None,
) -> HitField:
    """Declare an input variable whose name is *derived* from another field's value.

    Unlike :func:`input`, a derived field does *not* declare its own HIT option
    — it references an existing :func:`input` / :func:`output` / :func:`option`
    field by ``referenced`` (the field's ``name``) and computes the variable
    name as ``HIT[referenced] + suffix`` (or ``HIT[override]`` when that
    override option is set non-empty). This is the schema-driven counterpart
    to the time-integrator pattern of declaring one base ``variable`` input
    and deriving ``<variable>~1`` / ``<variable>_rate`` / ``<variable>_residual``
    from it: the user sees a single option per logical knob, and the secondary
    variables fall out automatically. No docstring is required — derived
    fields don't appear in the syntax catalog.
    """
    return HitField(
        "derived_input",
        referenced,
        type_cls,
        attr=attr,
        suffix=suffix,
        override=override,
    )


def derived_output(
    referenced: str,
    type_cls: type[TensorWrapper],
    *,
    attr: str | None = None,
    suffix: str | None = None,
    override: str | None = None,
) -> HitField:
    """Declare an output variable whose name is *derived* from another field's value.

    See :func:`derived_input`. The canonical use is
    ``derived_output("variable", T, suffix="_residual")`` for an implicit
    residual output named after the base ``variable`` option/input.

    Unlike :func:`derived_input`, a derived output is renamable by default: its
    *default* name — the referenced field name plus ``suffix`` (e.g.
    ``back_stress`` + ``_rate`` = ``back_stress_rate``) — doubles as a HIT
    rename knob, so ``back_stress_rate = 'X_rate'`` renames the output the same
    way ``stress = 'my_stress'`` renames a regular :func:`output`. The knob name
    is the *static* default (it does not shift when the referenced base is
    renamed); the value assigned to it is the new output name, and it takes
    precedence over the ``base + suffix`` cascade.
    """
    return HitField(
        "derived_output",
        referenced,
        type_cls,
        attr=attr,
        suffix=suffix,
        override=override,
    )


def var_inputs(
    option_name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    default: Any = _MISSING,
    attr: str | None = None,
    ctor: str | None = None,
) -> HitField:
    """Declare a *list* HIT option whose values name several inputs of one type.

    The C++ ``LinearCombination`` ``from = 'a b c'`` pattern: the option is a
    whitespace-separated list of variable names, each an input of ``type_cls``.
    When ``attr`` is provided, ``Model.__init__`` stores the parsed
    ``list[str]`` on the instance under that attribute (and extends
    ``input_spec`` automatically); otherwise the value is passed as a
    constructor kwarg. ``ctor`` is a deprecated alias for ``attr``.
    """
    return HitField(
        "var_inputs", option_name, type_cls, doc=doc, default=default, attr=attr or ctor
    )


def parameter(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
    allow_promotion: bool = False,
) -> HitField:
    return HitField(
        "parameter",
        name,
        type_cls,
        doc=doc,
        default=default,
        attr=attr,
        allow_promotion=allow_promotion,
    )


def parameters(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
    allow_promotion: bool = False,
) -> HitField:
    """Declare a *list* of parameters, one per token of a HIT list option.

    The HIT value is a whitespace-separated list of specs; each token is
    declared as an independent parameter (named ``<attr-or-name>_<i>``) via
    :meth:`Model.declare_typed_parameter`, so each entry can independently
    resolve as a literal, a ``[Tensors]`` cross-ref, a ``[Models]``-output
    promoted input (mode 3) or a bare-variable promoted input (mode 4) — same
    as a single :func:`parameter`. The list of registered parameter names is
    stored on ``self.<attr>`` for the leaf to iterate inside ``forward``
    (typically via :meth:`Model._get_param_list`).
    """
    return HitField(
        "parameters",
        name,
        type_cls,
        doc=doc,
        default=default,
        attr=attr,
        allow_promotion=allow_promotion,
    )


def buffer(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
) -> HitField:
    """Declare a typed buffer — a non-trainable, constant typed tensor.

    Resolves the HIT value through :meth:`Model.declare_typed_buffer`, which
    accepts the same spec shapes as :func:`parameter` (mode 1 literal, mode 2
    ``[Tensors]`` cross-ref) but does *not* support input-promotion (modes
    3/4). Buffers are frozen at construction time and baked into the AOTI
    artifact as constants — use this for material constants like gravity,
    Boltzmann's constant, or any value the user might want to override per
    block but never differentiate against.

    Replaces the manual two-step pattern of declaring a fixed default inside
    ``__post_init__`` and calling :meth:`Model.register_typed_buffer` —
    schemas can now declare the buffer in one line, with a default visible
    in the auto-generated syntax catalog.
    """
    return HitField(
        "buffer",
        name,
        type_cls,
        doc=doc,
        default=default,
        attr=attr,
    )


def buffers(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
) -> HitField:
    """Declare a *list* of typed buffers, one per token of a HIT list option.

    Mirrors :func:`parameters` for buffer semantics. Each list entry is
    registered as ``<attr-or-name>_<i>`` via :meth:`Model.declare_typed_buffer`
    and the list of registered names is stored on ``self.<attr>`` for the
    leaf to iterate inside ``forward``.
    """
    return HitField(
        "buffers",
        name,
        type_cls,
        doc=doc,
        default=default,
        attr=attr,
    )


def option(
    name: str,
    value_type: type,
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
    reader: Callable[[nmhit.Node, str], Any] | None = None,
    optional_reader: Callable[[nmhit.Node, str, Any], Any] | None = None,
) -> HitField:
    """Declare a scalar/list/string HIT option.

    When *attr* is provided and the model uses the inherited ``Model.__init__``,
    the parsed value is stored directly on the instance under that attribute
    name. Without *attr*, the parsed value is passed as a constructor kwarg and
    a custom ``__init__`` must consume it.
    """
    return HitField(
        "option",
        name,
        value_type,
        doc=doc,
        default=default,
        attr=attr,
        reader=reader,
        optional_reader=optional_reader,
    )


def dependency(
    name: str,
    factory_getter: str,
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
) -> HitField:
    return HitField(
        "dependency",
        name,
        str,
        doc=doc,
        default=default,
        attr=attr,
        factory_getter=factory_getter,
    )


def _validate_doc(name: str, doc: str) -> None:
    if not doc:
        raise ValueError(f"HitSchema field {name!r} must have a non-empty doc string")
    try:
        doc.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(f"HitSchema field {name!r} doc string must be ASCII") from e


def _read_field(node: nmhit.Node, field: HitField) -> Any:
    if field.kind in {"parameter", "buffer"}:
        # Parameter / buffer specs are normally strings (HIT literal, [Tensors]
        # cross-ref, or — for parameters — a model-output / bare-variable
        # promotion spec). But defaults may be typed wrappers (``Vec.fill(...)``
        # for a buffer constant, etc.). When the HIT block doesn't supply the
        # option, fall through to the raw default rather than ``str()``-ifying
        # it via ``_read_str`` and breaking the wrapper.
        if node.find(field.name) is None and not field.required:
            return field.default
        return _read_str(node, field.name, field.default)
    if field.reader is not None:
        if field.required:
            return field.reader(node, field.name)
        if field.optional_reader is None:
            raise ValueError(f"optional schema field {field.name!r} has no optional reader")
        return field.optional_reader(node, field.name, field.default)
    if field.value_type is str:
        return _read_str(node, field.name, field.default)
    if field.value_type is int:
        return _read_int(node, field.name, field.default)
    if field.value_type is float:
        return _read_float(node, field.name, field.default)
    if field.value_type is bool:
        # HIT bools accept ``true``/``false`` strings; route through nmhit's
        # native bool reader rather than coercing via ``int`` (which would
        # raise on ``true``).
        return bool(node.param_optional_bool(field.name, bool(field.default)))
    raise TypeError(
        f"HitSchema option {field.name!r} has unsupported value_type "
        f"{field.value_type!r}; provide reader/optional_reader."
    )


def _read_var_name(node: nmhit.Node, field: HitField) -> str | None:
    """Resolve a variable-name field (``input`` / ``output`` / ``var_*``) from HIT.

    Override / suffix (input/output only): the ``override`` HIT option, when
    set to a non-empty value, replaces the entire derivation; otherwise a
    non-empty ``suffix`` is appended to the resolved base name.

    Base resolution (after override/suffix are skipped):

    * ``default=BLOCK_NAME`` — HIT value if specified, else the model's HIT
      block name (the trailing segment of ``node.path()``).
    * ``default=None`` — HIT value if specified, else ``None`` (the leaf
      computes a derived name in its constructor).
    * No ``default`` (``_MISSING``) — HIT value if specified, else the option
      name itself. This is the standard "fall back to the canonical name"
      behavior that makes ``input("foo", T)`` mean "HIT-optional override
      of the canonical name ``foo``".
    * Literal string ``default`` — HIT value if specified, else that literal.
    """
    if field.override is not None:
        override_val = node.param_optional_str(field.override, "")
        if override_val:
            return override_val

    if field.default is BLOCK_NAME:
        if node.find(field.name) is not None:
            base = node.param_str(field.name)
        else:
            base = node.path().rsplit("/", 1)[-1]
    elif field.default is _MISSING:
        base = node.param_optional_str(field.name, field.name)
    else:
        base = _read_str(node, field.name, field.default)

    if field.suffix and base is not None:
        return f"{base}{field.suffix}"
    return base


def _read_derived_var_name(
    node: nmhit.Node, field: HitField, name_values: dict[str, str | None]
) -> str | None:
    """Resolve a ``derived_input`` / ``derived_output`` field's variable name.

    Resolution precedence:

    1. Explicit ``override`` (if set): check HIT for that named option; when
       non-empty, it short-circuits the derivation and is returned as-is.
    2. Implicit override (``derived_output`` only): the field's *default*
       derived name — the referenced option name plus ``suffix`` (e.g.
       ``back_stress`` + ``_rate`` = ``back_stress_rate``) — doubles as a HIT
       rename knob. When the user sets that option non-empty, it wins, exactly
       as setting a regular :func:`output`'s option renames it. This makes
       every derived output renamable with no per-model opt-in.
    3. Otherwise the base value comes from ``name_values[field.name]`` (the
       already-resolved value of the referenced input/output/option) and
       ``suffix`` is appended.
    """
    if field.override is not None:
        override_val = node.param_optional_str(field.override, "")
        if override_val:
            return override_val
    knob = field.implicit_override_name
    if knob is not None:
        auto_val = node.param_optional_str(knob, "")
        if auto_val:
            return auto_val
    base = name_values.get(field.name)
    if field.suffix and base is not None:
        return f"{base}{field.suffix}"
    return base


def _read_var_names(node: nmhit.Node, field: HitField) -> list[str]:
    """Resolve the variable-name list a ``var_inputs`` option points at."""
    if field.required:
        return list(node.param_list_str(field.name))
    if node.find(field.name) is None:
        return list(field.default)
    return list(node.param_list_str(field.name))


def _read_parameter_list(node: nmhit.Node, field: HitField) -> list[str]:
    """Resolve a :func:`parameters` list — each token is a per-entry spec.

    Each returned string is later fed to :meth:`Model.declare_typed_parameter`
    as if it were a single :func:`parameter` field's value.
    """
    if field.required:
        return list(node.param_list_str(field.name))
    if node.find(field.name) is None:
        return list(field.default) if field.default else []
    return list(node.param_list_str(field.name))


def _read_str(node: nmhit.Node, name: str, default: Any) -> str | None:
    if default is _MISSING:
        return node.param_str(name)
    # ``default=None`` marks an optional option that yields ``None`` when absent
    # (so the constructor can apply its own fallback, e.g. an optional parameter).
    if default is None:
        return node.param_str(name) if node.find(name) is not None else None
    return node.param_optional_str(name, str(default))


def _read_int(node: nmhit.Node, name: str, default: Any) -> int:
    if default is _MISSING:
        return int(node.param_int(name))
    return int(node.param_optional_int(name, int(default)))


def _read_float(node: nmhit.Node, name: str, default: Any) -> float:
    if default is _MISSING:
        return float(node.param_float(name))
    return float(node.param_optional_float(name, float(default)))


__all__ = [
    "BLOCK_NAME",
    "HitField",
    "HitSchema",
    "dependency",
    "derived_input",
    "derived_output",
    "input",
    "option",
    "output",
    "parameter",
    "parameters",
    "var_inputs",
]
