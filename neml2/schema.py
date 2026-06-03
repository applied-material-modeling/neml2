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
    allow_nonlinear: bool = False
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
            if field.kind == "parameters":
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
) -> HitField:
    """Declare an output variable.

    Same resolution rules as :func:`input` (HIT override → *default* → option
    name); pass ``default=BLOCK_NAME`` for "default to the HIT block name"
    (the C++ ``Interpolation`` convention) or ``default=None`` for a derived
    default the leaf computes itself. For secondary names derived from another
    option, see :func:`derived_output`.
    """
    return HitField("output", name, type_cls, doc=doc, default=default, attr=attr)


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
    allow_nonlinear: bool = False,
) -> HitField:
    return HitField(
        "parameter",
        name,
        type_cls,
        doc=doc,
        default=default,
        attr=attr,
        allow_nonlinear=allow_nonlinear,
    )


def parameters(
    name: str,
    type_cls: type[TensorWrapper],
    doc: str,
    *,
    attr: str | None = None,
    default: Any = _MISSING,
    allow_nonlinear: bool = False,
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
        allow_nonlinear=allow_nonlinear,
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
    if field.kind == "parameter":
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

    ``override`` (if set): check HIT for the named option; when non-empty,
    that value short-circuits the derivation and is returned as-is. Else the
    base value comes from ``name_values[field.name]`` (the already-resolved
    value of the referenced input/output/option) and ``suffix`` is appended.
    """
    if field.override is not None:
        override_val = node.param_optional_str(field.override, "")
        if override_val:
            return override_val
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
