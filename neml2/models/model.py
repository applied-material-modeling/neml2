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

"""Model — base class for Python-native NEML2 models.

Most subclasses declare a class-level ``hit = HitSchema(...)``; the base class
derives ``input_spec`` / ``output_spec`` (variable name → type) from its
``input(...)`` / ``output(...)`` fields and provides a default ``from_hit``.
Dynamic-I/O models may still set ``input_spec`` / ``output_spec`` directly and
override ``from_hit``. Every subclass implements :meth:`forward`.

**Unified forward contract**::

    # Pure forward — returns outputs only.
    out = model(*inputs)

    # Forward + Jacobian pushforward —
    # ``v[in_name][leaf_name]`` is a typed wrapper with leading seed axis K:
    # data shape ``(K, *B, *sub, *base_in)``.
    # Returns ``(*outputs, v_out_dict)`` where ``v_out_dict[out_name][leaf_name]``
    # is the output's typed wrapper with the same leading K. No explicit
    # Jacobian block is materialised inside the chain-rule graph.
    *vals, v_out = model(*inputs, v={"name": {"leaf": sensitivity_matrix, ...}})

    # Forward + first- and second-order Jacobian pushforward (opt-in) —
    # ``v2[in_name][seed_a][seed_b]`` is a typed wrapper with two leading seed
    # axes ``(N_a, N_b, *B, *sub, *base_in)``.
    # Returns ``(*outputs, v_out, v2_out)``. Only models that may appear inside
    # a Normality wrap implement v2; callers passing v2 must also pass v.
    *vals, v_out, v2_out = model(*inputs, v=..., v2=...)

Variable names in ``input_spec`` / ``output_spec`` are plain strings with no
hierarchical prefix, e.g. ``"strain"``, ``"stress"``, ``"plastic_strain"``.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import torch
from torch import nn

from ..types import TensorWrapper
from ..types._base import SubBatchStateFlag
from .chain_rule import (
    ChainRuleAction,
    ChainRuleDict,
    SecondOrderChainRuleAction,
    SecondOrderChainRuleDict,
    TangentAction,
)

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile
    from ..schema import HitSchema
    from ..types import Tensor

#: Bound to the concrete typed-wrapper class a caller expects from
#: :meth:`Model._get_param`, so the return type is narrowed at the call site.
_TW = TypeVar("_TW", bound=TensorWrapper)

#: Raised when a leaf reads a registered parameter as ``self.<attr>`` inside a
#: forward. Such a read bypasses the static-or-promoted dispatch in
#: :meth:`Model._get_param`, so it works only until that parameter is promoted to
#: a runtime input (``-p``) -- then the ``nn.Parameter`` is gone and the read
#: fails (or, worse, silently differentiates the wrong thing). Routing every
#: parameter read through ``_get_param`` keeps the leaf promotion-compatible.
_PARAM_ATTR_GUARD_MSG = (
    "{cls}.forward reads the registered parameter '{name}' as an attribute "
    "(self.{name}). This bypasses parameter promotion: once '{name}' is promoted "
    "to a runtime input (neml2-compile -p), the static nn.Parameter no longer "
    "exists and the read breaks. Read it through "
    "`self._get_param('{name}', promoted_params, <Type>)` instead (add `*promoted_params` to "
    "the forward signature if absent) -- that resolves the static slot or the "
    "promoted runtime input uniformly. For a list of parameters use "
    "`self._get_param_list('<attr>', promoted_params, <Type>)`."
)

__all__ = [
    "TangentAction",
    "ChainRuleDict",
    "ChainRuleAction",
    "SecondOrderChainRuleDict",
    "SecondOrderChainRuleAction",
    "Model",
    "PromotedParam",
    "register_submodule",
]


def register_submodule(
    parent: nn.Module,
    child: nn.Module,
    fallback: str,
    *,
    used: set[str] | None = None,
) -> str:
    """Add *child* to *parent* under its HIT block name if available.

    The factory stamps ``_hit_name`` on every object it constructs; preferring
    that name over an opaque attribute slot keeps ``named_parameters()``
    readable (``elasticity.E`` instead of ``_residual_model.E``). Falls back to
    *fallback* when the HIT name is missing (direct Python construction), is
    not a valid Python identifier, would collide with an existing attribute on
    *parent*, or is already in *used* (when a parent registers several children
    in one pass and must avoid collisions across siblings).

    Returns the attribute name the child was registered under.
    """
    hit_name = getattr(child, "_hit_name", None)
    attr = hit_name if hit_name and hit_name.isidentifier() else None
    if attr is None or (used is not None and attr in used) or hasattr(parent, attr):
        attr = fallback
    if used is not None:
        used.add(attr)
    parent.add_module(attr, child)
    return attr


@dataclass(frozen=True)
class PromotedParam:
    """Marker for a parameter resolved to a runtime input (modes 3 + 4).

    Records the input variable name added to the host's ``input_spec``, the
    parameter's position within ``forward``'s ``*promoted_params`` pack, and — for
    mode 3 — the provider model + its output variable name so the parent
    :class:`~neml2.models.common.ComposedModel.ComposedModel` can auto-pull the provider
    into the dependency graph (mirroring the C++ ``_promoted_params`` bookkeeping
    in ``ParameterStore.cxx::resolve_tensor_name``).

    ``tail_index`` is the zero-based slot of this parameter inside the
    ``*promoted_params`` pack passed to :meth:`Model._get_param`. Promoted parameters
    are appended to ``input_spec`` in declaration order, immediately after the
    fixed structural inputs, so this index is simply the number of parameters
    already promoted when this one was declared.

    For mode 4 (no provider — pure input promotion), ``provider`` is ``None``.
    """

    input_name: str
    tail_index: int
    provider: Model | None = None
    provider_output: str | None = None


class Model(nn.Module, ABC):
    """Base class for Python-native NEML2 models with declared variable names.

    Subclasses declare ``input_spec`` and ``output_spec`` as class-level dicts
    (static models) or instance attributes set in ``__init__`` (models whose
    output count depends on constructor arguments).

    ``input_spec`` key order matches ``forward()``'s positional argument order;
    ``output_spec`` key order matches the return-tuple / return-value order.

    When called without ``v``, ``forward`` returns the outputs directly (a
    single typed wrapper or a tuple thereof).  When called with ``v``, it
    additionally returns the sensitivity dict as the final element of the tuple.
    """

    #: HIT section every registered subclass belongs to. Inherited; subclasses
    #: that deliberately live elsewhere (none today) can override.
    SECTION: ClassVar[str] = "Models"

    input_spec: dict[str, type[TensorWrapper]]
    output_spec: dict[str, type[TensorWrapper]]
    #: HIT-bound output name → priority claim (``"high"`` / ``"low"`` /
    #: ``None``) sourced from each :func:`~neml2.schema.output` field's
    #: ``priority=`` kwarg. The :class:`~neml2.resolver.DependencyResolver`
    #: reads this to lift the duplicate-provider error when sibling models
    #: provide the same name with disambiguating priorities, and to add
    #: ``low → default → high`` ordering edges so the highest-priority
    #: writer runs last. Names absent from the dict default to ``None``.
    output_priorities: dict[str, str | None] = {}
    hit: HitSchema
    #: Tracks the typed-wrapper class for every name registered via
    #: :meth:`register_typed_buffer` or :meth:`register_typed_parameter`, so
    #: ``__getattr__`` can re-wrap the raw tensor / parameter on read.
    _typed_storage_classes: dict[str, type[TensorWrapper]]
    _typed_storage_sub_batch_ndim: dict[str, int]

    #: Opt-in flag: True iff this Model's ``forward`` accepts ``v2`` and ``vh``
    #: kwargs and propagates them via :meth:`apply_chain_rule_2`. Default False —
    #: most leaves only implement first-order chain rule and don't need v2.
    #: A leaf must set this to True if it may appear inside a
    #: :class:`~neml2.models.solid_mechanics.plasticity.Normality` wrap (directly or
    #: transitively through a ComposedModel); Normality's constructor walks the
    #: inner chain and raises if any leaf has this flag unset.
    #:
    #: Plain attribute (not ClassVar) so ``ComposedModel`` can shadow it on
    #: instances based on whether all its children support v2.
    SUPPORTS_SECOND_ORDER: bool = False

    # V2P-3: list_deriv / EdgeInfo machinery removed. The chain rule no
    # longer dispatches on per-(out, in) labels; positional K_pairing on
    # tangents carries per-axis intent.

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        schema = cls.__dict__.get("hit")
        if schema is None:
            return
        cls.input_spec = schema.input_spec
        cls.output_spec = schema.output_spec

    def __init__(self, **hit_values) -> None:
        super().__init__()
        self._typed_storage_classes = {}
        self._typed_storage_sub_batch_ndim = {}
        #: Parameter slot name → :class:`PromotedParam` for any parameter that was
        #: resolved to a runtime input (modes 3 + 4). The slot also appears in
        #: ``input_spec`` keyed by ``PromotedParam.input_name``; :meth:`_get_param`
        #: reads from positional inputs for these and from ``self`` otherwise.
        self._promoted_params: dict[str, PromotedParam] = {}
        #: Resolved ``{(output_name, input_name)}`` pairs whose first-order chain
        #: rule is auto-derived by reverse-mode autodiff instead of a hand-written
        #: ``forward(v=)`` branch. Populated by :meth:`request_AD`; empty for the
        #: usual analytic leaf. When non-empty, :meth:`__call__` routes a
        #: derivative request through :meth:`_ad_pushforward`.
        self._ad_pairs: set[tuple[str, str]] = set()
        self._store_schema_values(hit_values)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Hook called at the end of :meth:`__init__` for ctor-time logic.

        Default is a no-op. Subclasses override when they need construction-time
        validation or normalization that the schema can't express (e.g. weights
        length-broadcast, mutually-exclusive option checks). Runs after
        ``_store_schema_values`` but *before* :meth:`from_hit` declares
        auto-parameters — so attr-stored options / dependencies / var_* names
        are available on ``self``, but parameters are not yet registered.
        """

    def _store_schema_values(self, hit_values: dict[str, Any]) -> None:
        """Apply schema field values to the instance.

        For ``input`` / ``output`` / ``var_*`` fields, the resolved variable
        name (HIT override → schema *default* → option name) is written to
        ``self.<attr>`` when ``attr`` is set, and the per-instance
        ``input_spec`` / ``output_spec`` is updated so its keys are the
        resolved names rather than the class-level option names. Any
        class-level ``list_deriv`` is re-keyed through the same option-name →
        resolved-name map. This is the schema-side replacement for the old
        ``VariableRemapper`` boundary wrap (the factory used to insert one
        when HIT specified non-default variable names).

        For ``option`` / ``dependency`` fields, the value is stored on
        ``self.<attr>`` when ``attr`` is set.

        Any key in *hit_values* that no field declares storage for is rejected
        with ``TypeError`` — the leaf needs ``attr=`` on the field, or a
        custom ``__init__`` to consume it.
        """
        schema = getattr(type(self), "hit", None)
        name_kinds = {"input", "output", "var_inputs", "derived_input", "derived_output"}
        _typed_storage_kinds = {"parameter", "parameters", "buffer", "buffers"}
        storable_kinds = {"option", "dependency"} | _typed_storage_kinds | name_kinds

        # ``factory`` is reserved: from_hit may inject it for parameter
        # resolution, and direct Python construction may pass it too.
        factory = hit_values.pop("factory", None)

        storable: set[str] = set()
        for field in getattr(schema, "fields", ()):
            if field.kind in name_kinds or field.kind in _typed_storage_kinds:
                # Name-bearing and parameter fields are always accepted —
                # under either the schema option name (user-friendly direct
                # ctor) or the attr name when one is declared.
                storable.add(field.name)
                if field.attr is not None:
                    storable.add(field.attr)
            elif field.kind in storable_kinds and field.attr is not None:
                # Options/dependencies are only consumed when storage is
                # declared; accept either the option name or the attr.
                storable.add(field.name)
                storable.add(field.attr)
        unknown = set(hit_values) - storable
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(
                f"{type(self).__name__} received schema value(s) {names} but does not define "
                "__init__ to consume them. Add attr=... for automatic storage or override "
                "__init__."
            )

        instance_input: dict[str, type[TensorWrapper]] | None = None
        instance_output: dict[str, type[TensorWrapper]] | None = None
        rename_map: dict[str, str] = {}
        # Parameter / buffer declarations deferred to pass 2 so mode-3/4
        # promoted-input writes (which mutate ``self.input_spec`` inside
        # ``declare_typed_parameter``) don't get clobbered by the
        # ``instance_input`` assignment at the bottom of pass 1. Each entry
        # is ``(name, value, field)`` — pass 2 dispatches on ``field.kind`` to
        # the parameter or buffer declarator.
        deferred_typed: list[Any] = []

        # Resolve the base variable names (input/output/option fields) up front so
        # ``derived_*`` fields can be built as ``base + suffix`` on the DIRECT
        # Python-construction path too. The HIT path already resolves them in
        # ``kwargs_from_hit`` / ``_read_derived_var_name`` (and passes the resolved
        # name in ``hit_values``); without this, a directly-constructed model with a
        # ``derived_input(..., suffix="~1")`` would key the input by the bare
        # referenced name (no suffix), colliding with the base output.
        from ..schema import _MISSING  # noqa: PLC0415

        base_names: dict[str, str] = {}
        for field in getattr(schema, "fields", ()):
            if field.kind not in {"input", "output", "option"}:
                continue
            vk = field.attr if (field.attr is not None and field.attr in hit_values) else field.name
            if vk in hit_values and isinstance(hit_values[vk], str):
                base_names[field.name] = hit_values[vk]
            elif isinstance(field.default, str):
                base_names[field.name] = field.default
            else:
                base_names[field.name] = field.name

        for field in getattr(schema, "fields", ()):
            if field.kind not in storable_kinds:
                continue
            is_name = field.kind in name_kinds
            is_singular = field.kind in {"parameter", "buffer"}
            is_plural = field.kind in {"parameters", "buffers"}

            # Look up the value under attr (HIT-driven path uses ctor_name=attr)
            # or under the option name (direct Python ctor uses the user-friendly
            # schema name). attr wins when both are present. ``derived_*`` fields
            # share their ``name`` with the referenced base field, so they must NOT
            # fall back to that base's value (which would drop the ``suffix``);
            # they resolve as base+suffix in the ``is_name`` branch below, and only
            # their own ``attr`` slot (populated by the HIT path) is a direct value.
            if field.kind in {"derived_input", "derived_output"}:
                value_key = field.attr
            else:
                value_key = (
                    field.attr
                    if field.attr is not None and field.attr in hit_values
                    else field.name
                )
            if value_key in hit_values:
                value = hit_values[value_key]
            elif is_name:
                # Direct Python construction without HIT. ``derived_*`` fields
                # resolve through their referenced base name plus ``suffix`` (or an
                # ``override`` option value), mirroring ``_read_derived_var_name`` on
                # the HIT path. Plain input/output fields fall back to the schema
                # default, or the option name itself when there is none (_MISSING) —
                # the canonical-name convention ``_read_var_name`` applies on HIT.
                if field.kind in {"derived_input", "derived_output"}:
                    override_val = hit_values.get(field.override) if field.override else None
                    if isinstance(override_val, str) and override_val:
                        value = override_val
                    else:
                        base = base_names.get(field.name, field.name)
                        value = f"{base}{field.suffix}" if field.suffix else base
                else:
                    value = field.name if field.default is _MISSING else field.default
            elif (is_singular or is_plural) and not field.required:
                # Direct Python construction without HIT: a parameter/buffer
                # with a schema default (e.g. literal "1.0") still needs
                # declaring.
                value = field.default
            elif field.attr is not None and not field.required:
                value = field.default
            else:
                continue

            if is_singular:
                # Defer to pass 2 — declare_typed_parameter may extend
                # input_spec (mode 3/4), which would race against the
                # instance_input assignment below. Buffers are deferred for
                # symmetry even though they never extend input_spec.
                deferred_typed.append((field.ctor_name, value, field))
                continue

            if is_plural:
                # Expand each list entry into its own parameter/buffer named
                # ``<attr-or-name>_<i>``. Defer to pass 2 (same reason as
                # the singular case). Store the registered names on
                # ``self.<attr>`` so the leaf can iterate them in forward.
                if isinstance(value, str):
                    specs = value.split()
                elif isinstance(value, (list, tuple)):
                    specs = list(value)
                else:
                    raise TypeError(
                        f"{field.kind}({field.name!r}): expected list/tuple/string, "
                        f"got {type(value).__name__}"
                    )
                base = field.attr or field.name
                names = [f"{base}_{i}" for i in range(len(specs))]
                for name, spec in zip(names, specs, strict=True):
                    deferred_typed.append((name, spec, field))
                if field.attr is not None:
                    setattr(self, field.attr, names)
                continue

            if field.attr is not None:
                setattr(self, field.attr, value)

            if not is_name:
                continue

            if instance_input is None:
                instance_input = dict(type(self).input_spec)
            if instance_output is None:
                instance_output = dict(type(self).output_spec)

            if field.kind == "input":
                if isinstance(value, str):
                    instance_input.pop(field.name, None)
                    instance_input[value] = field.value_type
                    if value != field.name:
                        rename_map[field.name] = value
                else:
                    # value is None (default=None) — leaf rebuilds the spec
                    # in __post_init__ once it computes the derived name.
                    instance_input.pop(field.name, None)
            elif field.kind == "derived_input":
                # Derived fields reference an existing option (field.name); the
                # base option's class-level input_spec entry (if any) is owned
                # by that option, not this derived field, so don't pop it.
                if isinstance(value, str):
                    instance_input[value] = field.value_type
            elif field.kind == "var_inputs":
                if isinstance(value, list):
                    for name in value:
                        instance_input[name] = field.value_type
            elif field.kind == "output":
                if isinstance(value, str):
                    instance_output.pop(field.name, None)
                    instance_output[value] = field.value_type
                    if value != field.name:
                        rename_map[field.name] = value
                    if field.priority is not None:
                        self.output_priorities = {
                            **self.output_priorities,
                            value: field.priority,
                        }
                else:
                    instance_output.pop(field.name, None)
            elif field.kind == "derived_output":
                if isinstance(value, str):
                    instance_output[value] = field.value_type

        if instance_input is not None:
            self.input_spec = instance_input
        if instance_output is not None:
            self.output_spec = instance_output

        # V2P-3: list_deriv removed.

        # Stash the rename map so apply_chain_rule(_2) can translate the
        # leaf's option-name keys to the resolved external names a caller's
        # tangent dict actually carries. Skip when no rename happened.
        if rename_map:
            self._var_renames = rename_map

        # Pass 2: declare parameters / buffers. Runs after instance_input/output
        # were applied, so mode-3/4 promoted-input writes inside
        # ``declare_typed_parameter`` stick. ``buffer`` / ``buffers`` fields
        # dispatch to ``declare_typed_buffer``, which has no input-promotion
        # modes (constants only).
        for name, value, field in deferred_typed:
            if field.kind in {"buffer", "buffers"}:
                self.declare_typed_buffer(
                    name,
                    value,
                    field.value_type,
                    factory=factory,
                )
            else:
                self.declare_typed_parameter(
                    name,
                    value,
                    field.value_type,
                    factory=factory,
                    allow_promotion=field.allow_promotion,
                )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Any:
        """Construct this model from its declarative ``HitSchema``.

        Every schema field — options, dependencies, input/output renames,
        ``var_inputs`` lists, and parameters — flows through
        ``_store_schema_values`` during construction; a leaf whose only state
        is its schema needs no ``__init__`` at all. Models with dynamic I/O or
        non-trivial construction logic may still override this method.
        """
        schema = getattr(cls, "hit", None)
        if schema is None:
            raise TypeError(
                f"{cls.__name__}.from_hit requires a class-level HitSchema or an override."
            )
        from inspect import Parameter, signature

        params = signature(cls).parameters
        accepts_var_kw = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
        kwargs = schema.kwargs_from_hit(node, factory)
        kwargs["factory"] = factory

        if accepts_var_kw:
            # Leaf inherits Model.__init__(**hit_values) or otherwise forwards
            # everything to super(); the base + _store_schema_values handle all
            # schema fields uniformly.
            return cls(**kwargs)

        # Leaf has a custom __init__ that names specific kwargs. Pass only
        # those it accepts; route any remaining schema fields through
        # _store_schema_values after construction so input/output renames,
        # var_inputs, attr-stored options etc. still apply.
        accepted = {k: v for k, v in kwargs.items() if k in params}
        leftover = {k: v for k, v in kwargs.items() if k not in params and k != "factory"}
        obj = cls(**accepted)
        if leftover:
            leftover["factory"] = factory
            obj._store_schema_values(leftover)
        return obj

    def __call__(self, *args, **kwargs) -> Any:
        # ``-> Any`` matches ``nn.Module.__call__`` and the dynamic forward
        # contract (a leaf returns a typed wrapper, a tuple of them, or -- with
        # ``v`` -- ``(*outputs, v_out)``); without it the ``_ad_pushforward``
        # branch below would narrow the inferred return to ``tuple`` and break
        # every ``model(x).data`` call site under pyright.
        # Open a forward window so the AOTI-incompatible-function guard
        # (see _guard.py / DECISION.md D-060) is armed for the whole forward
        # subtree. Re-entrant: nested child Model calls just bump the depth.
        from ._guard import _enter_forward, _exit_forward

        _enter_forward()
        try:
            # request_AD leaves write only a value forward; when a first-order
            # derivative is requested, supply the chain rule by reverse-mode
            # autodiff instead of dispatching ``v`` into ``forward``.
            if kwargs.get("v") is not None and getattr(self, "_ad_pairs", None):
                return self._ad_pushforward(args, kwargs)
            return super().__call__(*args, **kwargs)
        finally:
            _exit_forward()

    # ------------------------------------------------------------------
    # Typed-buffer / typed-parameter support
    # ------------------------------------------------------------------

    def register_typed_buffer(
        self, name: str, value: TensorWrapper, persistent: bool = True
    ) -> None:
        """Register a typed tensor buffer (no autograd; baked as a constant by AOTI export)."""
        from ..factory import _check_python_attr_name  # noqa: PLC0415  (avoid import cycle)

        _check_python_attr_name(name, kind="buffer", owner=type(self).__name__)
        self.register_buffer(name, value.data, persistent=persistent)
        self._typed_storage_classes[name] = type(value)
        self._typed_storage_sub_batch_ndim[name] = value.sub_batch_ndim

    def register_typed_parameter(self, name: str, value: TensorWrapper) -> None:
        """Register a typed tensor as a calibration-tracked ``nn.Parameter``.

        Mirrors :meth:`register_typed_buffer` but stores via
        :meth:`nn.Module.register_parameter`, so the value appears in
        ``model.parameters()`` and PyTorch autograd flows through it in eager
        mode. AOTI export converts these back to constants before tracing (see
        ``aoti_export._freeze_parameters_to_buffers``); the forward-only AOTI
        graph is unchanged.
        """
        from ..factory import _check_python_attr_name  # noqa: PLC0415  (avoid import cycle)

        _check_python_attr_name(name, kind="parameter", owner=type(self).__name__)
        self.register_parameter(name, nn.Parameter(value.data))
        self._typed_storage_classes[name] = type(value)
        self._typed_storage_sub_batch_ndim[name] = value.sub_batch_ndim

    def declare_typed_parameter(
        self,
        name: str,
        spec,
        type_cls: type[TensorWrapper],
        *,
        factory: _NativeInputFile | None = None,
        allow_promotion: bool = False,
    ) -> None:
        """Resolve *spec* and register it as a parameter or promote it to an input.

        Python mirror of C++ ``ParameterStore::declare_parameter``
        (``src/neml2/models/ParameterStore.cxx``). Resolution order:

        1. ``TensorWrapper`` / ``torch.Tensor`` / ``float`` / ``int`` — wrap as
           ``type_cls`` and call :meth:`register_typed_parameter` (mode 1 with
           an already-loaded literal/batched value).
        2. ``str``:

           a. Try parse as a whitespace-separated list of floats — register as
              a typed parameter (mode 1, literal HIT value).
           b. If a *factory* is available, try ``factory.get_tensor(spec)`` —
              register as a typed parameter (mode 2, ``[Tensors]`` cross-ref).
           c. If *allow_promotion*: parse the string as a variable specifier
              (``model_name`` / ``model_name.var`` / ``var``). If it matches a
              ``[Models]`` entry, pull the provider and record the input
              promotion + provider in ``_promoted_params`` (mode 3). Otherwise treat
              the string as a bare variable name and add the input without a
              provider (mode 4).

        The host model's ``input_spec`` is extended in modes 3 + 4 with an
        entry keyed by the chosen input variable name (the provider's output
        name in mode 3, or the bare variable name in mode 4), appended after
        the fixed structural inputs. Inside ``forward`` — declared as
        ``forward(self, <structural inputs...>, *promoted_params)`` — fetch the value
        with :meth:`_get_param`, which resolves a static slot from ``self`` or
        a promoted slot from the ``*promoted_params`` pack uniformly.
        """
        # ── mode 1: already-loaded value ──────────────────────────────────────
        if isinstance(spec, TensorWrapper):
            self.register_typed_parameter(name, spec)
            return
        if isinstance(spec, torch.Tensor):
            self.register_typed_parameter(name, type_cls(spec.to(dtype=torch.float64)))
            return
        if isinstance(spec, (int, float)):
            self.register_typed_parameter(
                name, type_cls(torch.as_tensor(float(spec), dtype=torch.float64))
            )
            return

        if not isinstance(spec, str):
            raise TypeError(
                f"declare_typed_parameter({name!r}): unsupported spec type {type(spec).__name__}; "
                "expected float/int/Tensor/TensorWrapper/str"
            )

        # ── mode 1: HIT literal — try whitespace-separated float list ─────────
        try:
            floats = [float(tok) for tok in spec.split()]
        except ValueError:
            floats = None
        if floats is not None:
            t = torch.tensor(floats, dtype=torch.float64)
            if len(floats) == 1:
                t = t.squeeze(0)
            self.register_typed_parameter(name, type_cls(t))
            return

        # ── mode 2: factory.get_tensor lookup ([Tensors] cross-ref) ───────────
        if factory is not None:
            try:
                tensor_val = factory.get_tensor(spec)
            except KeyError:
                tensor_val = None
            if tensor_val is not None:
                # get_tensor returns torch.Tensor or TensorWrapper; wrap if needed.
                if isinstance(tensor_val, type_cls):
                    pass  # already the right typed wrapper
                elif isinstance(tensor_val, torch.Tensor):
                    tensor_val = type_cls(tensor_val.to(dtype=torch.float64))
                else:
                    raise TypeError(
                        f"[Tensors/{spec}] returned {type(tensor_val).__name__!r}, "
                        f"but parameter {name!r} expects {type_cls.__name__!r}."
                    )
                self.register_typed_parameter(name, tensor_val)
                return

        # ── mode 3/4: variable specifier ──────────────────────────────────────
        if not allow_promotion:
            raise ValueError(
                f"declare_typed_parameter({name!r}): cannot resolve spec {spec!r} as a literal "
                "or [Tensors] entry, and variable coupling is not enabled for this parameter."
            )

        tokens = spec.split(".")
        if len(tokens) not in (1, 2):
            raise ValueError(
                f"declare_typed_parameter({name!r}): invalid variable specifier {spec!r}; "
                "expected 'model_name', 'variable_name', or 'model_name.variable_name'."
            )

        provider: Model | None = None
        provider_output: str | None = None
        input_name: str

        if len(tokens) == 1:
            # Single token: try model name first, else fall back to bare variable promotion.
            mname = tokens[0]
            if factory is not None and factory.has_model(mname):
                prov: Model = factory.get_model(mname)
                out_specs = list(prov.output_spec)
                if len(out_specs) == 0:
                    raise ValueError(
                        f"declare_typed_parameter({name!r}): provider model {mname!r} has no "
                        "output variable."
                    )
                if len(out_specs) > 1:
                    raise ValueError(
                        f"declare_typed_parameter({name!r}): provider model {mname!r} has "
                        f"{len(out_specs)} output variables. Disambiguate with "
                        f"'{mname}.<variable_name>'."
                    )
                provider = prov
                provider_output = out_specs[0]
                input_name = provider_output
            else:
                # Mode 4: input promotion.
                input_name = mname
        else:
            # model.var form (mode 3, explicit output disambiguation).
            mname, vname = tokens
            if factory is None or not factory.has_model(mname):
                raise ValueError(
                    f"declare_typed_parameter({name!r}): variable specifier {spec!r} has no "
                    f"[Models/{mname}] entry."
                )
            prov2: Model = factory.get_model(mname)
            if vname not in prov2.output_spec:
                raise ValueError(
                    f"declare_typed_parameter({name!r}): model {mname!r} has no output "
                    f"variable {vname!r} (available: {list(prov2.output_spec)})."
                )
            provider = prov2
            provider_output = vname
            input_name = vname

        # Add to input_spec (must be a per-instance dict for this to be safe).
        if not isinstance(self.input_spec, dict) or self.input_spec is type(self).__dict__.get(
            "input_spec"
        ):
            # Promote class-level input_spec to an instance copy on first promoted_param.
            self.input_spec = dict(self.input_spec)
        if input_name in self.input_spec and self.input_spec[input_name] is not type_cls:
            raise ValueError(
                f"declare_typed_parameter({name!r}): would add input {input_name!r} of type "
                f"{type_cls.__name__} but an input of the same name already exists with type "
                f"{self.input_spec[input_name].__name__}."
            )
        self.input_spec[input_name] = type_cls
        self._promoted_params[name] = PromotedParam(
            input_name=input_name,
            tail_index=len(self._promoted_params),
            provider=provider,
            provider_output=provider_output,
        )

    def declare_typed_buffer(
        self,
        name: str,
        spec,
        type_cls: type[TensorWrapper],
        *,
        factory: _NativeInputFile | None = None,
    ) -> None:
        """Resolve *spec* as a constant value and register it as a typed buffer.

        Buffer-flavored sibling of :meth:`declare_typed_parameter`. Accepts
        the same literal / [Tensors]-cross-ref spec shapes (modes 1 and 2)
        but *not* the input-promotion modes (3 / 4): a buffer is a constant
        baked into the model, so promoting it to a chain-rule input would
        contradict its semantics.

        Resolution order:

        1. ``TensorWrapper`` / ``torch.Tensor`` / ``float`` / ``int`` — wrap
           as ``type_cls`` and register via :meth:`register_typed_buffer`.
        2. ``str``:

           a. Try parse as a whitespace-separated list of floats — register
              as a typed buffer (HIT literal).
           b. If a *factory* is available, try ``factory.get_tensor(spec)``
              — register as a typed buffer ([Tensors] cross-ref).

        Raises ``ValueError`` on any string spec that resolves to neither a
        literal nor a ``[Tensors]`` entry.
        """
        # ── mode 1: already-loaded value ──────────────────────────────────────
        # ``clone()`` defensively so a schema default ``Vec.fill(...)``
        # (constructed once at class-definition) doesn't end up shared across
        # every instance of the leaf — each model owns its buffer storage.
        if isinstance(spec, TensorWrapper):
            self.register_typed_buffer(name, type(spec)(spec.data.clone()))
            return
        if isinstance(spec, torch.Tensor):
            self.register_typed_buffer(name, type_cls(spec.to(dtype=torch.float64).clone()))
            return
        if isinstance(spec, (int, float)):
            self.register_typed_buffer(
                name, type_cls(torch.as_tensor(float(spec), dtype=torch.float64))
            )
            return

        if not isinstance(spec, str):
            raise TypeError(
                f"declare_typed_buffer({name!r}): unsupported spec type "
                f"{type(spec).__name__}; expected float/int/Tensor/TensorWrapper/str"
            )

        # ── mode 1: HIT literal — try whitespace-separated float list ─────────
        try:
            floats = [float(tok) for tok in spec.split()]
        except ValueError:
            floats = None
        if floats is not None:
            t = torch.tensor(floats, dtype=torch.float64)
            if len(floats) == 1:
                t = t.squeeze(0)
            self.register_typed_buffer(name, type_cls(t))
            return

        # ── mode 2: factory.get_tensor lookup ([Tensors] cross-ref) ───────────
        if factory is not None:
            try:
                tensor_val = factory.get_tensor(spec)
            except KeyError:
                tensor_val = None
            if tensor_val is not None:
                if isinstance(tensor_val, type_cls):
                    pass
                elif isinstance(tensor_val, torch.Tensor):
                    tensor_val = type_cls(tensor_val.to(dtype=torch.float64))
                else:
                    raise TypeError(
                        f"[Tensors/{spec}] returned {type(tensor_val).__name__!r}, "
                        f"but buffer {name!r} expects {type_cls.__name__!r}."
                    )
                self.register_typed_buffer(name, tensor_val)
                return

        raise ValueError(
            f"declare_typed_buffer({name!r}): cannot resolve spec {spec!r} as a literal "
            "or [Tensors] entry. Buffers are constant values — promoting them to an "
            "input is not supported (use a parameter with allow_promotion=True if you "
            "need that)."
        )

    def _get_param(
        self, name: str, promoted_params: Sequence[TensorWrapper], type_cls: type[_TW]
    ) -> _TW:
        """Return the value of parameter *name* — static attribute or promoted input.

        Models with parameters that may be either static or promoted should
        call this from ``forward`` instead of reading ``self.<name>`` directly.
        ``promoted_params`` is the ``*promoted_params`` pack captured by ``forward`` after
        its fixed structural inputs; a promoted parameter is read positionally
        from it via its precomputed :attr:`PromotedParam.tail_index`, and a static
        parameter is read from ``self``.

        *type_cls* is the concrete typed-wrapper class the caller expects (the
        same class passed to :meth:`declare_typed_parameter`). It only narrows
        the static return type — both runtime branches already yield that
        concrete type: static slots are re-wrapped by :meth:`__getattr__` from
        ``_typed_storage_classes``, and promoted slots are wrapped to
        ``input_spec``'s type by the caller before ``forward``.
        """
        from ._guard import _allow_param_attr  # noqa: PLC0415

        pparam = self._promoted_params.get(name)
        if pparam is None:
            # Static slot: read from ``self``. This is the ONE sanctioned
            # attribute read of a registered parameter, so lift the
            # __getattr__ guard that forbids it everywhere else in a forward.
            with _allow_param_attr():
                value = getattr(self, name)
        else:
            value = promoted_params[pparam.tail_index]
        assert isinstance(value, type_cls), (
            f"_get_param({name!r}) resolved to {type(value).__name__}, expected {type_cls.__name__}"
        )
        return value

    def _get_param_list(
        self, attr: str, promoted_params: Sequence[TensorWrapper], type_cls: type[_TW]
    ) -> list[_TW]:
        """Return values of every parameter registered via :func:`parameters`.

        The :func:`parameters` schema field registers entries as
        ``<attr>_0``, ``<attr>_1``, ... and stores the name list on
        ``self.<attr>``; this helper reads each through :meth:`_get_param`
        so individual entries that were promoted to runtime inputs (mode 3/4)
        come from ``promoted_params`` while static entries come from ``self``.
        """
        names: list[str] = getattr(self, attr)
        return [self._get_param(n, promoted_params, type_cls) for n in names]

    def __getattr__(self, name: str):
        # Forbid reading a registered parameter as a plain attribute inside a
        # forward -- it bypasses _get_param's static-or-promoted dispatch and
        # silently breaks promotion. _get_param itself reads through an
        # _allow_param_attr() window so it is exempt. Cheap guard: only true
        # parameters (in self._parameters) on a failed lookup reach this branch.
        params = self.__dict__.get("_parameters")
        if params is not None and name in params:
            from ._guard import param_attr_guarded  # noqa: PLC0415

            if param_attr_guarded():
                raise RuntimeError(_PARAM_ATTR_GUARD_MSG.format(cls=type(self).__name__, name=name))
        attr = super().__getattr__(name)
        typed = self.__dict__.get("_typed_storage_classes", {}).get(name)
        if typed is None or not isinstance(attr, torch.Tensor):
            return attr
        # Re-wrap with the per-parameter sub_batch_ndim that was captured at
        # register_typed_{buffer,parameter} time — nn.Parameter / nn.Buffer
        # only stores raw .data, so without this hint per-cell / per-bin
        # parameters would surface as sub_batch_ndim=0 and break alignment
        # against runtime inputs that carry the matching axis.
        sbn = self.__dict__.get("_typed_storage_sub_batch_ndim", {}).get(name, 0)
        return typed(attr, sub_batch_ndim=sbn)

    # ------------------------------------------------------------------
    # Named-I/O helpers
    # ------------------------------------------------------------------

    @property
    def consumed_items(self) -> frozenset[str]:
        return frozenset(self.input_spec)

    @property
    def provided_items(self) -> frozenset[str]:
        return frozenset(self.output_spec)

    def call_by_name(
        self,
        state: Mapping[str, TensorWrapper | torch.Tensor],
    ) -> dict[str, TensorWrapper]:
        """Call ``forward()`` (pure, no ``v``) with values keyed by variable name.

        Accepts typed wrappers (preferred per rule 1) or raw tensors
        (wrapped via the input_spec for caller convenience). Always
        returns typed wrappers -- consumers never have to re-attach
        metadata.
        """
        args = tuple(t(state[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        result = self(*args)
        if not isinstance(result, tuple):
            result = (result,)
        out: dict[str, TensorWrapper] = {}
        for n, v in zip(self.output_spec, result, strict=True):
            if isinstance(v, TensorWrapper):
                out[n] = v
            else:
                out[n] = self.output_spec[n](v)
        return out

    # ------------------------------------------------------------------
    # request_AD: auto-derived first-order chain rule (reverse-mode autodiff)
    # ------------------------------------------------------------------

    def request_AD(
        self,
        outputs: Sequence[str] | None = None,
        inputs: Sequence[str] | None = None,
    ) -> None:
        """Declare ``(output, input)`` derivative pairs to be auto-derived by autodiff.

        Mirrors v2's per-output ``request_AD``: a leaf that calls this writes only
        a value ``forward`` (no ``v=`` chain-rule branch) and the framework
        supplies the first-order chain rule for the declared pairs by reverse-mode
        autograd (see :meth:`_ad_pushforward` /
        :func:`neml2.models.input_ad.local_input_jacobians`). Call it from the
        leaf's ``__init__`` / ``__post_init__``.

        *outputs* defaults to every output; *inputs* defaults to every
        **structural** input -- promoted runtime parameters (``-p``) are excluded,
        as their derivatives are the separate :meth:`param_jacobian` surface. Both
        are validated against ``output_spec`` / ``input_spec``. Idempotent and
        additive (call multiple times to mix subsets).

        First-order only: an AD leaf keeps ``SUPPORTS_SECOND_ORDER = False`` and so
        cannot appear inside a :class:`~neml2.models.solid_mechanics.plasticity.Normality`
        wrap. v1 is plain-batch only (a sub-batched input/output is rejected at
        evaluation time).
        """
        out_names = list(self.output_spec) if outputs is None else list(outputs)
        promoted = {pparam.input_name for pparam in self._promoted_params.values()}
        structural = [n for n in self.input_spec if n not in promoted]
        in_names = structural if inputs is None else list(inputs)
        for o in out_names:
            if o not in self.output_spec:
                raise ValueError(
                    f"request_AD: {o!r} is not an output of {type(self).__name__} "
                    f"(available: {list(self.output_spec)})."
                )
        for i in in_names:
            if i not in self.input_spec:
                raise ValueError(
                    f"request_AD: {i!r} is not an input of {type(self).__name__} "
                    f"(available: {list(self.input_spec)})."
                )
            if i in promoted:
                raise ValueError(
                    f"request_AD: input {i!r} is a promoted runtime parameter; its "
                    "derivative is the param_jacobian surface, not the input chain rule."
                )
        for o in out_names:
            for i in in_names:
                self._ad_pairs.add((o, i))

    def _ad_pushforward(self, args: tuple, kwargs: dict) -> tuple:
        """Value + first-order ``v_out`` for a request_AD leaf via reverse-mode autodiff.

        Routed here from :meth:`__call__` when ``self._ad_pairs`` is non-empty and a
        ``v`` seed is present. Builds the dense local Jacobian for each requested
        pair by reverse-mode AD (:func:`neml2.models.input_ad.local_input_jacobians`),
        contracts each block with the incoming seed
        (:func:`neml2.types._boundary.contract_jacobian_block`), and accumulates the
        contributions through :meth:`apply_chain_rule` -- reusing the standard
        K-axis alignment + per-edge summation so the result is indistinguishable
        from a hand-written chain rule. Returns ``(*values, v_out)`` (or
        ``(value, v_out)`` for a single output), matching the forward contract.
        First-order only.
        """
        if kwargs.get("v2") is not None or kwargs.get("vh") is not None:
            raise NotImplementedError(
                f"{type(self).__name__}: request_AD is first-order only; it cannot supply "
                "the second-order chain rule (v2/vh) needed inside a Normality wrap."
            )
        from ..types._boundary import contract_jacobian_block  # noqa: PLC0415
        from .input_ad import local_input_jacobians  # noqa: PLC0415

        v = kwargs["v"]
        input_names = list(self.input_spec)
        typed_args = tuple(
            arg if isinstance(arg, type_cls) else type_cls(arg)  # type: ignore[call-arg]
            for type_cls, arg in zip(self.input_spec.values(), args, strict=True)
        )
        out_names = list(self.output_spec)
        typed_outs, blocks = local_input_jacobians(
            self, typed_args, self._ad_pairs, out_names, self.output_spec
        )

        v_out: ChainRuleDict = {}
        for o_typed, o_name in zip(typed_outs, out_names, strict=True):
            actions: dict[str, ChainRuleAction] = {}
            for in_name in input_names:
                block = blocks.get((o_name, in_name))
                if block is None:
                    continue
                o_type = type(o_typed)
                actions[in_name] = lambda V, _b=block, _ot=o_type: contract_jacobian_block(
                    _b, V, _ot
                )
            if actions:
                v_out.update(self.apply_chain_rule(v, o_name, actions, output=o_typed))

        if len(typed_outs) == 1:
            return (typed_outs[0], v_out)
        return (*typed_outs, v_out)

    # ------------------------------------------------------------------
    # Forward-mode input derivatives (the public jvp / jacobian surface)
    # ------------------------------------------------------------------
    #
    # The py-eager analogue of the ``forward`` / ``jvp`` / ``jacobian`` surface
    # every other route exposes (``neml2.aoti.Model`` / ``neml2::aoti::Model`` /
    # the cpp-eager ``_EagerModel``), so the same code reads identically on any
    # route. Both are thin, typed wrappers over the ``forward(v=)`` chain rule: a
    # leading-K seed is threaded per input through the forward, then the typed
    # contributions are assembled (the typed-output counterparts of the raw
    # ``neml2.types._boundary`` assembly helpers the C++-facing routes use). The
    # reverse-mode *parameter* derivatives (:meth:`param_jacobian` /
    # :meth:`param_vjp`) below are a separate surface.

    def jvp(
        self,
        inputs: Mapping[str, TensorWrapper | torch.Tensor],
        tangents: Mapping[str, TensorWrapper | torch.Tensor],
    ) -> tuple[dict[str, TensorWrapper], dict[str, TensorWrapper]]:
        """Evaluate the model and its Jacobian-vector product, all typed.

        Returns ``(outputs, jvp_outputs)``, both keyed by output name with typed
        values. *inputs* and *tangents* are keyed by input name (typed wrappers or
        raw tensors, wrapped via ``input_spec``); a missing tangent defaults to
        zero (that input contributes nothing). ``jvp_outputs[name]`` is the
        directional derivative -- a wrapper of the output's type at its natural
        ``(*batch, *sub_batch, *out_base)`` shape.

        Each supplied tangent is seeded as a single formal leading-K direction and
        threaded through the ``forward(v=)`` chain rule; the per-input
        contributions to each output are summed. The forward-mode companion of the
        reverse-mode :meth:`param_jacobian` / :meth:`param_vjp`, and the typed
        py-eager analogue of the (raw, name-keyed) ``jvp`` the compiled / embedded
        routes expose.
        """
        from ..types._boundary import (  # noqa: PLC0415
            assemble_jvp_outputs_typed,
            leading_k1_seed,
        )

        typed_args = tuple(t(inputs[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        seed: dict[str, dict[str, TensorWrapper]] = {}
        for n, t_cls in self.input_spec.items():
            tg = tangents.get(n)
            if tg is None:
                continue
            tw = tg if isinstance(tg, TensorWrapper) else t_cls(tg)  # type: ignore[call-arg]
            seed[n] = {n: leading_k1_seed(tw)}
        *typed_outs, v_out = self(*typed_args, v=seed)
        output_names = list(self.output_spec)
        outputs = dict(zip(output_names, typed_outs, strict=True))
        jvp_outputs = assemble_jvp_outputs_typed(v_out, tuple(typed_outs), output_names)
        return outputs, jvp_outputs

    def jacobian(
        self,
        inputs: Mapping[str, TensorWrapper | torch.Tensor],
    ) -> tuple[dict[str, TensorWrapper], dict[str, dict[str, Tensor]]]:
        """Evaluate the model and its full Jacobian as typed variable-pair blocks.

        Returns ``(outputs, J)`` -- ``outputs`` keyed by output name (typed) and
        ``J`` the nested ``{out_name: {in_name: block}}`` map (rows in
        ``output_spec`` order, cols in ``input_spec`` order). Each block is a
        dynamic-base :class:`~neml2.types.Tensor` of shape
        ``(*batch, *sub_batch, *out_base, *in_base)`` -- there is no single static
        wrapper for an arbitrary derivative pair. A constant ``(out, in)`` pair is
        an explicit zero block.

        Built on the same ``forward(v=)`` chain rule as :meth:`jvp`, seeding each
        input with a leading-K identity (one direction per input base component,
        reusing the AOTI export's :func:`~neml2.cli.aoti_export._leading_k_identity_seed`)
        and reshaping the result into per-pair blocks. The seed batch is
        ``broadcast(input batches, parameter batches)`` so a per-batch-element
        parameter is handled, matching ``_EagerModel.jacobian``. The typed
        py-eager analogue of the (raw, name-keyed) ``jacobian`` the compiled /
        embedded routes expose.
        """
        from ..cli.aoti_export import _leading_k_identity_seed  # noqa: PLC0415
        from ..types._boundary import assemble_jacobian_typed  # noqa: PLC0415
        from .param_ad import call_batch_shape, enumerate_typed_params  # noqa: PLC0415

        typed_args = tuple(t(inputs[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        typed_params = enumerate_typed_params(self)
        param_qnames = [q for q, _ in typed_params]
        param_base = {q: tuple(cls.BASE_SHAPE) for q, cls in typed_params}
        # The output batches on broadcast(input batches, parameter batches); build
        # every identity seed at that common call batch so the chain rule and the
        # output-batch-keyed assembly agree (a parameter may carry its own batch).
        call_batch = call_batch_shape(typed_args, self, param_qnames, param_base)
        seed = {
            n: {n: _leading_k_identity_seed(t_cls, call_batch, dtype=ta.dtype, device=ta.device)}
            for (n, t_cls), ta in zip(self.input_spec.items(), typed_args, strict=True)
        }
        *typed_outs, v_out = self(*typed_args, v=seed)
        output_names = list(self.output_spec)
        outputs = dict(zip(output_names, typed_outs, strict=True))
        jac = assemble_jacobian_typed(
            v_out,
            tuple(typed_outs),
            output_names,
            self.output_spec,
            list(self.input_spec),
            self.input_spec,
        )
        return outputs, jac

    # ------------------------------------------------------------------
    # Parameter derivatives (reverse-mode autograd over calibration params)
    # ------------------------------------------------------------------
    #
    # The py-eager surface of the cross-route ``param_jacobian`` / ``param_vjp``
    # methods (the same name-keyed contract the compiled routes expose via
    # ``neml2::aoti::Model`` / the pybind ``neml2.aoti.Model`` / the cpp-eager
    # ``_EagerModel`` adapter). Both delegate to the shared reverse-mode engine in
    # :mod:`neml2.models.param_ad`; the forward-mode input chain rule (``forward(v=)``)
    # is a separate path and is untouched.

    def param_jacobian(
        self,
        inputs: Mapping[str, TensorWrapper | torch.Tensor],
        params: list[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Evaluate the model and the dense Jacobian w.r.t. its calibration parameters.

        Returns ``(outputs, P)`` -- ``outputs`` the raw ``{out_name: tensor}`` value
        dict and ``P`` the nested ``{out_name: {param_qname: block}}`` map, each
        block ``(*batch, *out_base, *param_base)`` (a Scalar parameter contributes
        no trailing axis; a constant ``(out, param)`` pair is an explicit zero
        block). Rows in ``output_spec`` order, columns in *params* order.

        *inputs* is keyed by input name (typed wrappers or raw tensors, wrapped via
        ``input_spec``). *params* selects the parameters by their
        :meth:`~torch.nn.Module.named_parameters` qualified name; ``None`` (default)
        uses every typed parameter. Reverse-mode autograd over the parameters via
        :func:`neml2.models.param_ad.param_jacobian` -- one pass per output base
        component, independent of the parameter count; composition through an
        :class:`~neml2.models.common.ImplicitUpdate` is handled by that function's
        implicit-function-theorem adjoint.
        """
        from ..types._boundary import unwrap_outputs  # noqa: PLC0415
        from .param_ad import enumerate_typed_params  # noqa: PLC0415
        from .param_ad import param_jacobian as _param_jacobian  # noqa: PLC0415

        typed_args = tuple(t(inputs[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        typed_params = enumerate_typed_params(self)
        param_base_shapes = {q: tuple(cls.BASE_SHAPE) for q, cls in typed_params}
        param_qnames = params if params is not None else [q for q, _ in typed_params]
        output_names = list(self.output_spec)
        result = self(*typed_args)
        typed_outs = result if isinstance(result, tuple) else (result,)
        outputs = unwrap_outputs(typed_outs, output_names)
        pjac = _param_jacobian(
            self, typed_args, param_qnames, output_names, self.output_spec, param_base_shapes
        )
        return outputs, pjac

    def param_vjp(
        self,
        inputs: Mapping[str, TensorWrapper | torch.Tensor],
        cotangents: Mapping[str, TensorWrapper | torch.Tensor],
        params: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""Parameter adjoint ``dL/d\theta`` for ``L = sum_o <cotangent_o, out_o>``.

        Returns ``{param_qname: grad}`` at each parameter's natural shape (the batch
        summed out -- the global-parameter gradient). One reverse pass total,
        independent of the parameter count; the cheap form for many-parameter
        inverse-optimization gradients, and the py-eager analogue of the compiled
        ``param_vjp``. *cotangents* maps each output name to ``w_o`` at the output's
        ``(*batch, *out_base)`` shape (typed or raw). *inputs* / *params* as in
        :meth:`param_jacobian`. Composition through an
        :class:`~neml2.models.common.ImplicitUpdate` is handled by the
        implicit-function-theorem adjoint, exactly as in :meth:`param_jacobian`.
        """
        from .param_ad import enumerate_typed_params  # noqa: PLC0415
        from .param_ad import param_vjp as _param_vjp  # noqa: PLC0415

        typed_args = tuple(t(inputs[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        param_qnames = (
            params if params is not None else [q for q, _ in enumerate_typed_params(self)]
        )
        return _param_vjp(self, typed_args, param_qnames, list(self.output_spec), dict(cotangents))

    @property
    def parameter_base_shapes(self) -> dict[str, list[int]]:
        """Per-calibration-parameter natural base shape, keyed by qualified name.

        ``{param_qname: base_shape}`` (Scalar -> ``[]``, SR2 -> ``[6]``); the keys
        are every typed parameter (same keys as
        :meth:`~torch.nn.Module.named_parameters` restricted to typed parameters).
        The parameter analogue of the input/output base-shape surface and the
        unified read-only parameter-introspection accessor the compiled routes
        expose too (``neml2.aoti.Model`` / ``neml2::aoti::Model`` /
        ``neml2::eager::Model`` all expose ``parameter_base_shapes``), so a
        downstream consumer introspects the parameter surface the same way on any
        route. (The compiled routes report only the *promoted* subset; the native
        model carries every typed parameter live.)
        """
        from .param_ad import enumerate_typed_params  # noqa: PLC0415

        return {q: list(cls.BASE_SHAPE) for q, cls in enumerate_typed_params(self)}

    def set_parameter(self, name: str, value: torch.Tensor | float | int) -> None:
        """Set a calibration parameter's value (the write side of the cross-route
        parameter surface; the compiled routes expose the same ``set_parameter``
        on ``neml2::aoti::Model`` / the pybind ``Model`` / the cpp-eager runtime).

        *name* is the parameter's :meth:`~torch.nn.Module.named_parameters`
        qualified name (e.g. ``"elasticity.E"``); *value* is cast to the
        parameter's dtype / device. When *value* broadcasts into the parameter's
        current shape (the common case -- same shape, or a scalar into a batched
        parameter) it is written in place with ``torch.no_grad`` ``copy_``, so the
        ``nn.Parameter`` identity is preserved (optimizer state and the autograd
        graph are undisturbed). When *value* has an incompatible shape -- e.g.
        promoting a scalar parameter to a per-batch ``(B,)`` value, which ``copy_``
        cannot resize into -- the parameter is **replaced** with a fresh
        ``nn.Parameter`` of the new shape (preserving ``requires_grad``). Raises
        ``KeyError`` for an unregistered name.
        """
        params = dict(self.named_parameters())
        if name not in params:
            raise KeyError(
                f"set_parameter: {name!r} is not a registered parameter "
                f"(available: {sorted(params)})"
            )
        p = params[name]
        v = torch.as_tensor(value, dtype=p.dtype, device=p.device)
        # ``copy_`` broadcasts *into* p's shape but cannot resize it; fall back to
        # replacing the Parameter when the new value needs a different shape.
        try:
            broadcasts_into_p = tuple(torch.broadcast_shapes(v.shape, p.shape)) == tuple(p.shape)
        except RuntimeError:
            broadcasts_into_p = False
        if broadcasts_into_p:
            with torch.no_grad():
                p.copy_(v)
        else:
            module_path, _, pname = name.rpartition(".")
            owner = self.get_submodule(module_path) if module_path else self
            owner._parameters[pname] = nn.Parameter(
                v.detach().clone(), requires_grad=p.requires_grad
            )

    # ------------------------------------------------------------------
    # Chain-rule helpers
    # ------------------------------------------------------------------

    def apply_chain_rule(
        self,
        v: ChainRuleDict,
        output_name: str,
        actions: Mapping[str, ChainRuleAction],
        *,
        output: TensorWrapper | None = None,
    ) -> ChainRuleDict:
        """Apply local chain-rule actions and accumulate by seed leaf.

        ``actions`` maps each input variable name to a function that transforms
        an incoming sensitivity block for that input into its contribution to
        ``output_name``. Missing input/leaf sensitivities are structural zeros.

        When two actions contribute to the same seed leaf but with different
        sub-batch structure — typical of a per-crystal output that mixes a
        global input (e.g. ``d``, ``w``) and a per-crystal input (e.g. ``dp``,
        ``e``) — the accumulator pads the lower-ndim contribution with
        singleton axes at the start of its sub-batch region so the sum
        broadcasts correctly. This mirrors the C++ ``chain_rule`` helper's
        ``du_dx_f.intmd_unsqueeze(...)`` step
        (``src/neml2/tensors/functions/chain_rule.cxx``).

        When ``output`` (the leaf's forward result wrapper) is supplied, each
        accumulated contribution is retagged with ``output.sub_batch_ndim`` so
        the action body never has to declare sub-batch metadata explicitly.
        This is the foundational-op equivalent of the C++ side encoding
        sub-batch entirely in ``data.shape`` — the leaf does math, the
        accumulator owns the metadata.

        tangents are ordinary typed wrappers with K as the
        leading batch dim. Seeds that arrive as raw tensors (tests / export
        seeding) are wrapped as the input variable's type. Accumulation is plain
        typed ``+`` (``align_sub_batch`` under the hood).
        """
        # Translate option-name keys to the externally-visible (resolved)
        # names a caller's tangent dict carries. Empty when no rename.
        renames = self.__dict__.get("_var_renames", {})
        ext_output = renames.get(output_name, output_name)

        # Per-edge label-targeted promote (Phase D / D2). For each input
        # edge, look up this leaf's ``EdgeInfo`` and selectively
        # materialise only the V_in axes whose label is in
        # ``reduces | introduces``. ``reduces`` is the obvious case (the
        # leaf collapses that axis and needs the per-site directions
        # enumerated in K). ``introduces`` is included because a leaf
        # that adds a new labelled axis (e.g. ResolvedShear introducing
        # ``"slip"`` from per-grain stress) needs the input's other
        # labelled axes materialised too.
        #
        # When the edge declares non-empty labels but V_in's labels
        # don't match (legacy / anonymous-data path), fall back to the
        # all-axes :func:`promote_broadcast_tangent`. This preserves
        # the pre-Phase-D contract for HIT fixtures that haven't
        # labelled their ICs yet -- the perf win is conditional on
        # labels, but correctness is not.
        per_leaf: dict[str, list[TensorWrapper]] = {}
        for input_name, action in actions.items():
            ext_input = renames.get(input_name, input_name)
            for leaf, V_in in v.get(ext_input, {}).items():
                if isinstance(V_in, torch.Tensor):
                    V_in = self.input_spec[ext_input](V_in)
                # v2-parity: label-driven promote machinery removed. Leaves
                # that need to materialise paired-broadcast K axes (cross-mix
                # contractions) call ``fullify`` explicitly inside their
                # action; the chain rule itself ships K state through.
                contribution = action(V_in)
                if output is not None:
                    contribution = _retag_to_output(contribution, output)
                per_leaf.setdefault(leaf, []).append(contribution)

        from ..types._base import align_k  # noqa: PLC0415
        from .chain_rule import equalize_tangent_K  # noqa: PLC0415

        out: dict[str, TensorWrapper] = {}
        for leaf, contribs in per_leaf.items():
            # Align k_ndim across contributions so the per-axis K combine has
            # consistent shapes; then tile-equalize K storage sizes; then sum.
            k_aligned, _ = align_k(*contribs)
            equalized = equalize_tangent_K(list(k_aligned))
            summed = equalized[0]
            for c in equalized[1:]:
                summed = summed + c
            out[leaf] = summed
        return {ext_output: out}

    def apply_chain_rule_2(
        self,
        v: ChainRuleDict,
        v2: SecondOrderChainRuleDict,
        output_name: str,
        actions_1: Mapping[str, ChainRuleAction],
        actions_2: Mapping[tuple[str, str], SecondOrderChainRuleAction],
        vh: ChainRuleDict | None = None,
    ) -> SecondOrderChainRuleDict:
        """Propagate a second-order JVP through this leaf's local Jacobian.

        Implements $(g∘f)''[a, b] = g''(f) · (f'[a], f'[b]) + g' · f''[a, b]$:

        * The ``g''`` term iterates input pairs ``(i, j)`` and combines incoming
          first-order tangents ``v[i][a]`` (slot 1) and ``vh[j][b]`` (slot 2)
          into a two-leading-axis typed output tangent via ``actions_2[(i, j)]``.
          When ``vh`` is None it defaults to ``v`` (the original symmetric
          all-pairs behaviour). When ``vh`` is provided, only ``(v[i] × vh[j])``
          pairs are iterated — used by Normality to compute
          Hessian-applied-to-outer directly without materialising the full Hessian.
        * The ``g'`` term applies the existing first-order action to incoming
          second-order tangents ``v2[i][a][b]``. Re-uses ``actions_1`` so the
          inner-input → outer-seed contraction matches the first-order path
          exactly.

        Missing input pairs in ``actions_2`` are treated as ``f''=0``; missing
        entries in ``v`` / ``vh`` / ``v2`` are structural zeros. The resulting
        dict carries one outer key (``output_name``).
        """
        accum: dict[str, dict[str, TensorWrapper]] = {}

        v_slot_b = vh if vh is not None else v
        renames = self.__dict__.get("_var_renames", {})
        ext_output = renames.get(output_name, output_name)

        def _add(bucket: dict[str, TensorWrapper], b: str, contribution: TensorWrapper) -> None:
            bucket[b] = contribution if b not in bucket else bucket[b] + contribution

        # g'' · (f'[a], f'[b]) — Hessian action on incoming first-order tangents.
        # the framework now iterates over each
        # incoming tangent's leading seed axis (N_a, N_b indices) and hands the
        # leaf ``action_2`` a single ``(Va_one, Vb_one)`` pair shaped like the
        # primal (no leading seed dim), so the bilinear can be written in pure
        # typed-wrapper algebra. The framework restacks the K_a × K_b results
        # into a single ``(N_a, N_b, *dyn, *sub, *base)`` tangent — this is the
        # only place that materializes the seed-pair outer; leaves never see it.
        for (input_i, input_j), action_2 in actions_2.items():
            v_i = v.get(renames.get(input_i, input_i), {})
            v_j = v_slot_b.get(renames.get(input_j, input_j), {})
            for a, Va in v_i.items():
                for b, Vb in v_j.items():
                    _add(accum.setdefault(a, {}), b, _apply_action_2(action_2, Va, Vb))

        # g' · f''[a, b] — apply the linear first-order action to incoming v2.
        # Typed-wrapper algebra broadcasts over arbitrary leading seed axes, so
        # the same action used for first-order tangents also applies directly
        # to the two-leading-axis second-order wrapper.
        for input_name, action_1 in actions_1.items():
            for a, inner in v2.get(renames.get(input_name, input_name), {}).items():
                for b, V2_in in inner.items():
                    _add(accum.setdefault(a, {}), b, action_1(V2_in))

        return {ext_output: accum}

    def propagate_tangents(
        self,
        v: ChainRuleDict,
        output_name: str,
        actions_1: Mapping[str, ChainRuleAction],
        *,
        output: TensorWrapper | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        actions_2: Mapping[tuple[str, str], SecondOrderChainRuleAction] | None = None,
        vh: ChainRuleDict | None = None,
    ) -> tuple:
        """Dispatch ``v`` / ``v2`` / ``vh`` through the local chain-rule actions.

        Wraps the boilerplate every second-order-aware leaf otherwise has to
        spell out: call :meth:`apply_chain_rule` for ``v`` (always),
        :meth:`apply_chain_rule_2` for ``v2`` / ``vh`` (when requested), and
        return the right-length tuple. The return shape mirrors what the leaf
        was asked for:

        * ``v2 is None and vh is None`` → ``(v_out,)``
        * ``v2 is set, vh is None`` → ``(v_out, v2_out)``
        * ``vh is set`` (``v2`` may be ``None``, treated as ``{}``)
          → ``(v_out, v2_out, vh_out)``

        Linear leaves (LinearCombination, YieldFunction, …) call this with
        no ``actions_2`` — the second-order pass collapses to applying
        ``actions_1`` to ``v2`` entries (``g'' = 0``). Non-linear leaves
        (SR2Invariant, …) pass an explicit ``actions_2`` map.

        Usage::

            return out, *self.propagate_tangents(
                v, self._to, actions_1, output=out, v2=v2, vh=vh
            )
        """
        v_out = self.apply_chain_rule(v, output_name, actions_1, output=output)
        if v2 is None and vh is None:
            return (v_out,)
        if v2 is None:
            v2 = {}
        v2_out = self.apply_chain_rule_2(v, v2, output_name, actions_1, actions_2 or {}, vh=vh)
        if vh is None:
            return (v_out, v2_out)
        vh_out = self.apply_chain_rule(vh, output_name, actions_1, output=output)
        return (v_out, v2_out, vh_out)


def _apply_action_2(
    action_2: SecondOrderChainRuleAction,
    Va: TensorWrapper,
    Vb: TensorWrapper,
) -> TensorWrapper:
    """Evaluate the leaf's bilinear ``action_2`` over every (k_a, k_b) seed pair.

    The leaf-side ``action_2(Va_one, Vb_one)``
    receives a single tangent slice per slot — both shaped like the primal
    (no leading seed dim) — and returns a single typed-wrapper bilinear of
    the same primal shape. This helper handles the framework side: iterates
    the ``(N_a, N_b)`` seed-index outer, stitches the results back into the
    single ``(N_a, N_b, *dyn, *sub, *base)`` typed-wrapper tangent the
    chain-rule machinery expects.

    Concentrating the seed-outer here is the entire point of the refactor:
    leaves no longer reach into ``.data`` to unsqueeze + broadcast two
    differently-sized leading axes, so ``action_2`` bodies stay in pure
    typed-wrapper algebra (``2 * Va * Vb``, ``inner(Va, dev(Vb))``, …).
    """
    N_a = int(Va.data.shape[0])
    N_b = int(Vb.data.shape[0])
    Va_type, Vb_type = type(Va), type(Vb)
    sb_a, sb_b = Va.sub_batch_ndim, Vb.sub_batch_ndim
    results: list[TensorWrapper] = []
    for ka in range(N_a):
        Va_one = Va_type(Va.data[ka], sub_batch_ndim=sb_a)
        for kb in range(N_b):
            Vb_one = Vb_type(Vb.data[kb], sub_batch_ndim=sb_b)
            results.append(action_2(Va_one, Vb_one))
    if not results:  # defensive — N_a or N_b == 0
        raise ValueError("action_2 evaluated with empty seed axis")
    first = results[0]
    out_type = type(first)
    out_sb = first.sub_batch_ndim
    stacked = torch.stack([r.data for r in results], dim=0)
    stacked = stacked.reshape(N_a, N_b, *first.data.shape)
    return out_type(stacked, sub_batch_ndim=out_sb)


def _retag_to_output(contribution: TensorWrapper, output: TensorWrapper) -> TensorWrapper:
    """Normalize a first-order chain-rule contribution to the output variable's
    canonical sub-batch layout.

    The contribution is a typed wrapper of the output type with K as the leading
    batch dim. We force ``contribution.sub_batch_ndim ==
    output.sub_batch_ndim`` so the next leaf model that consumes this tangent
    as one of its actions' inputs sees a wrapper whose sub-batch structure
    matches the value's structure -- otherwise downstream leaf algebra
    (matmul, exp_map, jvp_*) would interpret some leading sub_batch dim as
    a base/batch dim and fold it incorrectly into the output, surfacing as
    cat-shape mismatches at the assembly boundary.

    Two cases:

    * ``contribution.sub_batch_ndim > output.sub_batch_ndim``: collapse the
      extra leading sub_batch axes by retagging them back into the dynamic
      batch region (pure metadata move). Used when an action computes more
      sub_batch than the output declares.
    * ``contribution.sub_batch_ndim < output.sub_batch_ndim``: prepend
      size-1 axes at the start of the sub_batch region (one ``unsqueeze``
      per missing axis) and retag with the new ndim. The new axes are
      broadcast-friendly -- subsequent leaf algebra that combines this
      tangent with per-site wrappers will broadcast them up to the right
      extent. This fixes the case where an action returns ``V`` unchanged
      (e.g. ``w_action(V) = V`` in :class:`OrientationRate`) but the
      output has higher sub_batch from a sibling broadcast like
      ``out = w - wp + twist``.
    """
    # V2P-3: labels removed; retag is now purely a sub_batch_ndim
    # adjustment with no label threading.
    if contribution.sub_batch_ndim == output.sub_batch_ndim:
        return contribution
    if contribution.sub_batch_ndim > output.sub_batch_ndim:
        return contribution.sub_batch.retag(output.sub_batch_ndim)
    # Expand: insert size-1 axes at the head of the sub_batch region.
    # Tag the new axes as ``"broadcast"`` with meta from the output's
    # declared sub_batch_shape so a later
    # :meth:`TensorWrapper.materialize` call (e.g. inside the assembly
    # pipeline at :func:`_tangent_block_to_trailing_k`) can expand the
    # size-1 axis to the output's true extent. Without the tag the
    # downstream :meth:`flatten_sub_batch_into_first_base_axis` would
    # fold the size-1 storage into row_base verbatim and the assembled
    # row would lose a factor of ``N``, surfacing as a downstream solve
    # shape mismatch.
    n_to_add = output.sub_batch_ndim - contribution.sub_batch_ndim
    new_data = contribution.data
    insert_pos = new_data.ndim - type(contribution).BASE_NDIM - contribution.sub_batch_ndim
    for _ in range(n_to_add):
        new_data = new_data.unsqueeze(insert_pos)
    new_state: tuple[SubBatchStateFlag, ...] = cast(
        "tuple[SubBatchStateFlag, ...]",
        ("broadcast",) * n_to_add + contribution.sub_batch_state,
    )
    out_sb_shape = tuple(int(s) for s in output.sub_batch_shape)
    new_meta = out_sb_shape[:n_to_add] + contribution.sub_batch_meta
    return contribution._rewrap(
        new_data,
        sub_batch_ndim=output.sub_batch_ndim,
        sub_batch_state=new_state,
        sub_batch_meta=new_meta,
    )
