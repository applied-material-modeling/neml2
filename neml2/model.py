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
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import nn

from .chain_rule import (
    ChainRuleAction,
    ChainRuleDict,
    ListDerivSpec,
    SecondOrderChainRuleAction,
    SecondOrderChainRuleDict,
    TangentAction,
)
from .types import TensorWrapper

if TYPE_CHECKING:
    import nmhit

    from .factory import _NativeInputFile
    from .schema import HitSchema

#: Bound to the concrete typed-wrapper class a caller expects from
#: :meth:`Model._get_param`, so the return type is narrowed at the call site.
_TW = TypeVar("_TW", bound=TensorWrapper)

__all__ = [
    "TangentAction",
    "ChainRuleDict",
    "ChainRuleAction",
    "ListDerivSpec",
    "SecondOrderChainRuleDict",
    "SecondOrderChainRuleAction",
    "Model",
    "NLParam",
]


@dataclass(frozen=True)
class NLParam:
    """Marker for a parameter resolved to a runtime input (modes 3 + 4).

    Records the input variable name added to the host's ``input_spec``, the
    parameter's position within ``forward``'s ``*nl_params`` pack, and — for
    mode 3 — the provider model + its output variable name so the parent
    :class:`~neml2.models.common.ComposedModel.ComposedModel` can auto-pull the provider
    into the dependency graph (mirroring the C++ ``_nl_params`` bookkeeping
    in ``ParameterStore.cxx::resolve_tensor_name``).

    ``tail_index`` is the zero-based slot of this parameter inside the
    ``*nl_params`` pack passed to :meth:`Model._get_param`. Promoted parameters
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

    input_spec: dict[str, type[TensorWrapper]]
    output_spec: dict[str, type[TensorWrapper]]
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

    #: Per-``(output, input)`` sub-batch sparsity declaration. Pairs absent
    #: from the dict are assumed ``"diagonal"`` (the safe default for any
    #: leaf that broadcasts naturally over sub-batch); pairs that couple
    #: across sub-batch sites (reductions, FV stencils, cross-cell ops) must
    #: be declared ``"dense"`` so the composed-model resolver can compute
    #: the end-to-end sparsity. See ``chain_rule.py`` for full semantics.
    #:
    #: Plain attribute (not ClassVar) so :class:`ComposedModel` can shadow
    #: it on instances with the resolved end-to-end map.
    list_deriv: ListDerivSpec = {}

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
        #: Parameter slot name → :class:`NLParam` for any parameter that was
        #: resolved to a runtime input (modes 3 + 4). The slot also appears in
        #: ``input_spec`` keyed by ``NLParam.input_name``; :meth:`_get_param`
        #: reads from positional inputs for these and from ``self`` otherwise.
        self._nl_params: dict[str, NLParam] = {}
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
        storable_kinds = {"option", "dependency", "parameter", "parameters"} | name_kinds

        # ``factory`` is reserved: from_hit may inject it for parameter
        # resolution, and direct Python construction may pass it too.
        factory = hit_values.pop("factory", None)

        storable: set[str] = set()
        for field in getattr(schema, "fields", ()):
            if field.kind in name_kinds or field.kind in {"parameter", "parameters"}:
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
        # Parameter declarations deferred to pass 2 so mode-3/4 promoted-input
        # writes (which mutate self.input_spec inside declare_typed_parameter)
        # don't get clobbered by the instance_input assignment at the bottom
        # of pass 1.
        deferred_params: list[Any] = []

        for field in getattr(schema, "fields", ()):
            if field.kind not in storable_kinds:
                continue
            is_name = field.kind in name_kinds
            is_parameter = field.kind == "parameter"
            is_parameters = field.kind == "parameters"

            # Look up the value under attr (HIT-driven path uses ctor_name=attr)
            # or under the option name (direct Python ctor uses the user-friendly
            # schema name). attr wins when both are present.
            value_key = (
                field.attr if field.attr is not None and field.attr in hit_values else field.name
            )
            if value_key in hit_values:
                value = hit_values[value_key]
            elif is_name:
                # Direct Python construction without HIT: fall back to the
                # schema default; if the field has none (_MISSING), use the
                # option name itself — the same canonical-name convention
                # _read_var_name applies on the HIT path.
                from .schema import _MISSING  # noqa: PLC0415

                value = field.name if field.default is _MISSING else field.default
            elif (is_parameter or is_parameters) and not field.required:
                # Direct Python construction without HIT: a parameter with a
                # schema default (e.g. literal "1.0") still needs declaring.
                value = field.default
            elif field.attr is not None and not field.required:
                value = field.default
            else:
                continue

            if is_parameter:
                # Defer to pass 2 — declare_typed_parameter may extend
                # input_spec (mode 3/4), which would race against the
                # instance_input assignment below.
                deferred_params.append((field.ctor_name, value, field))
                continue

            if is_parameters:
                # Expand each list entry into its own parameter named
                # ``<attr-or-name>_<i>``. Defer to pass 2 (same reason as
                # single parameters). Store the registered names on
                # self.<attr> so the leaf can iterate them in forward.
                if isinstance(value, str):
                    specs = value.split()
                elif isinstance(value, (list, tuple)):
                    specs = list(value)
                else:
                    raise TypeError(
                        f"parameters({field.name!r}): expected list/tuple/string, "
                        f"got {type(value).__name__}"
                    )
                base = field.attr or field.name
                names = [f"{base}_{i}" for i in range(len(specs))]
                for name, spec in zip(names, specs, strict=True):
                    deferred_params.append((name, spec, field))
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
                else:
                    instance_output.pop(field.name, None)
            elif field.kind == "derived_output":
                if isinstance(value, str):
                    instance_output[value] = field.value_type

        if instance_input is not None:
            self.input_spec = instance_input
        if instance_output is not None:
            self.output_spec = instance_output

        # Re-key class-level list_deriv into instance list_deriv when any
        # variable was renamed. (The class-level dict itself stays untouched.)
        class_ld: ListDerivSpec | None = type(self).__dict__.get("list_deriv")
        if (
            rename_map
            and class_ld
            and any(o in rename_map or i in rename_map for (o, i) in class_ld)
        ):
            self.list_deriv = {
                (rename_map.get(o, o), rename_map.get(i, i)): flag
                for (o, i), flag in class_ld.items()
            }

        # Stash the rename map so apply_chain_rule(_2) can translate the
        # leaf's option-name keys to the resolved external names a caller's
        # tangent dict actually carries. Skip when no rename happened.
        if rename_map:
            self._var_renames = rename_map

        # Pass 2: declare parameters. Runs after instance_input/output were
        # applied, so mode-3/4 promoted-input writes inside
        # declare_typed_parameter stick.
        for name, value, field in deferred_params:
            self.declare_typed_parameter(
                name,
                value,
                field.value_type,
                factory=factory,
                allow_nonlinear=field.allow_nonlinear,
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

    def __call__(self, *args, **kwargs):
        # Open a forward window so the AOTI-incompatible-function guard
        # (see _guard.py / DECISION.md D-060) is armed for the whole forward
        # subtree. Re-entrant: nested child Model calls just bump the depth.
        from ._guard import _enter_forward, _exit_forward

        _enter_forward()
        try:
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
        allow_nonlinear: bool = False,
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
           c. If *allow_nonlinear*: parse the string as a variable specifier
              (``model_name`` / ``model_name.var`` / ``var``). If it matches a
              ``[Models]`` entry, pull the provider and record the input
              promotion + provider in ``_nl_params`` (mode 3). Otherwise treat
              the string as a bare variable name and add the input without a
              provider (mode 4).

        The host model's ``input_spec`` is extended in modes 3 + 4 with an
        entry keyed by the chosen input variable name (the provider's output
        name in mode 3, or the bare variable name in mode 4), appended after
        the fixed structural inputs. Inside ``forward`` — declared as
        ``forward(self, <structural inputs...>, *nl_params)`` — fetch the value
        with :meth:`_get_param`, which resolves a static slot from ``self`` or
        a promoted slot from the ``*nl_params`` pack uniformly.
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
        if not allow_nonlinear:
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
            # Promote class-level input_spec to an instance copy on first nl_param.
            self.input_spec = dict(self.input_spec)
        if input_name in self.input_spec and self.input_spec[input_name] is not type_cls:
            raise ValueError(
                f"declare_typed_parameter({name!r}): would add input {input_name!r} of type "
                f"{type_cls.__name__} but an input of the same name already exists with type "
                f"{self.input_spec[input_name].__name__}."
            )
        self.input_spec[input_name] = type_cls
        self._nl_params[name] = NLParam(
            input_name=input_name,
            tail_index=len(self._nl_params),
            provider=provider,
            provider_output=provider_output,
        )

    def _get_param(self, name: str, nl_params: Sequence[TensorWrapper], type_cls: type[_TW]) -> _TW:
        """Return the value of parameter *name* — static attribute or nl input.

        Models with parameters that may be either static or nonlinear should
        call this from ``forward`` instead of reading ``self.<name>`` directly.
        ``nl_params`` is the ``*nl_params`` pack captured by ``forward`` after
        its fixed structural inputs; a promoted parameter is read positionally
        from it via its precomputed :attr:`NLParam.tail_index`, and a static
        parameter is read from ``self``.

        *type_cls* is the concrete typed-wrapper class the caller expects (the
        same class passed to :meth:`declare_typed_parameter`). It only narrows
        the static return type — both runtime branches already yield that
        concrete type: static slots are re-wrapped by :meth:`__getattr__` from
        ``_typed_storage_classes``, and promoted slots are wrapped to
        ``input_spec``'s type by the caller before ``forward``.
        """
        nlp = self._nl_params.get(name)
        value = getattr(self, name) if nlp is None else nl_params[nlp.tail_index]
        assert isinstance(value, type_cls), (
            f"_get_param({name!r}) resolved to {type(value).__name__}, expected {type_cls.__name__}"
        )
        return value

    def _get_param_list(
        self, attr: str, nl_params: Sequence[TensorWrapper], type_cls: type[_TW]
    ) -> list[_TW]:
        """Return values of every parameter registered via :func:`parameters`.

        The :func:`parameters` schema field registers entries as
        ``<attr>_0``, ``<attr>_1``, ... and stores the name list on
        ``self.<attr>``; this helper reads each through :meth:`_get_param`
        so individual entries that were promoted to nl inputs (mode 3/4)
        come from ``nl_params`` while static entries come from ``self``.
        """
        names: list[str] = getattr(self, attr)
        return [self._get_param(n, nl_params, type_cls) for n in names]

    def __getattr__(self, name: str):
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

    def call_by_name(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Call ``forward()`` (pure, no ``v``) with tensors keyed by variable name."""
        args = tuple(t(state[n]) for n, t in self.input_spec.items())  # type: ignore[call-arg]
        result = self(*args)
        if not isinstance(result, tuple):
            result = (result,)
        return {
            n: (v.data if isinstance(v, TensorWrapper) else v)
            for n, v in zip(self.output_spec, result, strict=True)
        }

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
        out: dict[str, TensorWrapper] = {}
        for input_name, action in actions.items():
            ext_input = renames.get(input_name, input_name)
            for leaf, V_in in v.get(ext_input, {}).items():
                if isinstance(V_in, torch.Tensor):
                    V_in = self.input_spec[ext_input](V_in)
                contribution = action(V_in)
                if output is not None:
                    contribution = _retag_to_output(contribution, output)
                out[leaf] = contribution if leaf not in out else out[leaf] + contribution
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
    batch dim. Unlike the retired trailing-K tangent path, we must not
    infer sub-batch rank from the number of batch axes: the leading K axis makes
    that ambiguous for B>1. Leaf typed algebra is responsible for preserving or
    explicitly adding sub-batch axes; lower-rank contributions broadcast later
    during accumulation or matrix assembly.
    """
    if contribution.sub_batch_ndim <= output.sub_batch_ndim:
        return contribution
    return contribution.with_sub_batch(output.sub_batch_ndim)
