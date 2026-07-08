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

"""ComposedModel — automatically assembled composition of Models.

Python equivalent of the C++ ``ComposedModel``. Child models are connected by
variable-name matching via :class:`~neml2.resolver.DependencyResolver`;
execution order is resolved automatically.

Both the value path and the Jacobian-pushforward path are exposed through a
single :meth:`forward` entry point, consistent with the leaf-model contract::

    # Pure forward.
    outputs = composed(*inputs)

    # Forward + Jacobian pushforward.
    # v[in_name][leaf_name] is a typed wrapper carrying leading seed axis K.
    # Seed only the inputs you care about; unseded inputs contribute no sensitivity.
    # Returns (*outputs, v_out) where v_out[out_name][leaf_name] is the output
    # type's leading-K typed tangent.
    *vals, v_out = composed(*inputs, v={"name": {"leaf": seed, ...}, ...})

Structural zeros (outputs with no dependency on a given leaf) are absent from
the inner dict rather than materialised as zero blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...factory import register_neml2_object
from ...schema import HitSchema, option
from ...types import TensorWrapper
from ..chain_rule import (
    ChainRuleDict,
    SecondOrderChainRuleDict,
)
from ..model import Model, PromotedParam, register_submodule
from ..resolver import DependencyResolver


def _coerce_to_input_type(
    state_val: TensorWrapper | torch.Tensor,
    type_cls: type[TensorWrapper],
) -> TensorWrapper:
    """Wrap or rewrap ``state_val`` as ``type_cls`` for child dispatch,
    preserving ``sub_batch_ndim`` across the leaf boundary.

    Three cases:

    * already an instance of ``type_cls`` — pass through unchanged (the
      typical hot path; preserves ``sub_batch_ndim`` exactly).
    * different ``TensorWrapper`` subclass (rare; only happens when a leaf
      returns a wrapper whose type differs from the next consumer's
      input_spec — e.g. an ``R2`` flowing into an op expecting raw ``data``)
      — rewrap with explicit ``sub_batch_ndim`` so the hint isn't lost.
    * raw ``torch.Tensor`` — wrap with the default ``sub_batch_ndim=0``;
      covers the top-level model.forward entry point where the caller hands
      in raw tensors.

    The ``TensorWrapper.__post_init__`` auto-unwrap would lose
    ``sub_batch_ndim`` if we naively called ``type_cls(wrapper)`` — the
    outer call's default ``sub_batch_ndim=0`` wins. The explicit
    ``sub_batch_ndim=state_val.sub_batch_ndim`` here is what keeps the hint
    alive across the rewrap.
    """
    if isinstance(state_val, type_cls):
        return state_val
    if isinstance(state_val, TensorWrapper):
        labels = getattr(state_val, "sub_batch_labels", ())  # noqa: B009
        if labels:
            return type_cls(  # type: ignore[call-arg]
                state_val.data,
                sub_batch_ndim=state_val.sub_batch_ndim,
            )
        return type_cls(state_val.data, sub_batch_ndim=state_val.sub_batch_ndim)  # type: ignore[call-arg]
    return type_cls(state_val)  # type: ignore[call-arg]


def _unwrap(val: TensorWrapper | torch.Tensor) -> torch.Tensor:
    """Strip a wrapper down to its raw tensor for the public return boundary."""
    return val.data if isinstance(val, TensorWrapper) else val


def _check_leaf_outputs_typed(model: nn.Module, out_names, outputs: tuple) -> None:
    """Refuse raw ``torch.Tensor`` outputs from a child leaf.

    Every leaf's forward must return ``TensorWrapper`` instances so the
    ``sub_batch_ndim`` hint survives across the leaf chain and downstream
    consumers (the export-side JVP/IFT wrappers, ``ComposedModel`` itself)
    can rely on a single uniform output type. A leaf returning a raw
    ``torch.Tensor`` is a bug in the leaf -- wrap the return in the
    declared output type before yielding.

    This catches the leak at the producing leaf rather than several frames
    downstream where a tuple of mixed wrappers/tensors trips a
    ``no attribute 'batch_shape'`` or similar generic error.
    """
    for name, val in zip(out_names, outputs, strict=True):
        if isinstance(val, TensorWrapper):
            continue
        raise TypeError(
            f"{type(model).__name__}.forward returned a non-TensorWrapper for "
            f"output {name!r}: got {type(val).__name__}. Every leaf forward "
            "must return TensorWrapper instances (Scalar, SR2, ...) so "
            "sub_batch_ndim is preserved across the leaf chain. Wrap the raw "
            "tensor in the leaf's declared output type before returning."
        )


def _walk_promoted_params(m: Model) -> dict[str, PromotedParam]:
    """Read ``_promoted_params`` off a Model.

    A nested ``ComposedModel`` will have already absorbed its own providers
    when it was built, so we don't recurse into it from the outer walk.
    """
    return dict(getattr(m, "_promoted_params", {}))


# v2-parity: EdgeInfo / per-label composition machinery removed. The chain
# rule no longer dispatches on labels -- it uses positional K_pairing
# metadata. The composed map is just inherited reachability now.


if TYPE_CHECKING:
    import nmhit

    from ...factory import _NativeInputFile


@register_neml2_object("ComposedModel")
class ComposedModel(Model):
    """Compose multiple Models together to form a single Model. The execution order
    of the composed models is determined by an internal dependency resolver such
    that input variables consumed by one model are guaranteed to be available
    from a previously executed model in the chain.
    """

    # Construction-only options (the composed I/O is resolved dynamically from
    # the children, so the schema declares no input/output variables). Drives
    # the syntax catalog; ``from_hit`` below owns the actual parsing (it also
    # auto-pulls promoted-parameter providers into the graph).
    hit = HitSchema(
        option("models", list, "Models being composed together"),
        option(
            "additional_outputs",
            list,
            "Extra output variables to be extracted from the composed model in addition to "
            "the ones identified through dependency resolution.",
            default=[],
        ),
    )

    def __init__(
        self,
        models: list[Model],
        additional_outputs: list[str] | None = None,
    ) -> None:
        super().__init__()

        resolver = DependencyResolver()
        for m in models:
            resolver.add_node(m)
        order = resolver.resolve()

        self.input_spec = resolver.inbound_items(order)
        self.output_spec = resolver.outbound_items(order, additional_outputs)

        self._in_names: list[str] = list(self.input_spec)
        self._out_names: list[str] = list(self.output_spec)

        self._plan: list[tuple[str, list[str], list[type[TensorWrapper]], list[str]]] = []
        self._child_supports_second_order: list[bool] = []
        used_attrs: set[str] = set()
        for i, m in enumerate(order):
            attr = register_submodule(self, m, fallback=f"_child_{i}", used=used_attrs)
            self._plan.append(
                (attr, list(m.input_spec), list(m.input_spec.values()), list(m.output_spec))
            )
            self._child_supports_second_order.append(bool(m.SUPPORTS_SECOND_ORDER))
        # Composed model supports v2 iff every child does. Instance-level
        # attribute shadows the class default so `child.SUPPORTS_SECOND_ORDER`
        # works uniformly for leaves and composed models.
        self.SUPPORTS_SECOND_ORDER = all(self._child_supports_second_order)

        # v2-parity: list_deriv propagation removed. The chain rule no longer
        # tracks per-(output, input) label composition; K_pairing on tangents
        # carries the per-axis dispatch directly.

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> ComposedModel:
        children: list[Model] = [factory.get_model(n) for n in node.param_list_str("models")]
        # Auto-pull providers for promoted-parameter wirings (mirrors C++
        # ParameterStore: a child whose parameter resolves to a [Models] entry
        # adds an input variable that the dependency resolver matches against
        # the provider's output). Walk transitively so providers-of-providers
        # are also included; deduplicate by object identity.
        ordered: list[Model] = list(children)
        seen_ids: set[int] = {id(m) for m in ordered}
        queue: list[Model] = list(ordered)
        while queue:
            m = queue.pop(0)
            for pparam in _walk_promoted_params(m).values():
                if pparam.provider is None:
                    continue
                if id(pparam.provider) in seen_ids:
                    continue
                ordered.append(pparam.provider)
                seen_ids.add(id(pparam.provider))
                queue.append(pparam.provider)

        ao_node = node.find("additional_outputs")
        additional_outputs = node.param_list_str("additional_outputs") if ao_node else None
        return cls(ordered, additional_outputs=additional_outputs)

    def forward(  # type: ignore[override]
        self,
        *inputs: TensorWrapper | torch.Tensor,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ) -> tuple:
        """Execute the resolved model chain.

        Parameters
        ----------
        *inputs
            One value per entry in ``input_spec``, in key order. Each may be
            a typed ``TensorWrapper`` (preferred — preserves ``sub_batch_ndim``
            across the boundary) or a raw ``torch.Tensor`` (gets wrapped with
            the input_spec type at entry, ``sub_batch_ndim=0``).
        v
            Optional first-order sensitivity seed
            ``{in_name: {leaf_name: leading-K typed tangent}}``.  Unseeded inputs
            contribute no sensitivity (absent from inner dict).
        v2
            Optional second-order sensitivity seed
            ``{in_name: {seed_a: {seed_b: two-leading-axis typed tangent}}}``.
            Requires ``v`` to also be provided. Only models in the chain that
            may appear inside a Normality wrap need to support v2 in their
            ``forward``.

        vh
            Optional "Hessian-direction" seed dict, same shape as ``v``. When
            provided, the leaf-level ``g''·(f'·a, f'·b)`` term iterates
            ``v[i] × vh[j]`` pairs (rather than ``v[i] × v[j]``).  Used by
            Normality to compute Hessian-times-outer-tangent directly,
            avoiding materialisation of the full ``(B, 1, n_i, n_j)`` Hessian
            inside the residual graph.

        Returns
        -------
        tuple[TensorWrapper, ...]
            When both ``v`` and ``v2`` are ``None``: one typed wrapper per
            output. Downstream consumers that need raw tensors (ImplicitUpdate
            / equation systems / call_by_name) read ``.data``.
        (*outputs, v_out)
            When only ``v`` is provided (and ``vh`` is None).
        (*outputs, v_out, v2_out)
            When ``v2`` (and/or ``vh``) is provided.
        """
        # ``state`` carries the typed wrapper instances (not raw tensors) so
        # ``sub_batch_ndim`` survives every leaf boundary fix
        # to D-044. Inputs that arrive as wrappers are stored as-is; inputs
        # that arrive raw get the default ``sub_batch_ndim=0`` via the
        # input_spec wrap below. Final outputs stay as typed wrappers at the
        # public boundary too — downstream consumers (ImplicitUpdate /
        # equation systems / call_by_name) already pattern-match
        # ``value.data if isinstance(value, TensorWrapper) else value``, and
        # the test harness (ModelUnitTest) needs the ``sub_batch_ndim`` hint
        # to compare against autograd JVPs on the leaf boundary.
        state: dict[str, TensorWrapper | torch.Tensor] = dict(
            zip(self._in_names, inputs, strict=True)
        )

        if v is None:
            if v2 is not None or vh is not None:
                raise ValueError("ComposedModel: v2/vh was provided without v")
            for attr, in_names, in_types, out_names in self._plan:
                model = getattr(self, attr)
                args = tuple(
                    _coerce_to_input_type(state[n], type_cls)
                    for n, type_cls in zip(in_names, in_types, strict=True)
                )
                result = model(*args)
                if not isinstance(result, tuple):
                    result = (result,)
                _check_leaf_outputs_typed(model, out_names, result)
                for name, val in zip(out_names, result, strict=True):
                    state[name] = val  # preserve wrapper + sub_batch_ndim
            return tuple(state[n] for n in self._out_names)

        # ------------------------------------------------------------------
        # Sensitivity propagation path
        # ------------------------------------------------------------------
        # Unseeded inputs get an empty inner dict (no leaf contribution).
        sens: ChainRuleDict = {name: v.get(name, {}) for name in self._in_names}
        sens_h: ChainRuleDict | None = (
            {name: vh.get(name, {}) for name in self._in_names} if vh is not None else None
        )
        sens2: SecondOrderChainRuleDict | None = (
            {name: v2.get(name, {}) for name in self._in_names}
            if v2 is not None
            else ({} if sens_h is not None else None)  # v2 implied empty when vh present
        )
        propagate_v2 = sens2 is not None

        for idx, (attr, in_names, in_types, out_names) in enumerate(self._plan):
            model = getattr(self, attr)
            args = tuple(
                _coerce_to_input_type(state[n], type_cls)
                for n, type_cls in zip(in_names, in_types, strict=True)
            )
            # An upstream leaf may produce an output that has no chain-rule
            # contribution (constant w.r.t. all seeded inputs) — in that case
            # the upstream's v_out omits the key entirely. Default to an empty
            # inner dict so the child still sees a well-formed seed.
            child_v = {n: sens.get(n, {}) for n in in_names}
            child_takes_v2 = propagate_v2 and self._child_supports_second_order[idx]
            if not child_takes_v2:
                if propagate_v2:
                    # Defensive: child can't carry the rank-4 v2 forward, so
                    # any v2 contribution that would pass through it is lost.
                    # Normality.__init__ guards against this, so reaching here
                    # means a user constructed a v2-using context manually
                    # around a v2-unaware leaf. Raise rather than silently
                    # producing wrong derivatives.
                    raise RuntimeError(
                        f"ComposedModel._forward: child {type(model).__name__!r} does not "
                        "support second-order chain rule (SUPPORTS_SECOND_ORDER=False) but v2/vh "
                        "is being propagated through it. Set SUPPORTS_SECOND_ORDER=True on the "
                        "child class and have its forward accept v2/vh kwargs, or wrap the chain "
                        "in a Normality that doesn't reach this leaf."
                    )
                result = model(*args, v=child_v)
            else:
                child_v2 = {n: sens2.get(n, {}) for n in in_names}  # type: ignore[union-attr]
                if sens_h is None:
                    result = model(*args, v=child_v, v2=child_v2)
                else:
                    child_vh = {n: sens_h.get(n, {}) for n in in_names}
                    result = model(*args, v=child_v, v2=child_v2, vh=child_vh)
            result_tuple: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
            # Unpack: child returns (*outputs, v_out) or (*outputs, v_out, v2_out)
            # or (*outputs, v_out, v2_out, vh_out) depending on what was threaded.
            if not propagate_v2:
                child_v_out = result_tuple[-1]
                child_outputs = result_tuple[:-1]
                sens.update(child_v_out)
            elif sens_h is None:
                child_v2_out = result_tuple[-1]
                child_v_out = result_tuple[-2]
                child_outputs = result_tuple[:-2]
                sens.update(child_v_out)
                sens2.update(child_v2_out)  # type: ignore[union-attr]
            else:
                child_vh_out = result_tuple[-1]
                child_v2_out = result_tuple[-2]
                child_v_out = result_tuple[-3]
                child_outputs = result_tuple[:-3]
                sens.update(child_v_out)
                sens2.update(child_v2_out)  # type: ignore[union-attr]
                sens_h.update(child_vh_out)
            _check_leaf_outputs_typed(model, out_names, child_outputs)
            for name, val in zip(out_names, child_outputs, strict=True):
                state[name] = val  # preserve wrapper + sub_batch_ndim

        outputs = tuple(state[n] for n in self._out_names)
        total_v_out = {n: sens[n] for n in self._out_names if n in sens}
        if not propagate_v2:
            return (*outputs, total_v_out)
        total_v2_out = {n: sens2[n] for n in self._out_names if n in sens2}  # type: ignore[union-attr]
        if sens_h is None:
            return (*outputs, total_v_out, total_v2_out)
        total_vh_out = {n: sens_h[n] for n in self._out_names if n in sens_h}
        return (*outputs, total_v_out, total_v2_out, total_vh_out)
