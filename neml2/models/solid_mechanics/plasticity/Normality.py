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

"""Python-native mirror of the C++ ``Normality`` model.

The generic :class:`Normality` operator uses the second-order
chain-rule infrastructure to compute ∂(inner.f)/∂(inner.from[i]) at runtime.
"""

from __future__ import annotations

from math import prod

import torch

from ....factory import register_neml2_object
from ....schema import HitSchema, dependency, option
from ....types import TensorWrapper
from ...chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ...model import Model, register_submodule


def _read_list_str(node, name):  # noqa: ANN001, ANN202
    return list(node.param_list_str(name))


def _base_size(type_cls: type[TensorWrapper]) -> int:
    """Flat Mandel-packed size of a TensorWrapper type."""
    shape = type_cls.BASE_SHAPE
    return prod(shape) if shape else 1


def _identity_seed(
    type_cls: type[TensorWrapper], batch_shape: torch.Size, ref: torch.Tensor
) -> TensorWrapper:
    """Leading-K identity seed for an input of given type.

    Returns a ``type_cls`` wrapper with data ``(n, *batch, *BASE_SHAPE)`` — the
    ``n = base_size`` seed directions index the base components of the variable
    (the Hessian-row / ``__id`` direction), carried as the leftmost K axis
    with ``k_ndim=1``, ``k_state=("full",)``. Without the K metadata the
    leading axis is misclassified as batch and downstream typed-wrapper
    binary ops corrupt the shape (the chain rule's align_k expects the
    leading axis to be K — same contract as user-provided outer seeds).
    """
    n = _base_size(type_cls)
    eye = torch.eye(n, dtype=ref.dtype, device=ref.device)  # (n, n)
    eye = eye.reshape(n, *([1] * len(batch_shape)), *type_cls.BASE_SHAPE)
    data = eye.expand(n, *batch_shape, *type_cls.BASE_SHAPE).contiguous()
    return type_cls(
        data,
        k_ndim=1,
        k_state=("full",),
        k_pairing=(None,),
    )


def _reshape_first_deriv(
    t: torch.Tensor | TensorWrapper, to_type: type[TensorWrapper]
) -> TensorWrapper:
    """Leading-K first-derivative tangent ``(n_from, *B)`` → ``to_type(*B, *BASE_SHAPE)``.

    ``t`` is a ``Scalar`` tangent of the (scalar) inner function with the
    ``n_from`` ``__id`` directions as the leftmost axis; moving that axis to the
    back and unflattening it into ``to_type``'s base recovers the derivative
    ``∂function/∂from`` as a typed wrapper. ``sub_batch_ndim`` is carried
    through when *t* is itself a wrapper, so Normality's outputs preserve the
    per-site axis hint across the leaf boundary (which ``ComposedModel`` and
    the export-side wrappers rely on -- raw-tensor returns from leaves are
    rejected by :func:`~neml2.models.common.ComposedModel._check_leaf_outputs_typed`).
    """
    if isinstance(t, TensorWrapper):
        raw = t.data
        sbn = t.sub_batch_ndim
    else:
        raw = t
        sbn = 0
    moved = raw.movedim(0, -1)  # (*B, n_from)
    base = to_type.BASE_SHAPE
    data = moved.squeeze(-1) if base == () else moved.reshape(*moved.shape[:-1], *base)
    return to_type(data, sub_batch_ndim=sbn)


def _collect_outer_seed_k(
    v: ChainRuleDict,
) -> dict[str, tuple[int, tuple, tuple]]:
    """Index the outer V dict by seed name → (k_ndim, k_state, k_pairing).

    Multiple inputs share the same seed_name (they were seeded with the same
    outer K). Pick one representative per seed.
    """
    seen: dict[str, tuple[int, tuple, tuple]] = {}
    for tans in v.values():
        for seed_name, tan in tans.items():
            if seed_name in seen:
                continue
            if not hasattr(tan, "k_ndim"):
                continue
            seen[seed_name] = (tan.k_ndim, tuple(tan.k_state), tuple(tan.k_pairing))
    return seen


def _wrap_with_outer_k(
    to_type: type[TensorWrapper],
    data: torch.Tensor,
    sub_batch_ndim: int,
    k_meta: tuple[int, tuple, tuple] | None,
) -> TensorWrapper:
    """Wrap ``data`` as ``to_type`` and re-attach outer-seed K metadata.

    ``data`` has shape ``(N_outer_flat, *batch, *base)`` where ``N_outer_flat``
    is the leading axis carrying the outer seed K (single axis, regardless of
    whether outer K was originally multi-axis — the second-order chain rule
    machinery treats Vb's K as a single leading dim per pair).

    When ``k_meta`` is provided and ``k_ndim > 1``, we unflatten the leading
    axis back into the outer K shape so downstream binary ops align K
    correctly. When ``k_ndim == 1`` or ``k_meta`` is missing, fall back to a
    single full K axis.
    """
    if k_meta is None:
        return to_type(data, sub_batch_ndim=sub_batch_ndim)
    k_ndim, k_state, k_pairing = k_meta
    if k_ndim <= 1:
        return to_type(
            data,
            sub_batch_ndim=sub_batch_ndim,
            k_ndim=1,
            k_state=("full",),
            k_pairing=(None,),
        )
    # Multi-axis outer K: leading flat dim → unflatten into the outer K's
    # per-axis storage shape. Each "broadcast" axis stores size 1; each
    # "full" axis stores the corresponding base/full extent.
    # The flat size is the product of storage extents (size-1 for broadcast).
    leading = data.shape[0]
    # Inspect the original outer V (one rep) to get the K storage shape: we
    # don't have direct access here, so reconstruct from k_state — broadcast=1,
    # full=leading_per_full. With exactly one "full" axis (the common case),
    # leading is fully on that axis.
    full_count = sum(1 for s in k_state if s == "full")
    if full_count == 1:
        # Each broadcast axis is size 1; one full axis carries `leading`.
        k_storage = tuple(1 if s == "broadcast" else leading for s in k_state)
    else:
        # Multiple full axes: caller would need original storage to split. For
        # now bail to single-axis K (correct for the cases we test).
        return to_type(
            data,
            sub_batch_ndim=sub_batch_ndim,
            k_ndim=1,
            k_state=("full",),
            k_pairing=(None,),
        )
    new_data = data.reshape(*k_storage, *data.shape[1:])
    return to_type(
        new_data,
        sub_batch_ndim=sub_batch_ndim,
        k_ndim=k_ndim,
        k_state=k_state,
        k_pairing=k_pairing,
    )


def _check_inner_supports_second_order(inner: Model) -> None:
    """Walk *inner*'s leaf chain and raise if any leaf has SUPPORTS_SECOND_ORDER=False.

    For a leaf, checks the class flag directly. For a ComposedModel, walks its
    children recursively (and reports the *deepest* offending leaf name so the
    error message points at the actual user-facing class, not an intermediate
    wrapper).
    """
    from ...common import ComposedModel  # avoid import cycle at module load

    def offending(m: Model) -> type[Model] | None:
        if isinstance(m, ComposedModel):
            for attr, *_ in m._plan:
                child_offender = offending(getattr(m, attr))
                if child_offender is not None:
                    return child_offender
            return None
        return None if m.SUPPORTS_SECOND_ORDER else type(m)

    offender = offending(inner)
    if offender is not None:
        raise TypeError(
            f"Normality: inner chain contains leaf {offender.__name__!r} which does not "
            "support second-order chain rule (SUPPORTS_SECOND_ORDER=False). Either set "
            "SUPPORTS_SECOND_ORDER=True on the leaf class and have its forward accept "
            "v2/vh kwargs (forwarding them to apply_chain_rule_2), or wrap a different "
            "inner model in Normality."
        )


@register_neml2_object("Normality")
class Normality(Model):
    r"""Store the first derivatives of a scalar-valued function in given variables,
    i.e. $u_i = \dfrac{f(\boldsymbol{v})}{v_i}$.
    """

    # ``from`` / ``to`` are variable-name lists, but their tensor *types* come
    # from the inner model (resolved in ``__init__``), so they are plain list
    # options rather than ``var_inputs`` (which need a static type).
    hit = HitSchema(
        dependency("model", "get_model", "The model which evaluates the scalar-valued function"),
        option("function", str, "Function to take derivative"),
        option(
            "from",
            list,
            "Function arguments to take derivatives with respect to",
            reader=_read_list_str,
            attr="from_",
        ),
        option("to", list, "Variables to store the first derivatives", reader=_read_list_str),
    )

    def __init__(self, model: Model, function: str, from_: list[str], to: list[str]) -> None:
        super().__init__()
        if len(from_) != len(to):
            raise ValueError(
                f"Normality: from and to must have the same length (got {len(from_)} vs {len(to)})"
            )
        if function not in model.output_spec:
            raise KeyError(
                f"Normality: inner model has no output named {function!r}; "
                f"available: {list(model.output_spec)}"
            )
        for name in from_:
            if name not in model.input_spec:
                raise KeyError(
                    f"Normality: inner model has no input named {name!r}; "
                    f"available: {list(model.input_spec)}"
                )
        # Guard: every leaf in the inner chain must support second-order chain
        # rule, otherwise the Hessian propagation through the chain produces
        # silently wrong derivatives. Catch this at construction so the user
        # gets a clear message naming the offending leaf class.
        _check_inner_supports_second_order(model)
        # Register the inner model under its HIT block name (stashed on the
        # object by the factory) instead of a fixed ``_inner`` slot, so
        # ``self.named_parameters()`` returns readable names like
        # ``yld.sy`` rather than ``_inner.sy``. Falls back to ``_inner`` when
        # no HIT name is available (direct Python construction) or when the
        # HIT name would collide with an existing attribute.
        self._inner_attr = register_submodule(self, model, fallback="_inner")
        self._function = function
        self._from = list(from_)
        self._to = list(to)
        # Surface the full inner input list so the outer ComposedModel can
        # route every variable the inner needs (not just those in `from`).
        self.input_spec = dict(model.input_spec)
        self.output_spec = {to[i]: model.input_spec[from_[i]] for i in range(len(from_))}

    @property
    def _inner(self) -> Model:
        """The wrapped inner model, registered under its HIT block name.

        Raises AttributeError (not KeyError) when ``_inner_attr`` hasn't been
        set yet — during ``__init__`` itself, ``add_module``'s
        ``hasattr(self, name)`` probe walks this descriptor while we're in
        the middle of wiring the slot up, and only AttributeError flows
        cleanly through ``hasattr``.
        """
        attr = self.__dict__.get("_inner_attr")
        if attr is None or attr not in self._modules:
            # The second branch covers add_module's hasattr(self, name) probe
            # which fires *between* setting _inner_attr and writing the slot
            # into _modules.
            raise AttributeError("_inner accessed before Normality.__init__ wired the slot")
        inner = self._modules[attr]
        assert isinstance(inner, Model)
        return inner

    def forward(  # type: ignore[override]
        self,
        *inputs: torch.Tensor,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
    ):
        if v2 is not None:
            # Third-order derivatives of the inner function would be required;
            # no current chain composes Normality inside another Normality.
            raise NotImplementedError(
                "Normality does not propagate v2 (would need ∂³(inner.function))."
            )

        in_names = list(self._inner.input_spec)
        in_types = list(self._inner.input_spec.values())
        if len(inputs) != len(in_names):
            raise ValueError(f"Normality expected {len(in_names)} inputs, got {len(inputs)}")

        # Accept either raw tensors or typed wrappers (ComposedModel passes the
        # latter); the inner model takes typed wrappers in its forward signature.
        raw_inputs: list[torch.Tensor] = [
            t.data if isinstance(t, TensorWrapper) else t for t in inputs
        ]
        wrapped: list[TensorWrapper] = [
            t_cls(t) for t_cls, t in zip(in_types, raw_inputs, strict=True)
        ]
        ref = raw_inputs[0]
        first_type = in_types[0]
        base_ndim = len(first_type.BASE_SHAPE)
        batch_shape = ref.shape[: ref.dim() - base_ndim]

        # Seed v on `from` inputs with identity matrices ("id" direction = the
        # output Hessian row index).
        seed_v: ChainRuleDict = {}
        for name in self._from:
            t_cls = self._inner.input_spec[name]
            seed_v[name] = {f"{name}__id": _identity_seed(t_cls, batch_shape, ref)}

        # ComposedModel.forward takes raw tensors; leaves take typed wrappers.
        from ...common import ComposedModel  # avoid import cycle at module load

        inner_args: tuple
        if isinstance(self._inner, ComposedModel):
            inner_args = tuple(raw_inputs)
        else:
            inner_args = tuple(wrapped)

        if v is None:
            # Pure forward: only need first derivatives of inner.function — no
            # Hessian work at all, no v2/vh propagation.
            inner_result = self._inner(*inner_args, v=seed_v)
            inner_v_out = inner_result[-1]
            f_v = inner_v_out.get(self._function, {})
            outputs: list[TensorWrapper] = []
            for i, from_name in enumerate(self._from):
                to_type = self.output_spec[self._to[i]]
                tangent = f_v.get(f"{from_name}__id")
                if tangent is None:
                    n = _base_size(to_type)
                    zero = torch.zeros(n, *batch_shape, dtype=ref.dtype, device=ref.device)
                    outputs.append(_reshape_first_deriv(zero, to_type))
                else:
                    outputs.append(_reshape_first_deriv(tangent, to_type))
            del wrapped
            return tuple(outputs) if len(outputs) > 1 else outputs[0]

        # First-order chain-rule path: outer needs ∂(Normality.outputs)/∂outer.
        # That is H_inner applied to the incoming outer tangents. Use vh to
        # pass the outer tangents into the second slot of inner's Hessian
        # actions, so inner directly produces (B, n_from[i], N_outer) blocks
        # WITHOUT materialising the full (B, n_from[i], n_from[j]) Hessian.
        # Passing `vh` alone implicitly enables v2 propagation inside inner
        # (which is essential for linear leaves like YieldFunction to scale
        # the upstream Hessian-applied-V via their g'·f'' term).
        # Why every inner input (not just ``self._from``): the cross Hessian
        # ∂²f/∂(from_i)∂(inner_input_j) needs an outer tangent on slot j for
        # *every* j the inner consumes. Mixed-control's kinematic backstress
        # X enters the inner via ``overstress = mandel - X`` but is not in
        # ``self._from``; without widening here the chain through X would
        # silently drop, ruining Jacobian quality and Newton convergence.
        # ``self._from`` entries are added unconditionally (preserves
        # existing inner-chain semantics for tangent-checks that seed only
        # one input). Non-from inputs are only added when they actually
        # carry an outer tangent — adding empty entries for inputs the outer
        # didn't seed can perturb inner-leaf accounting.
        seed_vh: ChainRuleDict = {name: dict(v.get(name, {})) for name in self._from}
        for name in self._inner.input_spec:
            if name in seed_vh:
                continue
            outer = v.get(name, {})
            if outer:
                seed_vh[name] = dict(outer)

        inner_result = self._inner(*inner_args, v=seed_v, vh=seed_vh)
        # ComposedModel with v + vh returns (*vals, v_out, v2_out, vh_out).
        inner_v2_out = inner_result[-2]
        inner_v_out = inner_result[-3]
        f_v = inner_v_out.get(self._function, {})
        f_v2 = inner_v2_out.get(self._function, {})

        # Outputs: ∂f/∂from[i] from the identity-seeded v.
        outputs: list[TensorWrapper] = []
        for i, from_name in enumerate(self._from):
            to_type = self.output_spec[self._to[i]]
            tangent = f_v.get(f"{from_name}__id")
            if tangent is None:
                n = _base_size(to_type)
                zero = torch.zeros(n, *batch_shape, dtype=ref.dtype, device=ref.device)
                outputs.append(_reshape_first_deriv(zero, to_type))
            else:
                outputs.append(_reshape_first_deriv(tangent, to_type))
        del wrapped

        # Normality's v_out: for each output to[i] and each outer seed (carried
        # in v[from[j]]), read f_v2["{from[i]}__id"][outer_seed] — a Scalar
        # second-order tangent with leading axes (n_from[i], N_outer, *batch).
        # The leading n_from[i] is the __id direction (= base index of to[i]);
        # moving it to the back and unflattening into to[i]'s base yields a
        # leading-K (= N_outer) tangent of to[i] directly, no matmul.
        #
        # The leading N_outer axis carries the OUTER seed's K. We re-attach the
        # outer K metadata (k_ndim, k_state, k_pairing) by looking up any v
        # entry that carries this seed_name — they share the same K layout by
        # construction. Without this, the rewrapped contribution has k_ndim=0
        # and the K-axis leaks into batch, corrupting downstream shape alignment.
        v_out: ChainRuleDict = {}
        outer_k_meta = _collect_outer_seed_k(v)
        for i, to_name in enumerate(self._to):
            from_i = self._from[i]
            to_type = self.output_spec[to_name]
            base = to_type.BASE_SHAPE
            row = f_v2.get(f"{from_i}__id", {})
            inner_block: dict[str, TensorWrapper] = {}
            for outer_seed_name, block in row.items():
                moved = block.data.movedim(0, -1)  # (N_outer, *batch, n_from[i])
                data = moved.squeeze(-1) if base == () else moved.reshape(*moved.shape[:-1], *base)
                k_meta = outer_k_meta.get(outer_seed_name)
                contribution = _wrap_with_outer_k(to_type, data, block.sub_batch_ndim, k_meta)
                if outer_seed_name in inner_block:
                    inner_block[outer_seed_name] = inner_block[outer_seed_name] + contribution
                else:
                    inner_block[outer_seed_name] = contribution
            v_out[to_name] = inner_block

        if len(outputs) == 1:
            return outputs[0], v_out
        return (*outputs, v_out)
