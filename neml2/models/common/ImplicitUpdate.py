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

"""Composable implicit update model for Python-native residuals."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch

from ...chain_rule import ChainRuleDict
from ...equation_systems import (
    AssembledMatrix,
    AxisLayout,
    ModelNonlinearSystem,
    _flatten_base,
    _flatten_sub_batch_and_base,
    _unflatten_sub_batch_and_base,
)
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, dependency
from ...solvers import Newton, RetCode
from ...types import TensorWrapper

if TYPE_CHECKING:
    import nmhit

    from ...factory import _NativeInputFile


def _var_offset(layout: AxisLayout, group_index: int, name: str) -> tuple[int, int]:
    offset = 0
    for candidate in layout.groups[group_index]:
        size = layout.var_size(candidate)
        if candidate == name:
            return offset, offset + size
        offset += size
    raise KeyError(f"Variable {name!r} is not in layout group {group_index}")


def _matrix_variable_block(
    matrix: AssembledMatrix,
    row_name: str,
    col_name: str,
) -> torch.Tensor:
    row_group_index = next(
        i for i, group in enumerate(matrix.row_layout.groups) if row_name in group
    )
    col_group_index = next(
        i for i, group in enumerate(matrix.col_layout.groups) if col_name in group
    )
    row_start, row_end = _var_offset(matrix.row_layout, row_group_index, row_name)
    col_start, col_end = _var_offset(matrix.col_layout, col_group_index, col_name)
    return matrix.tensors[row_group_index][col_group_index][
        ..., row_start:row_end, col_start:col_end
    ]


def _lookup_history_sbn(unknown_name: str, input_sbn: dict[str, int]) -> int:
    """Resolve an unknown's ``sub_batch_ndim`` from its history input.

    Mirrors the lookup in :meth:`ImplicitUpdate._wrap_outputs`: prefer the
    one-step-old ``<name>~1`` entry, fall back to ``<name>~k`` for ``k >= 2``,
    then the bare ``<name>`` if it appears as an input itself. Returns 0 when
    no history entry exists — the deterministic "nothing declared" default
    that matches every existing single-group scenario.
    """
    sbn = input_sbn.get(f"{unknown_name}~1")
    if sbn is None:
        for k in range(2, 10):
            sbn = input_sbn.get(f"{unknown_name}~{k}")
            if sbn is not None:
                break
    if sbn is None:
        sbn = input_sbn.get(unknown_name, 0)
    return sbn


def _matrix_pushforward(
    block: torch.Tensor,
    tangent: TensorWrapper,
    output_type: type[TensorWrapper],
    output_sub_batch_shape: torch.Size,
) -> TensorWrapper:
    """Apply an assembled ``du/dg`` block to a leading-K typed tangent.

    Tangents in a ``ChainRuleDict`` are always typed wrappers (D-061/D-062).
    """
    flat_tangent = _flatten_sub_batch_and_base(
        tangent.data,
        type(tangent),
        tangent.sub_batch_shape,
        len(tangent.dynamic_batch_shape),
    )
    flat_contribution = torch.matmul(block.unsqueeze(0), flat_tangent.unsqueeze(-1)).squeeze(-1)
    contribution = _unflatten_sub_batch_and_base(
        flat_contribution,
        output_type,
        output_sub_batch_shape,
        flat_contribution.ndim - 1,
    )
    return output_type(contribution, sub_batch_ndim=len(output_sub_batch_shape))


@register_native("ImplicitUpdate")
class ImplicitUpdate(Model):
    """Update an implicit model by solving the underlying nonlinear system of equations."""

    # Construction-only options (the I/O is dynamic — derived from the wrapped
    # system's unknowns/givens in __init__ — so the schema declares no
    # input/output variables). Drives the syntax catalog; ``from_hit`` below
    # owns the actual parsing (predictor needs the not-natively-registered
    # fallback warning).
    hit = HitSchema(
        dependency(
            "equation_system",
            "get_equation_system",
            "The nonlinear system of equations to solve",
        ),
        dependency(
            "solver",
            "get_solver",
            "Solver used to solve the nonlinear system of equations",
        ),
        dependency(
            "predictor",
            "get_model",
            "An optional predictor to provide an initial guess for the nonlinear solve.",
            default=None,
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> ImplicitUpdate:
        system = factory.get_equation_system(node.param_str("equation_system"))
        solver = factory.get_solver(node.param_str("solver"))
        predictor: Model | None = None
        pred_node = node.find("predictor")
        if pred_node is not None:
            pred_name = node.param_str("predictor")
            try:
                predictor = factory.get_model(pred_name)
            except KeyError:
                warnings.warn(
                    f"Predictor {pred_name!r} is not natively registered; "
                    "using no predictor (initial guess = zero).",
                    stacklevel=2,
                )
        return cls(system, solver, predictor=predictor)

    def __init__(
        self,
        system: ModelNonlinearSystem,
        solver: Newton,
        predictor: Model | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        # Register the residual model as a submodule so ``parameters()`` /
        # ``named_parameters()`` reach the leaves' calibration parameters.
        # ``ModelNonlinearSystem`` is a plain Python object (not an nn.Module),
        # so its ``.model`` attribute is otherwise invisible to the walker.
        self._residual_model = system.model
        self.solver = solver
        self.predictor = predictor
        self.last_iterations = 0
        self.last_status = RetCode.FAILURE

        predictor_outputs = set(predictor.output_spec) if predictor is not None else set()
        input_names = [
            name
            for name in system.model.input_spec
            if name in system.given_names or name not in predictor_outputs
        ]
        # The predictor may consume variables (e.g. <unknown>~1) that the
        # residual surface does not — those need to surface as ImplicitUpdate
        # inputs so the caller can supply them. C++ ImplicitUpdate does the
        # same: with a ConstantExtrapolationPredictor on the J2 return-map,
        # `flow_rate~1` appears as an input even though `surface` never
        # consumes it.
        if predictor is not None:
            for pred_in in predictor.input_spec:
                if pred_in not in input_names:
                    input_names.append(pred_in)

        all_specs = dict(system.model.input_spec)
        if predictor is not None:
            for pred_in, pred_type in predictor.input_spec.items():
                all_specs.setdefault(pred_in, pred_type)
        self.input_spec = {name: all_specs[name] for name in input_names}
        self.output_spec = {name: system.model.input_spec[name] for name in system.unknown_names}

    def _initial_unknowns(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.predictor is None:
            return {name: state[name] for name in self.system.unknown_names}

        predicted = self.predictor.call_by_name(state)
        unknowns: dict[str, torch.Tensor] = {}
        for name in self.system.unknown_names:
            if name in predicted:
                unknowns[name] = predicted[name]
            else:
                unknowns[name] = state[name]
        return unknowns

    def _solve(
        self,
        state: dict[str, torch.Tensor],
        sub_batch_ndim: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        givens = {name: state[name] for name in self.system.given_names}
        unknowns = self._initial_unknowns(state)
        # Bridge: caller's ``sub_batch_ndim`` dict is keyed by INPUT names
        # (e.g. ``concentration~1``), but the equation system needs entries
        # keyed by every state name — given names AND unknown names. Unknowns
        # are derived from history inputs by the same ``name~k`` rule
        # ``_wrap_outputs`` uses, so we mirror that here. Without this bridge
        # the system would default unknowns' sub_batch_ndim to 0 even when the
        # corresponding history declared it non-zero, silently reclassifying
        # what should be a per-site sub-batch axis as a dynamic-batch axis.
        full_sbn = dict(sub_batch_ndim) if sub_batch_ndim is not None else {}
        for uname in self.system.unknown_names:
            if uname in full_sbn:
                continue
            full_sbn[uname] = _lookup_history_sbn(uname, sub_batch_ndim or {})
        # Derive the system-wide dyn shape from the caller's state values —
        # specifically the widest leading-batch shape across all inputs after
        # stripping each one's declared sub_batch axes. This is deterministic
        # (the caller's typed-wrapper inputs are the source of truth at call
        # time) and gives the equation system the target axis layout for
        # identity-seed padding, replacing earlier state-shape heuristics
        # that broke once Newton mutated unknown shapes during iteration.
        dyn_shape: tuple[int, ...] = ()
        for name, val in state.items():
            tcls = self.system.model.input_spec.get(name)
            if tcls is None:
                continue  # predictor-only inputs not in the inner model spec
            sbn = full_sbn.get(name, 0)
            dyn_n = max(val.ndim - tcls.BASE_NDIM - sbn, 0)
            if dyn_n > len(dyn_shape):
                dyn_shape = tuple(val.shape[:dyn_n])
        self.system.initialize(u=unknowns, g=givens, sub_batch_ndim=full_sbn, dyn_shape=dyn_shape)
        result = self.solver.solve(self.system)
        self.last_iterations = result.iterations
        self.last_status = result.ret
        if result.ret is not RetCode.SUCCESS:
            raise RuntimeError(f"Nonlinear solve failed with status {result.ret.name}")
        return self.system.u().disassemble()

    def _output_sensitivities(self, v: ChainRuleDict) -> ChainRuleDict:
        A, B = self.system.A_and_B()
        du_dg = -self.solver.linear_solver.solve(A, B)
        assert isinstance(du_dg, AssembledMatrix)

        out: ChainRuleDict = {}
        for unknown_name in self.system.unknown_names:
            unknown_sens: dict[str, TensorWrapper] = {}
            unknown_type = du_dg.row_layout.type_of(unknown_name)
            unknown_sub_batch_shape = du_dg.row_layout.sub_batch_shape(unknown_name)
            for given_name in self.system.given_names:
                given_sens = v.get(given_name, {})
                if not given_sens:
                    continue
                block = _matrix_variable_block(du_dg, unknown_name, given_name)
                for leaf_name, V in given_sens.items():
                    contribution = _matrix_pushforward(
                        block, V, unknown_type, unknown_sub_batch_shape
                    )
                    if leaf_name in unknown_sens:
                        unknown_sens[leaf_name] = unknown_sens[leaf_name] + contribution
                    else:
                        unknown_sens[leaf_name] = contribution
            out[unknown_name] = unknown_sens
        return out

    def forward(  # type: ignore[override]
        self,
        *inputs,
        v: ChainRuleDict | None = None,
    ):
        # Accept either raw torch.Tensor (when called directly by user code
        # or by an ``EquationSystem``) or ``TensorWrapper`` (when called as a
        # child of ``ComposedModel``, which wraps every input to its declared
        # ``input_spec`` type before dispatch). Unwrapping here keeps the
        # internal Newton solve and IFT machinery operating on raw tensors,
        # which is what ``_solve`` and ``_ImplicitUpdateFn`` expect.
        from ...types import TensorWrapper  # noqa: PLC0415

        raw_inputs = tuple(t.data if isinstance(t, TensorWrapper) else t for t in inputs)
        # Capture each input's caller-side ``sub_batch_ndim`` (0 when the
        # caller handed us a raw tensor) so we can mirror it onto the
        # corresponding output unknowns — see ``_wrap_outputs``.
        input_sbn = {
            name: (t.sub_batch_ndim if isinstance(t, TensorWrapper) else 0)
            for name, t in zip(self.input_spec, inputs, strict=True)
        }

        if v is not None:
            # Chain-rule pushforward path — used by the AOTI export wrappers and
            # the v-aware ComposedModel composition. Pure forward, no autograd
            # graph through the Newton solve. (D-039 does not apply here; the
            # IFT is expressed directly in `_output_sensitivities`.)
            state = dict(zip(self.input_spec, raw_inputs, strict=True))
            solved = self._solve(state, sub_batch_ndim=input_sbn)
            raw_outputs = tuple(solved[name] for name in self.output_spec)
            wrapped = self._wrap_outputs(raw_outputs, input_sbn)
            return (*wrapped, self._output_sensitivities(v))

        # Eager autograd path: Newton runs under no_grad inside the
        # autograd.Function's forward; the Function's backward implements IFT,
        # re-evaluating the residual at u* under autograd so torch.autograd.grad
        # recovers `dr/dinput` and `dr/dparam` in one sweep — no per-leaf
        # parameter chain-rule actions needed.
        params = tuple(self.parameters())
        raw_outputs = _ImplicitUpdateFn.apply(
            self, len(raw_inputs), input_sbn, *raw_inputs, *params
        )
        if not isinstance(raw_outputs, tuple):
            raw_outputs = (raw_outputs,)
        return self._wrap_outputs(raw_outputs, input_sbn)

    def _wrap_outputs(
        self,
        raw_outputs: tuple[torch.Tensor, ...],
        input_sbn: dict[str, int],
    ) -> tuple[TensorWrapper, ...]:
        """Return one typed wrapper per unknown.

        Each output unknown's ``sub_batch_ndim`` is inherited from the
        caller-supplied history input (``alpha~1`` → ``alpha``). This is the
        right anchor because that's the shape the user already committed to —
        the system's internally inferred ``ulayout.sub_batch_shape`` can
        over-credit sub-batch axes when a sibling input (e.g. a 0-d $T$)
        forces ``_dynamic_batch_ndim`` to 0 and re-categorizes everyone
        else's batch axes as sub-batch.

        Returning raw ``torch.Tensor`` here would drop the hint and the
        next consumer (typically ``ComposedModel._coerce_to_input_type``)
        would re-wrap with the default ``sub_batch_ndim=0`` — broken for any
        per-bin scenario.
        """
        wrapped: list[TensorWrapper] = []
        for name, raw in zip(self.output_spec, raw_outputs, strict=True):
            type_cls = self.output_spec[name]
            wrapped.append(type_cls(raw, sub_batch_ndim=_lookup_history_sbn(name, input_sbn)))
        return tuple(wrapped)


class _ImplicitUpdateFn(torch.autograd.Function):
    """Autograd-aware Newton solve with implicit-function-theorem backward.

    forward
        Run Newton to convergence in ``no_grad`` mode (so no graph builds up
        through the iterations) and return the converged unknowns as a tuple
        in ``system.unknown_names`` order.

    backward
        At converged ``u*`` we have $r(u*, g, p) = 0$ so by the IFT
        $du*/dθ = -A^{-1} · dr/dθ$ for any parameter ``θ`` (input or model
        parameter). The adjoint vector $λ = A^{-T} · grad_u*$ is computed by
        a single linear solve, then ``torch.autograd.grad(r, [inputs, params],
        -λ)`` extracts the corresponding gradient contributions in one pass.
        No per-leaf parameter chain-rule actions are needed — autograd discovers
        ``dr/dθ`` for free.

    Notes
    -----
    * Only ``inputs`` are passed through ``ctx.save_for_backward`` — that's
      what autograd's version-tracking machinery is designed for. ``u_star``
      is stored as a plain ctx attribute (it's a Function output, not a
      tracked input). ``params`` are accessed live from
      ``owner.parameters()`` in backward — saved-tensor copies are returned
      detached, which would break the autograd path back to the live
      ``nn.Parameter`` objects that ``model.forward`` actually reads from.
    * The Function's positional signature is
      ``(owner, n_inputs, input_sbn, *inputs_and_params)``; only the
      ``*inputs_and_params`` tail participates in PyTorch's grad routing (the
      leading three are non-tensors). The returned grad tuple must therefore
      start with ``(None, None, None, ...)``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        owner: ImplicitUpdate,
        n_inputs: int,
        input_sbn: dict[str, int],
        *inputs_and_params: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs = inputs_and_params[:n_inputs]
        params = inputs_and_params[n_inputs:]

        # Newton runs detached — backward uses IFT, not iteration backprop.
        state = dict(zip(owner.input_spec, (t.detach() for t in inputs), strict=True))
        with torch.no_grad():
            solved = owner._solve(state, sub_batch_ndim=input_sbn)
        u_star = tuple(solved[name].detach() for name in owner.system.unknown_names)

        ctx.owner = owner
        ctx.n_inputs = n_inputs
        ctx.n_params = len(params)
        ctx.input_sbn = input_sbn
        ctx.u_star = u_star
        # save_for_backward gives us autograd's version-tracking for the
        # actual graph-input tensors (inputs + params). Params are saved
        # only so any in-place mutation between forward and backward raises
        # cleanly — backward uses live ``owner.parameters()`` to find the
        # autograd graph from r_flat to each leaf nn.Parameter.
        ctx.save_for_backward(*inputs_and_params)
        return u_star

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        owner: ImplicitUpdate = ctx.owner
        system = owner.system
        n_inputs: int = ctx.n_inputs

        saved = ctx.saved_tensors
        inputs = saved[:n_inputs]
        # Live params, not saved (saved copies are detached and don't share
        # identity with the nn.Parameter objects model.forward reads).
        params = tuple(owner.parameters())
        u_star = ctx.u_star

        input_names = tuple(owner.input_spec)
        unknown_names = tuple(system.unknown_names)
        given_set = set(system.given_names)

        # Re-attach grad on the given inputs so the residual graph picks them up.
        # Predictor-only inputs (not in given_names) don't appear in the
        # residual; we propagate zero gradient for them (their entry in the
        # returned grad tuple is built below).
        inputs_g = tuple(
            t.detach().clone().requires_grad_(True) if name in given_set else t.detach()
            for t, name in zip(inputs, input_names, strict=True)
        )

        # Initialize state: u = u* (detached), g = the slice of inputs_g whose
        # names match given_names. u must be detached on read — although
        # ``ctx.u_star`` was detached when stored, ``torch.autograd.Function``
        # attaches the Function's own backward grad_fn to its returned tensors
        # in place, so ``ctx.u_star[i]`` arrives in backward with
        # ``grad_fn = _ImplicitUpdateFnBackward`` and would otherwise leak the
        # outer Function back into the residual graph, causing recursion when
        # ``torch.autograd.grad`` below traverses it.
        u_dict = {name: u_star[i].detach() for i, name in enumerate(unknown_names)}
        g_dict = {
            name: inp for inp, name in zip(inputs_g, input_names, strict=True) if name in given_set
        }
        system.initialize(u=u_dict, g=g_dict, sub_batch_ndim=ctx.input_sbn)

        # Build the autograd-traced residual r(u*, g, p). enable_grad is
        # required because torch.autograd.Function.backward runs in a
        # no_grad context by default.
        with torch.enable_grad():
            _, _, b = system.assemble(need_A=False, need_B=False, need_b=True)
            assert b is not None
            r_flat = -b.tensors[0]  # b = -r per Newton sign convention

        # A = dr/du computed under no_grad — we only need its numerical
        # value, not its derivative.
        with torch.no_grad():
            A_only, _, _ = system.assemble(need_A=True, need_B=False, need_b=False)
        assert A_only is not None
        A_tensor = A_only.tensors[0][0]  # (*B, n, n)

        # Flatten grad_outputs in unknown_names order (= ulayout flat layout).
        # The default `residual = <unknown>_residual` naming preserves the
        # 1-to-1 row/column ordering between A_tensor and grad_u_flat.
        grad_u_parts = [
            _flatten_base(g, system.model.input_spec[name])
            for g, name in zip(grad_outputs, unknown_names, strict=True)
        ]
        grad_u_flat = torch.cat(grad_u_parts, dim=-1).unsqueeze(-1)  # (*B, n, 1)

        lam = torch.linalg.solve(A_tensor.transpose(-1, -2), grad_u_flat).squeeze(-1)

        # IFT adjoint: grad_θ = -(dr/dθ)^T · λ for θ ∈ {given_inputs, params}.
        differentiable: list[torch.Tensor] = [
            t for t, name in zip(inputs_g, input_names, strict=True) if name in given_set
        ]
        differentiable.extend(params)
        raw_grads = torch.autograd.grad(
            outputs=r_flat,
            inputs=differentiable,
            grad_outputs=-lam,
            allow_unused=True,
            retain_graph=False,
        )

        # Re-assemble the input-grad tuple in the original input order;
        # predictor-only inputs get zero gradients.
        grad_iter = iter(raw_grads)
        grad_inputs: list[torch.Tensor] = []
        for inp, name in zip(inputs, input_names, strict=True):
            if name in given_set:
                g = next(grad_iter)
                grad_inputs.append(g if g is not None else torch.zeros_like(inp))
            else:
                grad_inputs.append(torch.zeros_like(inp))
        grad_params = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grad_iter, params, strict=True)
        ]

        # Order matches forward: (owner, n_inputs, input_sbn, *inputs, *params).
        return (None, None, None, *grad_inputs, *grad_params)


__all__ = ["ImplicitUpdate"]
