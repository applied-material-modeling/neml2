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

"""AOTI export wrappers for the implicit-segment Newton path.

The linear solve is **un-baked** from the operators (schema v10): each graph
either ASSEMBLES an operator or SOLVES; the C++ runtime chains operator -> solve.
This lets the same residual Jacobian feed a direct solve today or a matrix-free
iterative solver later, and keeps a single solver implementation (the Python
``[Solvers]`` classes, run live for eager and compiled for AOTI).

Operator graphs (assemble; no solve):

- :class:`RHS`          -- ``(*u,*g,*params) -> (*b_groups)`` (residual eval,
                           cheap; every line-search trial).
- :class:`Jacobian`     -- ``(*u,*g,*params) -> (*A_blocks, *b_groups)``
                           (``A = ∂r/∂u`` row-major (residual × unknown) grid +
                           ``b = -r``; the Newton-step operator).
- :class:`JacobianGiven`-- ``(*u,*g,*params) -> (*B_blocks)`` (``B = ∂r/∂g``;
                           the IFT's given-side operator, paired with
                           ``Jacobian``'s ``A``).
- :class:`DrDParam`     -- ``(*u,*g,*params_per_batch) -> (A_dense, ∂r/∂θ)``
                           (dense operators for the parameter sensitivity, via
                           reverse-mode ``torch.autograd.grad`` -- the only AD
                           that lowers through AOTInductor; strict + per-batch
                           parameter, the runtime broadcasts the stored scalar).

Solve graphs (consume operators; no assembly):

- :class:`LinearSolve`     -- ``(*A_blocks, *b_groups) -> (*du_groups)`` (Newton
                              step ``du = A^{-1} b`` via the configured solver).
- :class:`LinearSolveIFT`  -- ``(*A_blocks, *B_blocks) -> *blocks`` (IFT
                              ``du/dg = -A^{-1} B``, one block per ``(unknown,
                              given)`` pair via ``AssembledMatrix.disassemble``;
                              the C++ runtime composes each against ``dg_dmaster``
                              with the same per-pair path a forward segment uses).
- :class:`LinearSolveParam`-- ``(A_dense, ∂r/∂θ) -> *blocks`` (parameter
                              sensitivity ``du/dθ = -A^{-1} ∂r/∂θ``, one dense
                              block per ``(unknown, param)`` pair).

The promoted-parameter tail (``*params``) is empty in the common case (no
``--parameter`` targeting an attribute inside the implicit region); when
present it lists, in graph-call order, the promoted parameters that live
inside the implicit segment's residual model. After
:func:`~neml2.cli.aoti_export._promote_parameters` these appear in
``system.model.input_spec`` but are neither unknowns nor givens, so the
operator wrappers inject them into the per-variable state from the trailing
forward args. The operator graphs take them as the stored scalar (constant
across the solve and the input-Jacobian); ``DrDParam`` takes them per-batch.

The operator graphs take/return per-group raw tensors at the natural
``AssembledVector`` / ``AssembledMatrix`` group shape -- BLOCK groups preserve
their ``sub_batch_shape`` axes, DENSE groups have sub_batch folded into the last
base axis. The solve graphs reconstruct the typed operands from those raw blocks
(:func:`~neml2.es.assembled.group_block_sub_batch_ndim` /
:func:`~neml2.es.assembled.wrap_block_raw`). The C++ runtime maintains
per-variable ``dstate`` for downstream forward composition; the per-variable ↔
per-group conversion happens twice per solve (once at solve start to pack
``u_groups`` / ``g_groups``, once at solve end to unpack converged
``u_groups`` back to ``dstate``). The Newton inner loop is fully per-group.
"""

from __future__ import annotations

from math import prod

import torch
from torch import nn

from neml2.models.chain_rule import ChainRuleDict
from neml2.types import TensorWrapper

from ._helpers import _flatten_base, build_identity_seed
from .assembled import (
    AssembledMatrix,
    AssembledVector,
    _build_block_matrix,
    group_block_sub_batch_ndim,
    wrap_block_raw,
    wrap_group_raw,
)
from .axis_layout import AxisLayout
from .system import ModelNonlinearSystem


def enumerate_group_var_names(layout: AxisLayout) -> tuple[tuple[str, ...], ...]:
    """Canonical per-group variable-name iteration order for *layout*.

    Single source of truth for the order in which group tensors are
    enumerated in segment forward signatures and in metadata
    emission. See :func:`~neml2.cli.aoti_export._enumerate_group_infos`
    for the matching emitter side.
    """
    return tuple(tuple(g) for g in layout.groups)


class _SystemModule(nn.Module):
    """Tensor-only export surface for a frozen :class:`ModelNonlinearSystem`.

    The operator subclasses (:class:`RHS`, :class:`Jacobian`,
    :class:`JacobianGiven`, :class:`DrDParam`) take per-group raw tensors at the
    graph signature -- ``forward(*u_groups, *g_groups, *params)`` where the leading
    ``len(unknown_groups)`` positional args are the unknown groups (in
    ``ulayout.groups`` order), the next ``len(given_groups)`` are the given
    groups (in ``glayout.groups`` order), and the trailing ``len(param_names)``
    are the promoted parameters that live inside the implicit residual (in
    ``param_names`` order; empty in the common no-promotion case).
    """

    def __init__(self, system: ModelNonlinearSystem, param_names: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.model = system.model
        self.ulayout = system.ulayout
        self.glayout = system.glayout
        self.blayout = system.blayout
        self.unknown_names = tuple(system.unknown_names)
        self.given_names = tuple(system.given_names)
        self.residual_names = tuple(system.residual_names)
        # Per-group variable names (canonical iteration order for the
        # per-group tensors in forward args / returns).
        self.unknown_groups: tuple[tuple[str, ...], ...] = enumerate_group_var_names(self.ulayout)
        self.given_groups: tuple[tuple[str, ...], ...] = enumerate_group_var_names(self.glayout)
        self.residual_groups: tuple[tuple[str, ...], ...] = enumerate_group_var_names(self.blayout)
        self.input_names = tuple(system.model.input_spec)
        self.output_names = tuple(system.model.output_spec)
        self.dyn_ndim: dict[str, int] = dict(system._dynamic_batch_ndim)
        self.sub_batch_shapes = dict(system._sub_batch_shapes)
        # Promoted parameters threaded as a positional tail after the givens.
        # After ``_promote_parameters`` these are in ``model.input_spec`` but
        # are neither unknowns nor givens; ``_state_from_per_group_args``
        # injects them into the per-variable state from the trailing args so
        # ``_call_model_from_state`` (which iterates the full input_spec) finds
        # them. ``param_types`` is looked up from the (post-promotion) spec.
        self.param_names = tuple(param_names)
        self.param_types: tuple[type[TensorWrapper], ...] = tuple(
            system.model.input_spec[p] for p in self.param_names
        )

    def _state_from_per_group_args(
        self, args: tuple[torch.Tensor, ...]
    ) -> dict[str, TensorWrapper]:
        """Split positional per-group inputs into a per-variable typed state.

        Inputs:
            ``args`` -- ``(*u_groups, *g_groups, *params)`` raw tensors. The
            first ``len(self.unknown_groups)`` are unknown groups (in
            ``ulayout.groups`` order); the next ``len(self.given_groups)`` are
            given groups (in ``glayout.groups`` order); the trailing
            ``len(self.param_names)`` are the promoted parameters (in
            ``param_names`` order).

        The per-group → per-variable split is delegated to
        :meth:`AssembledVector.disassemble`, which already handles both
        BLOCK (preserve sub_batch axes, narrow per-var base) and DENSE
        (unfold sub_batch from trailing base, narrow per-var
        ``var_size``) groups. The traced graph contains the narrows /
        reshapes as standard torch ops, compiled in by Inductor. Promoted
        parameters are injected verbatim (wrapped to their typed class);
        wrapping preserves any autograd graph on the incoming tensor so the
        :class:`DrDParam` reverse pass can differentiate through them.
        """
        n_u = len(self.unknown_groups)
        n_g = len(self.given_groups)
        u_group_raws = args[:n_u]
        g_group_raws = args[n_u : n_u + n_g]
        param_raws = args[n_u + n_g :]
        # AssembledVector takes a list of typed dynamic-base ``Tensor``
        # wrappers, one per group. The raws coming in here are already the
        # group tensor data; :func:`wrap_group_raw` re-attaches the right
        # batch_ndim / sub_batch_ndim per group so disassemble interprets
        # them correctly.
        u_tensors = [
            wrap_group_raw(raw, gnames, structure, self.ulayout)
            for raw, gnames, structure in zip(
                u_group_raws, self.unknown_groups, self.ulayout.structure, strict=True
            )
        ]
        g_tensors = [
            wrap_group_raw(raw, gnames, structure, self.glayout)
            for raw, gnames, structure in zip(
                g_group_raws, self.given_groups, self.glayout.structure, strict=True
            )
        ]
        u_vec = AssembledVector(self.ulayout, u_tensors)
        g_vec = AssembledVector(self.glayout, g_tensors)
        state: dict[str, TensorWrapper] = {}
        # ``.values`` is the plain ``{name: wrapper}`` dict; pass it (not the
        # SparseVector itself) to ``dict.update`` so strict export traces it -- a
        # SparseVector iterates KEYS, which Dynamo's update tries to unpack as pairs.
        state.update(u_vec.disassemble().values)
        state.update(g_vec.disassemble().values)
        # Inject the promoted-parameter tail. Wrap each raw to its typed class
        # (wrapping preserves any incoming autograd graph -- the ParamIFT
        # reverse pass relies on it). No sub_batch: promoted parameters are
        # plain-batch (the implicit-promotion guard rejects sub-batched ones).
        for name, type_cls, raw in zip(self.param_names, self.param_types, param_raws, strict=True):
            state[name] = raw if isinstance(raw, type_cls) else type_cls(raw)
        return state

    def _call_model_from_state(
        self,
        state: dict[str, TensorWrapper],
        seed_names: tuple[str, ...],
    ) -> tuple[dict[str, TensorWrapper], ChainRuleDict]:
        """Build chain-rule seed (if any) and call the model on the typed state."""
        # Shared seed builder (same one ModelNonlinearSystem uses) so the
        # exported graph and the native eager assembly cannot drift -- notably
        # the dynamic-batch left-padding that this wrapper previously omitted.
        seed: ChainRuleDict | None = (
            build_identity_seed(
                state,
                seed_names,
                len(self.residual_groups),
                self.model.input_spec,
                self.sub_batch_shapes,
            )
            if seed_names
            else None
        )

        args = tuple(state[name] for name in self.input_names)
        result = self.model(*args, v=seed) if seed is not None else self.model(*args)
        result_tuple = result if isinstance(result, tuple) else (result,)
        if seed is None:
            output_values = result_tuple
            v_out: ChainRuleDict = {}
        else:
            output_values = result_tuple[:-1]
            v_out = result_tuple[-1]
        # Keep model outputs as typed wrappers (rule 1: no raw-tensor leaks).
        # Leaves that return raw get rewrapped via the declared output spec.
        output_state: dict[str, TensorWrapper] = {}
        for name, value in zip(self.output_names, output_values, strict=True):
            if isinstance(value, TensorWrapper):
                output_state[name] = value
            else:
                output_state[name] = self.model.output_spec[name](value)
        return output_state, v_out

    def _assembled_matrix(
        self,
        col_layout: AxisLayout,
        v_out: ChainRuleDict,
        output_state: dict[str, TensorWrapper],
    ) -> AssembledMatrix:
        """Multi-group AssembledMatrix via the sub-batch-aware block builder."""
        residual_values = {name: output_state[name] for name in self.residual_names}
        like_by_row = {
            name: _flatten_base(residual_values[name], self.model.output_spec[name])
            for name in self.residual_names
        }
        return _build_block_matrix(
            self.model,
            self.blayout,
            col_layout,
            v_out,
            like_by_row,
        )

    def _assembled_b(self, output_state: dict[str, TensorWrapper]) -> AssembledVector:
        """Multi-group b = -r as ``AssembledVector`` (per-group tensors)."""
        residual_values = {name: output_state[name] for name in self.residual_names}
        b = AssembledVector.from_dict(self.blayout, residual_values)
        return -b


def _vector_to_per_group_raws(vec: AssembledVector) -> tuple[torch.Tensor, ...]:
    """Extract per-group raw tensors from an AssembledVector.

    Single ``.data`` extract per group at the AOTI segment-output
    boundary -- the legitimate framework-imposed exception case from
    CLAUDE.md rule 2.
    """
    return tuple(t.data for t in vec.tensors)  # data-ok AOTI


def _matrix_to_per_block_raws(mat: AssembledMatrix) -> tuple[torch.Tensor, ...]:
    """Row-major per-``(row_group, col_group)`` raw blocks of an AssembledMatrix.

    The legitimate framework-imposed ``.data`` exception at the AOTI
    segment-output boundary (CLAUDE.md rule 2), mirroring
    :func:`_vector_to_per_group_raws` for the matrix operator.
    """
    return tuple(
        mat.tensors[i][j].data  # data-ok AOTI
        for i in range(mat.row_layout.ngroup)
        for j in range(mat.col_layout.ngroup)
    )


class RHS(_SystemModule):
    """Exportable residual graph.

    Contract: ``(*u_groups, *g_groups, *params) -> (*b_groups)`` -- per-group
    raw tensors. ``b_group = -r_group`` for each residual group; the C++
    runtime computes a per-batch convergence norm by reducing each group
    tensor over its trailing sub_batch + base axes and summing across
    groups (no per-variable narrow on the hot path). The promoted-parameter
    tail (scalar; constant across the solve) is empty in the common case.
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        output_state, _ = self._call_model_from_state(state, ())
        b = self._assembled_b(output_state)
        return _vector_to_per_group_raws(b)


class Jacobian(_SystemModule):
    """Exportable residual-Jacobian operator graph.

    Contract: ``(*u_groups, *g_groups, *params) -> (*A_blocks, *b_groups)`` where
    ``A_blocks`` is the row-major ``(residual_group × unknown_group)`` grid of
    ``A = ∂r/∂u`` and ``b_groups`` is ``b = -r`` at the current iterate. The
    linear solve is NOT baked here -- it lives in the separate :class:`LinearSolve`
    graph, so the same operator can feed a direct solve today or a matrix-free
    iterative solver later. The C++ runtime chains ``Jacobian -> LinearSolve`` for
    the Newton step (``step``) and reuses the ``b_groups`` tail for the
    line-search residual. The promoted-parameter tail is empty in the common case.
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        output_state, v_out = self._call_model_from_state(state, self.unknown_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        b = self._assembled_b(output_state)
        return (*_matrix_to_per_block_raws(A), *_vector_to_per_group_raws(b))


class Matvec(_SystemModule):
    """Exportable matrix-free residual jvp: ``(*u, *g, *params, *v) -> (*Jv)``.

    ``J.v = ∂r/∂u . v`` at fixed ``(u, g, params)``, where ``v`` is a tangent in
    the unknown space (same per-group layout as ``u``) and ``Jv`` is in the
    residual space (``blayout``). Never assembles ``A`` -- it threads the single
    direction ``v`` through the ``forward(v=)`` chain rule (one forward pass + one
    pushforward), the matvec an iterative/Krylov linear solver calls each inner
    iteration. This is the matrix-free counterpart of :class:`Jacobian` (which
    assembles the full ``A``): ``Matvec(u,g,v)`` equals ``Jacobian(u,g).A @ v``,
    but at O(N^2) with no O(N^3) factorization downstream. The C++ Krylov runtime
    drives it (compiled loader on the AOTI route, callback on the eager route);
    both feed the shared ``krylov_solve``.
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from ..types._boundary import assemble_jvp_outputs, leading_k1_seed  # noqa: PLC0415

        n_u = len(self.unknown_groups)
        n_g = len(self.given_groups)
        n_p = len(self.param_names)
        u_raws = args[:n_u]
        g_raws = args[n_u : n_u + n_g]
        p_raws = args[n_u + n_g : n_u + n_g + n_p]
        v_raws = args[n_u + n_g + n_p :]
        state = self._state_from_per_group_args((*u_raws, *g_raws, *p_raws))
        # v arrives per-unknown-group (ulayout); disassemble to per-variable typed
        # tangents and seed each unknown with a single K=1 direction.
        v_tensors = [
            wrap_group_raw(raw, gnames, structure, self.ulayout)
            for raw, gnames, structure in zip(
                v_raws, self.unknown_groups, self.ulayout.structure, strict=True
            )
        ]
        v_typed = AssembledVector(self.ulayout, v_tensors).disassemble().values
        seed = {name: {name: leading_k1_seed(v_typed[name])} for name in self.unknown_names}
        args_typed = tuple(state[name] for name in self.input_names)
        result = self.model(*args_typed, v=seed)
        result_tuple = result if isinstance(result, tuple) else (result,)
        typed_outs, v_out = result_tuple[:-1], result_tuple[-1]
        raw_jvp = assemble_jvp_outputs(v_out, tuple(typed_outs), list(self.output_names))
        output_state = dict(zip(self.output_names, typed_outs, strict=True))
        Jv = AssembledVector.from_dict(
            self.blayout,
            {
                r: type(output_state[r])(raw_jvp[r], sub_batch_ndim=output_state[r].sub_batch_ndim)
                for r in self.residual_names
            },
        )
        return _vector_to_per_group_raws(Jv)


class _PrecondSetup(_SystemModule):
    """Base for a preconditioner's compiled *setup* graph: assemble the single
    dense unknown group's Jacobian ``A = ∂r/∂u`` as one ``(Bflat, N, N)`` block.

    v1 targets a single dense unknown group. Concrete setups (Full / Jacobi /
    BlockJacobi) factor / select from ``A`` and return the preconditioner *state*
    (raw graph outputs) that the matching apply consumes. (A future targeted
    seed-side assembly -- computing only the blocks a preconditioner needs so
    dead-code elimination prunes the rest -- keeps this same setup->state
    interface.)
    """

    def _dense_a(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(self.unknown_groups) != 1:
            raise ValueError(
                "preconditioned Krylov currently requires a single dense unknown "
                "group (v1); use preconditioner=none for multi-group systems."
            )
        state = self._state_from_per_group_args(args)
        output_state, v_out = self._call_model_from_state(state, self.unknown_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        a = _matrix_to_per_block_raws(A)[0]  # single group -> (*B, N, N)
        n = a.shape[-1]
        return a.reshape(-1, n, n)  # (Bflat, N, N)


class FullPrecondSetup(_PrecondSetup):
    """Full preconditioner setup: ``(*u,*g,*params) -> (LU, pivots)`` = the LU
    factorization of the assembled Jacobian (M = A, so M^-1 is the exact solve)."""

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        A = self._dense_a(args)
        LU, piv = torch.linalg.lu_factor(A)
        return LU, piv


class JacobiPrecondSetup(_PrecondSetup):
    """Jacobi preconditioner setup: ``(*u,*g,*params) -> (1/diag(A),)`` with a
    sign-preserving floor so a ~0 pivot cannot produce an inf apply."""

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        A = self._dense_a(args)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)  # (Bflat, N)
        eps = torch.finfo(diag.dtype).eps
        sign = torch.where(diag.sign() == 0, torch.ones_like(diag), diag.sign())
        safe = torch.where(diag.abs() < eps, sign * eps, diag)
        return (1.0 / safe,)


class BlockJacobiPrecondSetup(_PrecondSetup):
    """Block-Jacobi preconditioner setup: ``(*u,*g,*params) -> (*inv_blocks)`` =
    the inverse of each per-variable on-diagonal block of ``A`` (one tensor per
    unknown variable, in ``unknown_names`` order)."""

    def __init__(
        self,
        system: ModelNonlinearSystem,
        block_sizes: tuple[int, ...],
        param_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__(system, param_names)
        self._block_sizes = tuple(int(s) for s in block_sizes)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        A = self._dense_a(args)  # (Bflat, N, N)
        invs: list[torch.Tensor] = []
        o = 0
        for s in self._block_sizes:
            invs.append(torch.linalg.inv(A[:, o : o + s, o : o + s]))
            o += s
        return tuple(invs)


class FullPrecondApply(nn.Module):
    """``(LU, pivots, r_flat) -> z_flat`` = ``A^-1 r`` via the cached LU."""

    def forward(self, LU: torch.Tensor, piv: torch.Tensor, r_flat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lu_solve(LU, piv, r_flat.unsqueeze(-1)).squeeze(-1)


class JacobiPrecondApply(nn.Module):
    """``(diag_recip, r_flat) -> z_flat`` = elementwise ``r / diag(A)``."""

    def forward(self, diag_recip: torch.Tensor, r_flat: torch.Tensor) -> torch.Tensor:
        return r_flat * diag_recip


class BlockJacobiPrecondApply(nn.Module):
    """``(*inv_blocks, r_flat) -> z_flat`` = block-diagonal apply of the per-variable
    inverses to the matching slices of the flat residual."""

    def __init__(self, block_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self._block_sizes = tuple(int(s) for s in block_sizes)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        *invs, r_flat = args
        pieces: list[torch.Tensor] = []
        o = 0
        for inv, s in zip(invs, self._block_sizes, strict=True):
            # Batched block matvec inv @ r_block; explicit matmul (einsum is
            # guard-blocked inside an active Model.forward -- the preconditioner
            # apply runs during the ImplicitUpdate Newton solve).
            piece = torch.matmul(inv, r_flat[:, o : o + s].unsqueeze(-1)).squeeze(-1)
            pieces.append(piece)
            o += s
        return torch.cat(pieces, dim=-1)


class LinearSolve(_SystemModule):
    """Exportable linear-solve graph: ``(*A_blocks, *b_groups) -> (*du_groups)``.

    Reconstructs the typed ``AssembledMatrix`` ``A`` (row_layout = residuals,
    col_layout = unknowns) and ``AssembledVector`` ``b`` from the raw per-group
    blocks emitted by :class:`Jacobian`, then applies the configured Python linear
    solver (``DenseLU`` / ``SchurComplement``). Un-baked from :class:`Jacobian` so
    the solver is a separate, swappable stage and the same operator can feed an
    iterative solver. Per-block ``sub_batch_ndim`` is a structural constant of the
    layouts (:func:`~neml2.es.assembled.group_block_sub_batch_ndim`); ``batch_ndim``
    follows from the runtime tensor ndim, so this graph is batch-size agnostic.
    The promoted-parameter tail carries no state here (the solve is pure linear
    algebra over the assembled operators) but is accepted for a uniform signature.
    """

    def __init__(
        self,
        system: ModelNonlinearSystem,
        linear_solver,
        param_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__(system, param_names)
        self._linear_solver = linear_solver
        self._n_r = len(self.residual_groups)
        self._n_u = len(self.unknown_groups)
        # Structural sub_batch_ndim per (residual_group, unknown_group) A block --
        # the reconstruction key at the raw solver boundary.
        self._block_sub_ndims: list[list[int]] = [
            [group_block_sub_batch_ndim(self.blayout, i, self.ulayout, j) for j in range(self._n_u)]
            for i in range(self._n_r)
        ]

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n_a = self._n_r * self._n_u
        a_raws = args[:n_a]
        b_raws = args[n_a : n_a + self._n_r]
        a_blocks = [
            [
                wrap_block_raw(a_raws[i * self._n_u + j], self._block_sub_ndims[i][j])
                for j in range(self._n_u)
            ]
            for i in range(self._n_r)
        ]
        A = AssembledMatrix(self.blayout, self.ulayout, a_blocks)
        b_tensors = [
            wrap_group_raw(raw, gnames, structure, self.blayout)
            for raw, gnames, structure in zip(
                b_raws, self.residual_groups, self.blayout.structure, strict=True
            )
        ]
        b = AssembledVector(self.blayout, b_tensors)
        du = self._linear_solver.solve(A, b)
        return _vector_to_per_group_raws(du)


class JacobianGiven(_SystemModule):
    """Exportable ``∂r/∂g`` operator graph -- the IFT's given-side Jacobian.

    Contract: ``(*u_groups, *g_groups, *params) -> (*B_blocks)`` where ``B_blocks``
    is the row-major ``(residual_group × given_group)`` grid of ``B = ∂r/∂g`` at
    the converged state. The un-baked given-side companion of :class:`Jacobian`
    (which supplies ``A = ∂r/∂u``): the C++ runtime runs ``Jacobian`` (for ``A``)
    + ``JacobianGiven`` (for ``B``) at the converged point and feeds both to
    :class:`LinearSolveIFT` for the implicit-function-theorem solve
    ``du/dg = -A^{-1} B``. Seeds the givens only. The promoted-parameter tail is
    the stored scalar (constant across the input Jacobian).
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        output_state, v_out = self._call_model_from_state(state, self.given_names)
        B = self._assembled_matrix(self.glayout, v_out, output_state)
        return _matrix_to_per_block_raws(B)


class LinearSolveIFT(_SystemModule):
    """Exportable IFT solve: ``(*A_blocks, *B_blocks) -> *du_dg_pair_blocks``.

    Reconstructs the typed ``A = ∂r/∂u`` (blayout × ulayout, from :class:`Jacobian`)
    and ``B = ∂r/∂g`` (blayout × glayout, from :class:`JacobianGiven`) from the raw
    per-group blocks, applies the configured linear solver (``du/dg = -A^{-1} B``,
    matrix RHS), then disassembles into per-``(unknown, given)`` blocks in
    :meth:`emitted_pairs` order -- matching the ``jacobian_pairs`` metadata. Shares
    the same solver as the forward Newton step (un-baked from the operators). The
    emitted pairs are plain-batch (guarded at compile time), but ``A`` itself may
    carry BLOCK unknown groups; per-block ``sub_batch_ndim`` follows the layout
    convention (:func:`~neml2.es.assembled.group_block_sub_batch_ndim`).
    """

    def __init__(
        self,
        system: ModelNonlinearSystem,
        linear_solver,
        selected_pairs: set[tuple[str, str]] | None = None,
        param_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__(system, param_names)
        self._linear_solver = linear_solver
        # Local (unknown, given) pairs to emit; ``None`` = all.
        self._selected_pairs = selected_pairs
        self._n_r = len(self.residual_groups)
        self._n_u = len(self.unknown_groups)
        self._n_g = len(self.given_groups)
        self._a_sub: list[list[int]] = [
            [group_block_sub_batch_ndim(self.blayout, i, self.ulayout, j) for j in range(self._n_u)]
            for i in range(self._n_r)
        ]
        self._b_sub: list[list[int]] = [
            [group_block_sub_batch_ndim(self.blayout, i, self.glayout, j) for j in range(self._n_g)]
            for i in range(self._n_r)
        ]

    def emitted_pairs(self) -> list[tuple[str, str]]:
        """The (unknown, given) pairs this graph emits, in emission order."""
        return [
            (u, g)
            for u in self.unknown_names
            for g in self.given_names
            if self._selected_pairs is None or (u, g) in self._selected_pairs
        ]

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n_a = self._n_r * self._n_u
        n_b = self._n_r * self._n_g
        a_raws = args[:n_a]
        b_raws = args[n_a : n_a + n_b]
        A = AssembledMatrix(
            self.blayout,
            self.ulayout,
            [
                [
                    wrap_block_raw(a_raws[i * self._n_u + j], self._a_sub[i][j])
                    for j in range(self._n_u)
                ]
                for i in range(self._n_r)
            ],
        )
        B = AssembledMatrix(
            self.blayout,
            self.glayout,
            [
                [
                    wrap_block_raw(b_raws[i * self._n_g + j], self._b_sub[i][j])
                    for j in range(self._n_g)
                ]
                for i in range(self._n_r)
            ],
        )
        du_dg = self._linear_solver.solve(A, B)
        cells = (-du_dg).disassemble().cells
        # Emit per-(unknown, given) raw blocks in the canonical order. The ``.data``
        # reads are the legitimate AOTI segment-output boundary.
        return tuple(
            cells[u][g].data  # data-ok AOTI
            for (u, g) in self.emitted_pairs()
        )


class _ParamIFTBase(_SystemModule):
    r"""Exportable parameter sensitivity $du/d\theta = -A^{-1}\,\partial r/\partial\theta$
    for a converged :class:`ImplicitUpdate`.

    Contract: ``(*u_groups, *g_groups, *params) -> *blocks`` where each block is
    one per-variable-pair ``(unknown, param)`` entry of ``du_dθ``, emitted in
    ``unknown_names`` (outer) × ``param_names`` (inner) order -- matching the
    ``param_jacobian_pairs`` metadata in
    :func:`~neml2.cli.aoti_export._compile_implicit_segment`.

    The implicit AOTI path assembles its forward Jacobian analytically (no
    autograd ``Function`` to backprop through), so the parameter sensitivity of
    the converged solution is obtained by differentiating the implicit
    constraint ``r(u(θ), g, θ) = 0``:

    .. math:: \frac{du}{d\theta} = -A^{-1}\,\frac{\partial r}{\partial\theta},
        \qquad A = \frac{\partial r}{\partial u}.

    Unlike the other three segment graphs, this one is compiled under
    ``strict=True`` (the only mode in which ``torch.autograd.grad`` lowers
    through AOTInductor), and the strict dynamo tracer does NOT tolerate the
    generator-heavy equation-system assembly machinery (``AssembledVector`` /
    ``AssembledMatrix`` / the chain rule). So this graph is deliberately
    self-contained: it reconstructs the typed model inputs by plain ``narrow`` /
    ``reshape`` (offsets precomputed in ``__init__``), runs the residual model
    forward ONCE, and forms BOTH ``A = ∂r/∂u`` and ``∂r/∂θ`` from reverse-mode
    ``torch.autograd.grad`` over the flat residual vector (``A`` via the unknown
    leaves, ``∂r/∂θ`` via the parameter leaves -- the residual Jacobian computed
    by reverse-mode IS the same ``A`` the Newton solve used). It then solves the
    full dense system ``A · du/dθ = -∂r/∂θ`` with ``torch.linalg.solve``.

    Plain-batch only (the implicit-promotion guard rejects sub-batched
    unknowns / givens / params), so the unknown and residual storage flatten to
    a single dense ``(*batch, U)`` / ``(*batch, R)`` vector with ``R == U`` and
    the full dense solve is exact -- the per-group Schur structure is only a
    forward-solve optimization and is not needed for the one-shot sensitivity.

    The promoted parameter enters PER-BATCH (``(*dyn, *param_base)``) so the
    reverse pass yields a per-batch-element ``∂r/∂θ`` (no summation across the
    batch); the C++ runtime broadcasts the stored scalar parameter to the runtime
    batch before the call, mirroring the forward dense parameter-Jacobian graph.
    Cost: ``R`` reverse passes (one per residual component), independent of the
    number of parameters.
    """

    def __init__(
        self,
        system: ModelNonlinearSystem,
        linear_solver,
        param_names: tuple[str, ...],
        selected_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(system, param_names)
        # The dense solve is done in-graph with torch.linalg.solve; the
        # configured linear solver (Schur etc.) is a forward-solve optimization
        # not needed for the one-shot sensitivity, so it is intentionally unused.
        del linear_solver
        # Local (unknown, param) pairs to emit; ``None`` = all. Emitted in
        # unknown x param order to match the metadata.
        self._selected_pairs = selected_pairs

        spec = system.model.input_spec
        out_spec = system.model.output_spec

        def _storage(base: tuple[int, ...]) -> int:
            return prod(base) if base else 1

        # Flat unknown layout (group order, then var order within group) -- the
        # COLUMN order of A and the ROW order of du/dθ. Each group tensor arrives
        # as (*batch, group_storage); per-var narrow offsets are within-group.
        # Plain-batch => DENSE groups => var storage == prod(base) (no sub-batch).
        self._u_groups_meta: list[list[tuple[str, type, tuple[int, ...], int]]] = []
        self._u_flat: list[tuple[str, tuple[int, ...], int, int]] = []  # name, base, storage, off
        u_off = 0
        for group in self.unknown_groups:
            gmeta: list[tuple[str, type, tuple[int, ...], int]] = []
            for name in group:
                tc = spec[name]
                base = tuple(int(s) for s in tc.BASE_SHAPE)
                st = _storage(base)
                gmeta.append((name, tc, base, st))
                self._u_flat.append((name, base, st, u_off))
                u_off += st
            self._u_groups_meta.append(gmeta)
        self._u_total = u_off

        # Flat given layout (values; no grad). Group order, then var order.
        self._g_groups_meta: list[list[tuple[str, type, tuple[int, ...], int]]] = []
        for group in self.given_groups:
            gmeta = []
            for name in group:
                tc = spec[name]
                base = tuple(int(s) for s in tc.BASE_SHAPE)
                gmeta.append((name, tc, base, _storage(base)))
            self._g_groups_meta.append(gmeta)

        # Flat residual layout in residual_groups order (matches the unknown
        # group/var order one-for-one, so A is square with row k <-> unknown k).
        self._r_flat: list[tuple[str, int]] = [
            (rname, _storage(tuple(int(s) for s in out_spec[rname].BASE_SHAPE)))
            for rgroup in self.residual_groups
            for rname in rgroup
        ]
        self._r_total = sum(st for _, st in self._r_flat)

        # Per-param flat metadata (base shape + storage), input_spec / tail order.
        self._param_meta: list[tuple[str, tuple[int, ...], int]] = []
        for name in self.param_names:
            base = tuple(int(s) for s in spec[name].BASE_SHAPE)
            self._param_meta.append((name, base, _storage(base)))

    def emitted_param_pairs(self) -> list[tuple[str, str]]:
        """The (unknown, param) pairs this graph emits, in emission order."""
        return [
            (u, p)
            for u in self.unknown_names
            for p in self.param_names
            if self._selected_pairs is None or (u, p) in self._selected_pairs
        ]

    # This body executes only inside `torch.export`'s Dynamo trace when the
    # ParamIFT graph is compiled; Dynamo runs transformed bytecode, so coverage.py
    # never sees these source lines even though the AOTI implicit parameter-
    # derivative tests exercise it end-to-end (compile + run + FD check). Hence the
    # coverage exclusion on the def below.


class DrDParam(_ParamIFTBase):
    """Exportable operator graph for the implicit parameter sensitivity.

    Contract: ``(*u_groups, *g_groups, *params_per_batch) -> (A_dense, Bp)`` where
    ``A = ∂r/∂u`` is the dense ``(*batch, R, U)`` residual Jacobian and ``Bp =
    ∂r/∂θ`` the dense ``(*batch, R, P)`` parameter Jacobian, both formed by
    reverse-mode ``torch.autograd.grad`` over the flat residual (the only AD that
    lowers through AOTInductor -- so this graph, like the old ``ParamIFT``, is
    compiled strict + ``trace_autograd_ops``). The solve is un-baked into
    :class:`LinearSolveParam`. Plain-batch only (the implicit-promotion guard
    rejects sub-batched unknowns / givens / params, so ``R == U`` and the system
    is a single dense block).
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:  # pragma: no cover
        n_u = len(self.unknown_groups)
        n_g = len(self.given_groups)
        u_group_raws = args[:n_u]
        g_group_raws = args[n_u : n_u + n_g]
        param_raws = args[n_u + n_g :]

        batch = tuple(u_group_raws[0].shape[:-1])  # (*batch, group_storage)

        # Re-leaf each unknown group + each per-batch parameter so autograd.grad
        # has fresh leaves (a graph input is not a grad leaf until cloned +
        # requires_grad_). Unknown leaves -> A = ∂r/∂u; param leaves -> ∂r/∂θ.
        u_leaves = [t.clone().requires_grad_(True) for t in u_group_raws]
        param_leaves: list[torch.Tensor] = []
        for type_cls, raw in zip(self.param_types, param_raws, strict=True):
            # (*batch, *param_base); .data is the AOTI input boundary unwrap.
            r = raw.data if isinstance(raw, type_cls) else raw  # data-ok AOTI
            param_leaves.append(r.clone().requires_grad_(True))

        # Reconstruct the typed per-variable state by plain narrow / reshape.
        state: dict[str, TensorWrapper] = {}
        for leaf, gmeta in zip(u_leaves, self._u_groups_meta, strict=True):
            off = 0
            for name, tc, base, st in gmeta:
                part = leaf.narrow(-1, off, st).reshape(*batch, *base)
                state[name] = tc(part)
                off += st
        for raw, gmeta in zip(g_group_raws, self._g_groups_meta, strict=True):
            off = 0
            for name, tc, base, st in gmeta:
                part = raw.narrow(-1, off, st).reshape(*batch, *base)
                state[name] = tc(part)
                off += st
        for (name, _base, _st), leaf in zip(self._param_meta, param_leaves, strict=True):
            state[name] = self.model.input_spec[name](leaf)

        # Plain forward (no chain rule): the residual values carry the autograd
        # graph w.r.t. both the unknown leaves and the parameter leaves.
        args_in = tuple(state[name] for name in self.input_names)
        result = self.model(*args_in)
        result_tuple = result if isinstance(result, tuple) else (result,)
        out_state: dict[str, TensorWrapper] = {}
        for name, value in zip(self.output_names, result_tuple, strict=True):
            out_state[name] = (
                value if isinstance(value, TensorWrapper) else self.model.output_spec[name](value)
            )

        # Flat residual vector r (*batch, R) in unknown-corresponding order. The
        # ``.data`` reads are the AOTI boundary unwrap (this file is the export
        # boundary; the residual values carry the autograd graph through .data).
        r_parts = [
            out_state[rname].data.reshape(*batch, st)  # data-ok AOTI
            for rname, st in self._r_flat
        ]
        r_flat = torch.cat(r_parts, dim=-1) if len(r_parts) > 1 else r_parts[0]
        R = self._r_total

        # Reverse-mode: one pass per residual component builds the rows of both
        # A = ∂r/∂u (*batch, R, U) and Bp = ∂r/∂θ (*batch, R, P).
        a_rows: list[torch.Tensor] = []
        bp_rows: list[torch.Tensor] = []
        for k in range(R):
            seed = torch.zeros(*batch, R, dtype=r_flat.dtype, device=r_flat.device)
            seed[..., k] = 1.0
            grads = torch.autograd.grad(
                r_flat,
                [*u_leaves, *param_leaves],
                grad_outputs=seed,
                retain_graph=True,
                allow_unused=True,
            )
            ug = grads[: len(u_leaves)]
            pg = grads[len(u_leaves) :]
            a_row = torch.cat(
                [
                    g if g is not None else torch.zeros_like(leaf)
                    for g, leaf in zip(ug, u_leaves, strict=True)
                ],
                dim=-1,
            )  # (*batch, U)
            a_rows.append(a_row)
            p_cols = [
                (g if g is not None else torch.zeros_like(leaf)).reshape(*batch, st)
                for g, leaf, (_n, _b, st) in zip(pg, param_leaves, self._param_meta, strict=True)
            ]
            # (*batch, P)
            bp_rows.append(torch.cat(p_cols, dim=-1) if len(p_cols) > 1 else p_cols[0])

        A = torch.stack(a_rows, dim=-2)  # (*batch, R, U), R == U
        Bp = torch.stack(bp_rows, dim=-2)  # (*batch, R, P)
        # Emit the dense operators; the solve lives in LinearSolveParam. Detach:
        # AOTAutograd drops requires_grad on graph outputs.
        return A.detach(), Bp.detach()


class LinearSolveParam(_ParamIFTBase):
    """Exportable parameter-sensitivity solve: ``(A_dense, Bp) -> *du_dparam_blocks``.

    Solves ``A du/dθ = -∂r/∂θ`` with a dense ``torch.linalg.solve`` (exact for the
    plain-batch square system :class:`DrDParam` emits) and slices per-``(unknown,
    param)`` block in :meth:`emitted_param_pairs` order -- matching the
    ``param_jacobian_pairs`` metadata. Un-baked from :class:`DrDParam`; pure
    linear algebra (plain compile).
    """

    def forward(self, A: torch.Tensor, Bp: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch = tuple(A.shape[:-2])
        # du/dθ = -A^{-1} ∂r/∂θ via a full dense solve (exact for plain batch).
        du_dtheta = -torch.linalg.solve(A, Bp)  # (*batch, U, P)

        u_off = {name: off for name, _b, _st, off in self._u_flat}
        u_base = {name: base for name, base, _st, _off in self._u_flat}
        u_storage = {name: st for name, _b, st, _off in self._u_flat}
        p_off: dict[str, int] = {}
        p_base: dict[str, tuple[int, ...]] = {}
        p_storage: dict[str, int] = {}
        off = 0
        for name, base, st in self._param_meta:
            p_off[name] = off
            p_base[name] = base
            p_storage[name] = st
            off += st

        blocks: list[torch.Tensor] = []
        for u, p in self.emitted_param_pairs():
            sub = du_dtheta.narrow(-2, u_off[u], u_storage[u]).narrow(-1, p_off[p], p_storage[p])
            blocks.append(sub.reshape(*batch, *u_base[u], *p_base[p]))
        return tuple(blocks)


__all__ = [
    "RHS",
    "Jacobian",
    "LinearSolve",
    "JacobianGiven",
    "LinearSolveIFT",
    "DrDParam",
    "LinearSolveParam",
    "enumerate_group_var_names",
]
