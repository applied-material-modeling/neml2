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

Three ``nn.Module`` graphs, one per piece of the Newton orchestration:

- :class:`RHS`        -- ``(*u_groups, *g_groups) -> (*b_groups)``
                         (residual eval, cheap; called every line-search
                         trial).
- :class:`NewtonStep` -- ``(*u_groups, *g_groups) -> (*du_groups, *b_groups)``
                         (Newton step direction + residual at the current
                         iterate).
- :class:`IFT`        -- ``(*u_groups, *g_groups) -> *cells`` (implicit
                         function theorem Jacobian at the converged state,
                         emitted as one typed cell per (row_group,
                         col_group) entry of the AssembledMatrix; the
                         C++ runtime consumes cells with per-cell matmul
                         accumulation against ``dg_dmaster``).

All three segments take and return per-group raw tensors at the natural
``AssembledVector`` / ``AssembledMatrix`` group shape -- BLOCK groups
preserve their ``sub_batch_shape`` axes, DENSE groups have sub_batch
folded into the last base axis. The C++ runtime maintains per-variable
``dstate`` for downstream forward composition; the per-variable ↔
per-group conversion happens twice per solve (once at solve start to
pack ``u_groups`` / ``g_groups``, once at solve end to unpack converged
``u_groups`` back to ``dstate``). The Newton inner loop is fully
per-group.
"""

from __future__ import annotations

import torch
from torch import nn

from neml2.models.chain_rule import ChainRuleDict
from neml2.types import TensorWrapper

from ._helpers import _expanded_identity_seed, _flatten_base
from .assembled import AssembledMatrix, AssembledVector, _build_block_matrix, wrap_group_raw
from .axis_layout import AxisLayout
from .system import ModelNonlinearSystem


def enumerate_group_var_names(layout: AxisLayout) -> tuple[tuple[str, ...], ...]:
    """Canonical per-group variable-name iteration order for *layout*.

    Single source of truth for the order in which group tensors are
    enumerated in segment forward signatures and in metadata
    emission. See :func:`~neml2.cli.aoti_export._enumerate_groups_and_cells`
    for the matching emitter side.
    """
    return tuple(tuple(g) for g in layout.groups)


class _SystemModule(nn.Module):
    """Tensor-only export surface for a frozen :class:`ModelNonlinearSystem`.

    Segment subclasses (:class:`RHS`, :class:`NewtonStep`, :class:`IFT`)
    take per-group raw tensors at the graph signature --
    ``forward(*u_groups, *g_groups)`` where the leading
    ``len(unknown_groups)`` positional args are the unknown groups (in
    ``ulayout.groups`` order) and the rest are the given groups (in
    ``glayout.groups`` order).
    """

    def __init__(self, system: ModelNonlinearSystem) -> None:
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

    def _state_from_per_group_args(
        self, args: tuple[torch.Tensor, ...]
    ) -> dict[str, TensorWrapper]:
        """Split positional per-group inputs into a per-variable typed state.

        Inputs:
            ``args`` -- ``(*u_groups, *g_groups)`` raw tensors. The first
            ``len(self.unknown_groups)`` are unknown groups (in
            ``ulayout.groups`` order); the remaining are given groups (in
            ``glayout.groups`` order).

        The per-group → per-variable split is delegated to
        :meth:`AssembledVector.disassemble`, which already handles both
        BLOCK (preserve sub_batch axes, narrow per-var base) and DENSE
        (unfold sub_batch from trailing base, narrow per-var
        ``var_size``) groups. The traced graph contains the narrows /
        reshapes as standard torch ops, compiled in by Inductor.
        """
        n_u = len(self.unknown_groups)
        u_group_raws = args[:n_u]
        g_group_raws = args[n_u:]
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
        state.update(u_vec.disassemble())
        state.update(g_vec.disassemble())
        return state

    def _call_model_from_state(
        self,
        state: dict[str, TensorWrapper],
        seed_names: tuple[str, ...],
    ) -> tuple[dict[str, TensorWrapper], ChainRuleDict]:
        """Build chain-rule seed (if any) and call the model on the typed state."""
        seed: ChainRuleDict | None
        if seed_names:
            # V2P-5 canonical seed: one per (input, residual_group), built
            # from the primal wrapper.
            seed = {}
            for name in seed_names:
                wrapper = state[name]
                group_seeds: dict[str, TensorWrapper] = {}
                for gi, _rgroup in enumerate(self.residual_groups):
                    group_seeds[f"{name}:rgroup{gi}"] = _expanded_identity_seed(wrapper)
                seed[name] = group_seeds
        else:
            seed = None

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
    return tuple(t.data for t in vec.tensors)  # noqa: data-ok AOTI


def _matrix_to_per_cell_raws(mat: AssembledMatrix) -> tuple[torch.Tensor, ...]:
    """Extract per-(row_group, col_group) raw tensors from an AssembledMatrix.

    Row-major order (rows outer, cols inner). Matches
    :func:`~neml2.cli.aoti_export._enumerate_groups_and_cells` on the
    metadata side -- both must use the same nested loop or the schema
    cells will desync from the loader output positions.
    """
    cells: list[torch.Tensor] = []
    for i in range(mat.row_layout.ngroup):
        for j in range(mat.col_layout.ngroup):
            cells.append(mat.tensors[i][j].data)  # noqa: data-ok AOTI
    return tuple(cells)


class RHS(_SystemModule):
    """Exportable residual graph.

    Contract: ``(*u_groups, *g_groups) -> (*b_groups)`` -- per-group raw
    tensors. ``b_group = -r_group`` for each residual group; the C++
    runtime computes a per-batch convergence norm by reducing each group
    tensor over its trailing sub_batch + base axes and summing across
    groups (no per-variable narrow on the hot path).
    """

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        output_state, _ = self._call_model_from_state(state, ())
        b = self._assembled_b(output_state)
        return _vector_to_per_group_raws(b)


class NewtonStep(_SystemModule):
    """Exportable Newton step-direction graph.

    Contract: ``(*u_groups, *g_groups) -> (*du_groups, *b_groups)`` where
    ``du_groups`` are the per-unknown-group step directions (in
    ``ulayout.groups`` order) and ``b_groups`` are the per-residual-group
    ``b = -r(u)`` at the current iterate (in ``blayout.groups`` order).
    The C++ runtime applies ``u_groups[i] = u_groups[i] + alpha *
    du_groups[i]`` per-group for line-search trials via cheap
    :class:`RHS` evaluations.
    """

    def __init__(self, system: ModelNonlinearSystem, linear_solver) -> None:
        super().__init__(system)
        self._linear_solver = linear_solver

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        output_state, v_out = self._call_model_from_state(state, self.unknown_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        b = self._assembled_b(output_state)
        du = self._linear_solver.solve(A, b)
        return (*_vector_to_per_group_raws(du), *_vector_to_per_group_raws(b))


class IFT(_SystemModule):
    """Exportable IFT Jacobian $du/dg = -A^{-1} B$ for a converged
    :class:`ImplicitUpdate`.

    Contract: ``(*u_groups, *g_groups) -> *cells`` where each cell is one
    ``(row_group_i, col_group_j)`` entry of ``-du_dg`` in row-major
    order (matches
    :func:`~neml2.cli.aoti_export._enumerate_groups_and_cells`). Each
    cell stays at its natural per-(row_structure, col_structure) shape;
    no fold to dense.

    The C++ runtime consumes the cells via per-cell matmul against the
    matching ``dg_dmaster`` segment, accumulating into per-unknown
    ``du_dmaster`` slots. Per-grain BLOCK + BLOCK cells stay
    block-diagonal-compact throughout -- the runtime never materialises
    the N² dense block.
    """

    def __init__(self, system: ModelNonlinearSystem, linear_solver) -> None:
        super().__init__(system)
        self._linear_solver = linear_solver

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._state_from_per_group_args(args)
        seed_names = (*self.unknown_names, *self.given_names)
        output_state, v_out = self._call_model_from_state(state, seed_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        B = self._assembled_matrix(self.glayout, v_out, output_state)
        du_dg = self._linear_solver.solve(A, B)
        return _matrix_to_per_cell_raws(-du_dg)


__all__ = ["RHS", "NewtonStep", "IFT", "enumerate_group_var_names"]
