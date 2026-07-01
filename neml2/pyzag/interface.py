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

"""Adapt a NEML2 ``ModelNonlinearSystem`` to pyzag's ``NonlinearFunctionOperatorFactory``."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from pyzag import nonlinear
from pyzag.operators.neml2 import (
    BlockLayout,
    BlockMatrix,
    BlockTensor,
    BlockVectorAM,
    GroupSpec,
    IStructure,
    NEML2BlockJacobian,
    NEML2Wrapper,
    _av_to_flat,
    _pack_per_var_to_av,
    _split_flat_per_var,
)

from neml2.es import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    ModelNonlinearSystem,
)
from neml2.types import Tensor, TensorWrapper
from neml2.types._boundary import to_torch


def lag_order(var: str) -> tuple[str, int]:
    """Split ``"name"`` (lag 0) or ``"name~n"`` into base name and lag order."""
    tokens = var.split("~")
    if len(tokens) == 1:
        return tokens[0], 0
    if len(tokens) == 2:
        return tokens[0], int(tokens[1])
    raise ValueError(
        f"Variable {var!r} has invalid format; expected 'name' or 'name~n' where n is an integer"
    )


def change_lag_order(var: str, new_order: int) -> str:
    """Re-tag a variable name to a different lag order; inverse of :func:`lag_order`."""
    base, _ = lag_order(var)
    return f"{base}~{new_order}" if new_order != 0 else base


def _extract_sublayout(
    layout: AxisLayout,
    order: int,
    new_order: int | None = None,
) -> AxisLayout:
    """Group-preserving sub-layout of variables whose lag matches ``order``."""
    target_order = new_order if new_order is not None else 0
    grouped: list[list[str]] = []
    specs: dict = {}
    sub: dict = {}
    structure: list[str] = []
    for gi, group in enumerate(layout.groups):
        kept: list[str] = []
        for name in group:
            if lag_order(name)[1] != order:
                continue
            new_name = change_lag_order(name, target_order)
            kept.append(new_name)
            specs[new_name] = layout.type_of(name)
            sub[new_name] = layout.sub_batch_shape(name)
        if kept:
            grouped.append(kept)
            structure.append(layout.structure[gi])
    if not grouped:
        return AxisLayout([[]], {}, {}, ("dense",))
    return AxisLayout(grouped, specs, sub, tuple(structure))


def _neml2_to_pyzag_layout(layout: AxisLayout) -> BlockLayout:
    """NEML2 ``AxisLayout`` -> pyzag ``BlockLayout``."""
    groups = []
    for gi, group in enumerate(layout.groups):
        names = tuple(group)
        intmd_sizes = tuple(tuple(layout.sub_batch_shape(n)) for n in group)
        base_sizes = tuple(tuple(layout.type_of(n).BASE_SHAPE or ()) for n in group)
        istr = (
            IStructure.BLOCK
            if layout.structure[gi] == "block"
            else IStructure.DENSE
        )
        groups.append(
            GroupSpec(
                names=names,
                istructure=istr,
                intmd_sizes=intmd_sizes,
                base_sizes=base_sizes,
            )
        )
    return BlockLayout(groups=tuple(groups))


def _neml2_av_to_pyzag_bv(av: AssembledVector) -> BlockVectorAM:
    """NEML2 ``AssembledVector`` -> pyzag ``BlockVectorAM``."""
    layout = _neml2_to_pyzag_layout(av.layout)
    tensors = []
    for t in av.tensors:
        dynamic_dim = t.ndim - 1 - t.sub_batch_ndim
        tensors.append(BlockTensor(to_torch(t), dynamic_dim, t.sub_batch_ndim))
    return BlockVectorAM(layout, tensors)


def _neml2_am_to_pyzag_bm(am: AssembledMatrix) -> BlockMatrix:
    """NEML2 ``AssembledMatrix`` -> pyzag ``BlockMatrix``."""
    row = _neml2_to_pyzag_layout(am.row_layout)
    col = _neml2_to_pyzag_layout(am.col_layout)
    n_row = am.row_layout.ngroup
    n_col = am.col_layout.ngroup
    tensors = [[BlockTensor() for _ in range(n_col)] for _ in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            t = am.tensors[i][j]
            dynamic_dim = t.ndim - 2 - t.sub_batch_ndim
            tensors[i][j] = BlockTensor(to_torch(t), dynamic_dim, t.sub_batch_ndim)
    return BlockMatrix(row, col, tensors)


def _expand_bm_dynamic(
    bm: BlockMatrix, target_dynamic_shape: tuple[int, ...]
) -> BlockMatrix:
    """Broadcast every defined block's dynamic dims to ``target_dynamic_shape``."""
    target_ndim = len(target_dynamic_shape)
    n_row = bm.row_layout.ngroup()
    n_col = bm.col_layout.ngroup()
    out = [[BlockTensor() for _ in range(n_col)] for _ in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            blk = bm.tensors[i][j]
            if not blk.defined():
                continue
            raw = blk.torch()
            intmd_dim = blk.intmd.dim()
            cur_dyn = blk.dynamic.dim()
            while cur_dyn < target_ndim:
                raw = raw.unsqueeze(0)
                cur_dyn += 1
            full_target = tuple(target_dynamic_shape) + tuple(raw.shape[target_ndim:])
            raw = raw.expand(full_target).contiguous()
            out[i][j] = BlockTensor(raw, target_ndim, intmd_dim)
    return BlockMatrix(bm.row_layout, bm.col_layout, out)


class NEML2PyzagFactory(torch.nn.Module, nonlinear.NonlinearFunctionOperatorFactory):
    """Adapt a NEML2 ``ModelNonlinearSystem`` to pyzag's ``NonlinearFunctionOperatorFactory``.

    Args:
        sys: the native NEML2 nonlinear system to wrap.

    Keyword Args:
        exclude_parameters (list of str): NEML2 parameters to *not* mirror as torch
            parameters. Mutually exclusive with ``include_parameters``.
        include_parameters (list of str): the *only* NEML2 parameters to mirror as
            torch parameters. Mutually exclusive with ``exclude_parameters``.
        compile (bool): compile the residual model in place via ``neml2.compile`` on
            construction. Default ``True``; pass ``False`` as a correctness oracle.
    """

    def __init__(
        self,
        sys: ModelNonlinearSystem,
        *args,
        exclude_parameters: list[str] | None = None,
        include_parameters: list[str] | None = None,
        compile: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if exclude_parameters is not None and include_parameters is not None:
            raise ValueError(
                "exclude_parameters and include_parameters are mutually exclusive; "
                "pass at most one (omit both to train all parameters)."
            )

        if not isinstance(sys, ModelNonlinearSystem):
            raise TypeError(
                f"sys should be a neml2.es.ModelNonlinearSystem, "
                f"instead got {type(sys)}. Use neml2.load_nonlinear_system "
                "or load_input(path).get_equation_system(name) to obtain one."
            )

        self.sys = sys
        object.__setattr__(self, "model", sys.model)
        self._lookback = 1

        self._setup_maps()
        self._check_model()
        self._setup_parameters(exclude_parameters, include_parameters)
        self._wrapper = NEML2Wrapper(_neml2_to_pyzag_layout(self.slayout))
        if compile:
            self._compile()

    _COMPILE_CACHE_SIZE_LIMIT = 512

    def _compile(self) -> None:
        """Compile the residual model in place and raise dynamo cache limits."""
        import torch._dynamo.config
        import torch._functorch.config

        from neml2.models.compile import compile as _neml2_compile

        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, self._COMPILE_CACHE_SIZE_LIMIT
        )
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit,
            8 * self._COMPILE_CACHE_SIZE_LIMIT,
        )
        _neml2_compile(self.sys)

    @property
    def lookback(self) -> int:
        return self._lookback

    @lookback.setter
    def lookback(self, lookback: int):
        if lookback != 1:
            raise ValueError("NEML2 models only support lookback of 1")
        self._lookback = lookback

    @property
    def nstate(self) -> int:
        return sum(
            math.prod(self.slayout.specs[v].BASE_SHAPE or (1,))
            for v in self.slayout.vars()
        )

    @property
    def nforce(self) -> int:
        return sum(
            math.prod(self.flayout.specs[v].BASE_SHAPE or (1,))
            for v in self.flayout.vars()
        )

    @property
    def wrapper(self):
        return self._wrapper

    def _infer_sub_batch(self, value_dict, layout: AxisLayout, dynamic_dim: int) -> dict:
        """Infer each variable's runtime sub_batch shape from its tensor shape."""
        out: dict = {}
        for name in layout.vars():
            t = value_dict[name]
            base_ndim = len(layout.type_of(name).BASE_SHAPE or ())
            out[name] = tuple(t.shape[dynamic_dim : t.ndim - base_ndim])
        return out

    def _assemble_dict_to_flat(
        self, value_dict, pyzag_layout: BlockLayout, dynamic_dim: int
    ) -> torch.Tensor:
        """Pack a per-variable tensor dict into a single flat tensor for ``pyzag_layout``."""
        parts = []
        for gspec in pyzag_layout.groups:
            for vi, name in enumerate(gspec.names):
                intmd_dim = len(gspec.intmd_sizes[vi])
                parts.append(BlockTensor(value_dict[name], dynamic_dim, intmd_dim))
        return _av_to_flat(_pack_per_var_to_av(pyzag_layout, parts))

    def assemble_state(self, ic_dict, dynamic_dim: int = 1) -> torch.Tensor:
        """Build a flat initial-state tensor from a per-variable IC dict."""
        sb_state = self._infer_sub_batch(ic_dict, self.slayout, dynamic_dim)
        self.slayout = self.slayout.with_sub_batch_shapes(
            {n: torch.Size(sb_state[n]) for n in sb_state}
        )
        self.snlayout = self.snlayout.with_sub_batch_shapes(
            {n: torch.Size(sb_state.get(lag_order(n)[0], ())) for n in self.snlayout.vars()}
        )
        r_sub = {
            r_name: torch.Size(sb_state[s_name])
            for r_name, s_name in zip(self.rlayout.vars(), self.slayout.vars(), strict=True)
        }
        self.rlayout = self.rlayout.with_sub_batch_shapes(r_sub)
        self._wrapper = NEML2Wrapper(_neml2_to_pyzag_layout(self.slayout))
        return self._assemble_dict_to_flat(
            ic_dict, _neml2_to_pyzag_layout(self.slayout), dynamic_dim
        )

    def assemble_forces(self, forces_dict, dynamic_dim: int = 2) -> torch.Tensor:
        """Build a flat forces tensor from a per-variable forces dict."""
        sb_force = self._infer_sub_batch(forces_dict, self.flayout, dynamic_dim)
        self.flayout = self.flayout.with_sub_batch_shapes(
            {n: torch.Size(sb_force[n]) for n in sb_force}
        )
        self.fnlayout = self.fnlayout.with_sub_batch_shapes(
            {n: torch.Size(sb_force.get(lag_order(n)[0], ())) for n in self.fnlayout.vars()}
        )
        return self._assemble_dict_to_flat(
            forces_dict, _neml2_to_pyzag_layout(self.flayout), dynamic_dim
        )

    def make_operator(
        self,
        prev_solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
        inverse_operator,
    ) -> nonlinear.ChunkOp:
        return nonlinear.ChunkOp(self, prev_solution, forces, inverse_operator)

    def evaluate_raw(
        self, x_full: torch.Tensor, forces: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, NEML2BlockJacobian]:
        """Return ``(r, NEML2BlockJacobian)`` for one chunk's state and forces."""
        self._update_parameter_values()
        self._update_sys(x_full, forces[0])
        A, B, b = self.sys.A_and_B_and_b()
        r, J_diag, J_sub = self._adapt_for_pyzag(A, B, b)
        return r, NEML2BlockJacobian(
            J_diag, J_sub, _neml2_to_pyzag_layout(self.slayout)
        )

    def _check_model(self):
        """Every old-state / old-force variable must have a current counterpart."""
        snvars_renamed = {change_lag_order(v, 0) for v in self.snlayout.vars()}
        if not snvars_renamed <= set(self.slayout.vars()):
            raise ValueError(
                "Input old state variables must be a subset of input state "
                f"variables. State variables: {list(self.slayout.vars())}; "
                f"old state variables: {list(self.snlayout.vars())}."
            )
        fnvars_renamed = {change_lag_order(v, 0) for v in self.fnlayout.vars()}
        if not fnvars_renamed <= set(self.flayout.vars()):
            raise ValueError(
                "Input old force variables must be a subset of input force "
                f"variables. Force variables: {list(self.flayout.vars())}; "
                f"old force variables: {list(self.fnlayout.vars())}."
            )

    def _setup_maps(self):
        """Build the state / force / residual layouts and their old-variable subsets."""
        glayout = self.sys.glayout
        gvar_set = set(glayout.vars())

        self.rlayout = self.sys.blayout
        self.slayout = self.sys.ulayout
        self.snlayout = _extract_sublayout(self.slayout, 0, new_order=1)
        self.flayout = _extract_sublayout(glayout, 0)
        self.fnlayout = _extract_sublayout(glayout, 0, new_order=1)
        self._sn_in_glayout: set[str] = {v for v in self.snlayout.vars() if v in gvar_set}
        self._fn_in_glayout: set[str] = {v for v in self.fnlayout.vars() if v in gvar_set}

        self.svars = self.slayout.vars()
        self.fvars = self.flayout.vars()

    def _setup_parameters(
        self,
        exclude_parameters: list[str] | None,
        include_parameters: list[str] | None,
    ):
        """Mirror each HIT-named NEML2 parameter as a ``torch.nn.Parameter``."""
        exclude = set(exclude_parameters or [])
        include = None if include_parameters is None else set(include_parameters)

        self.parameter_names: list[str] = []
        self._param_targets: dict[str, tuple[torch.nn.Module, str]] = {}
        available: set[str] = set()

        for module in self.sys.model.modules():
            mod_hit = getattr(module, "_hit_name", None)
            if not (mod_hit and mod_hit.isidentifier()):
                continue
            for leaf_name, param in module.named_parameters(recurse=False):
                flat = f"{mod_hit}_{leaf_name}"
                available.add(flat)
                if flat in self._param_targets:
                    continue
                if include is not None:
                    if flat not in include:
                        continue
                elif flat in exclude:
                    continue
                self.parameter_names.append(flat)
                param.requires_grad_(True)
                self.register_parameter(flat, torch.nn.Parameter(param.detach().clone()))
                self._param_targets[flat] = (module, leaf_name)

        if include is not None:
            unknown = include - available
            if unknown:
                raise ValueError(
                    f"include_parameters contains unknown parameter(s): {sorted(unknown)}. "
                    f"Available parameters: {sorted(available)}."
                )

    def _update_parameter_values(self):
        """Bind the factory's parameter tensors onto the underlying NEML2 model."""
        for pname in self.parameter_names:
            new_value = getattr(self, pname)
            module, leaf = self._param_targets[pname]
            typed_cls = getattr(module, "_typed_storage_classes", {}).get(leaf)
            sbn = getattr(module, "_typed_storage_sub_batch_ndim", {}).get(leaf, 0)

            if isinstance(new_value, torch.nn.Parameter):
                module.__dict__.pop(leaf, None)
                module._parameters[leaf] = new_value
            else:
                module._parameters.pop(leaf, None)
                bound = (
                    typed_cls(new_value, sub_batch_ndim=sbn)
                    if typed_cls is not None
                    else new_value
                )
                object.__setattr__(module, leaf, bound)

    def _update_sys(self, state: torch.Tensor, forces: torch.Tensor):
        """Disassemble pyzag state / forces into the per-variable dicts the system consumes."""
        assert state.shape[:-1] == forces.shape[:-1]

        state_now = state[self.lookback :]
        state_prev = state[: -self.lookback]
        forces_now = forces[self.lookback :]
        forces_prev = forces[: -self.lookback]

        u = self._split_by_layout(state_now, self.slayout)
        sn = self._split_by_layout(state_prev, self.snlayout, rename_to_lag=0)
        f = self._split_by_layout(forces_now, self.flayout)
        fn = self._split_by_layout(forces_prev, self.fnlayout, rename_to_lag=0)

        sub_batch_ndim: dict[str, int] = {}
        for name in u:
            sub_batch_ndim[name] = len(self.slayout.sub_batch_shape(name))

        g_state: dict[str, TensorWrapper] = {}
        for sn_name, tensor in sn.items():
            sn_full = change_lag_order(sn_name, 1)
            if sn_full in self._sn_in_glayout:
                g_state[sn_full] = tensor
                sub_batch_ndim[sn_full] = len(self.snlayout.sub_batch_shape(sn_full))
        for f_name, tensor in f.items():
            g_state[f_name] = tensor
            sub_batch_ndim[f_name] = len(self.flayout.sub_batch_shape(f_name))
        for fn_name, tensor in fn.items():
            fn_full = change_lag_order(fn_name, 1)
            if fn_full in self._fn_in_glayout:
                g_state[fn_full] = tensor
                sub_batch_ndim[fn_full] = len(self.fnlayout.sub_batch_shape(fn_full))

        dyn_shape = tuple(state_now.shape[:-1])
        u_sv, g_sv = self.sys.to_sparse(u, g_state, sub_batch_ndim)
        self.sys.initialize(u=u_sv, g=g_sv, dyn_shape=dyn_shape)

    def _split_by_layout(
        self,
        flat: torch.Tensor,
        layout: AxisLayout,
        rename_to_lag: int | None = None,
    ) -> dict[str, TensorWrapper]:
        """Split a flat ``(*batch, nflat)`` tensor into a per-variable typed wrapper dict."""
        pyzag_layout = _neml2_to_pyzag_layout(layout)
        per_var = _split_flat_per_var(flat, pyzag_layout)
        out: dict[str, TensorWrapper] = {}
        for name in layout.vars():
            type_cls = layout.type_of(name)
            sbn = len(layout.sub_batch_shape(name))
            out_name = (
                change_lag_order(name, rename_to_lag)
                if rename_to_lag is not None
                else name
            )
            out[out_name] = type_cls(per_var[name], sub_batch_ndim=sbn)
        return out

    def _adapt_for_pyzag(
        self, A: AssembledMatrix, B: AssembledMatrix, b: AssembledVector
    ) -> tuple[torch.Tensor, BlockMatrix, BlockMatrix]:
        """Convert NEML2 ``(A, B, b)`` into pyzag-native ``(r, J_diag, J_sub)``."""
        r = -_av_to_flat(_neml2_av_to_pyzag_bv(b))
        target_batch = tuple(r.shape[:-1])

        J_diag = _expand_bm_dynamic(_neml2_am_to_pyzag_bm(A), target_batch)
        J_sub = _expand_bm_dynamic(
            self._build_jn_pyzag(B, target_batch), target_batch
        )
        return r, J_diag, J_sub

    def _build_jn_pyzag(
        self, B: AssembledMatrix, target_batch: tuple[int, ...]
    ) -> BlockMatrix:
        """Assemble the subdiagonal ``Jn = dr/du_old`` from the old-state columns of ``B``."""
        cells = B.disassemble().cells
        prow = _neml2_to_pyzag_layout(self.rlayout)
        pcol = _neml2_to_pyzag_layout(self.snlayout)
        out = [
            [BlockTensor() for _ in range(pcol.ngroup())]
            for _ in range(prow.ngroup())
        ]
        for i, rg in enumerate(prow.groups):
            row_bases = [int(math.prod(b)) for b in rg.base_sizes]
            row_off = [sum(row_bases[:k]) for k in range(len(row_bases))]
            for j, cg in enumerate(pcol.groups):
                col_bases = [int(math.prod(b)) for b in cg.base_sizes]
                col_off = [sum(col_bases[:k]) for k in range(len(col_bases))]

                if rg.istructure == IStructure.BLOCK:
                    grain = tuple(rg.intmd_sizes[0])
                elif cg.istructure == IStructure.BLOCK:
                    grain = tuple(cg.intmd_sizes[0])
                else:
                    grain = ()

                found: dict = {}
                ref = None
                for ri, rn in enumerate(rg.names):
                    inner = cells.get(rn, {})
                    for ci, cn in enumerate(cg.names):
                        if cn in self._sn_in_glayout and cn in inner:
                            found[(ri, ci)] = inner[cn]
                            ref = inner[cn]
                if ref is None:
                    continue

                block = torch.zeros(
                    tuple(target_batch) + grain + (sum(row_bases), sum(col_bases)),
                    dtype=ref.dtype,
                    device=ref.device,
                )
                for (ri, ci), cell in found.items():
                    rb, cb = row_bases[ri], col_bases[ci]
                    norm = self._normalize_cell(cell, target_batch, grain, rb, cb)
                    ro, co = row_off[ri], col_off[ci]
                    block[..., ro : ro + rb, co : co + cb] = norm
                out[i][j] = BlockTensor(block, len(target_batch), len(grain))
        return BlockMatrix(prow, pcol, out)

    @staticmethod
    def _normalize_cell(
        cell: Tensor,
        target_batch: tuple[int, ...],
        grain: tuple[int, ...],
        rb: int,
        cb: int,
    ) -> torch.Tensor:
        """Broadcast a disassembled cell to ``(*target_batch, *grain, rb, cb)``."""
        x = to_torch(cell)
        sbn = cell.sub_batch_ndim
        for _ in range(len(grain) - sbn):
            x = x.unsqueeze(x.ndim - 2)
        full = tuple(target_batch) + tuple(grain) + (rb, cb)
        while x.ndim < len(full):
            x = x.unsqueeze(0)
        return x.expand(full)
