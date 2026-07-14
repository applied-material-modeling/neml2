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

"""Abstract :class:`LinearSystem` / :class:`NonlinearSystem` interfaces +
the model-backed :class:`ModelNonlinearSystem` concrete implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, dependency, option
from neml2.types import Tensor, TensorWrapper

from ._helpers import (
    _batch_shape,
    _flatten_base,
    build_identity_seed,
)
from .assembled import AssembledMatrix, AssembledVector, _build_block_matrix, wrap_group_raw
from .axis_layout import AxisLayout, SubBatchStructure
from .sparse import SparseVector

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


# V2P-5: _compact_axes_for_pair / _group_compact_axes removed. The
# canonical no-classification seed lives in :func:`_expanded_identity_seed`
# and emits broadcast K_paired for every sub_batch axis -- no per-pair
# label classification needed.


class LinearSystem:
    """Base class for systems with assembled operators."""

    #: HIT section for ``neml2-syntax`` classification -- inherited by every
    #: registered subclass (``ModelNonlinearSystem`` lives under
    #: ``[EquationSystems]`` in the input file).
    SECTION = "EquationSystems"

    def __init__(self) -> None:
        self._ulayout = self.setup_ulayout()
        self._glayout = self.setup_glayout()
        self._blayout = self.setup_blayout()

    @property
    def ulayout(self) -> AxisLayout:
        return self._ulayout

    @property
    def glayout(self) -> AxisLayout:
        return self._glayout

    @property
    def blayout(self) -> AxisLayout:
        return self._blayout

    def setup_ulayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        raise NotImplementedError

    def setup_glayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        raise NotImplementedError

    def setup_blayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        raise NotImplementedError

    def set_u(self, u: AssembledVector | SparseVector) -> None:
        raise NotImplementedError

    def set_g(self, g: AssembledVector | SparseVector) -> None:
        raise NotImplementedError

    def u(self) -> AssembledVector:
        raise NotImplementedError

    def g(self) -> AssembledVector:
        raise NotImplementedError

    def assemble(
        self,
        need_A: bool,
        need_B: bool,
        need_b: bool,
    ) -> tuple[AssembledMatrix | None, AssembledMatrix | None, AssembledVector | None]:
        raise NotImplementedError

    def A(self) -> AssembledMatrix:
        A, _, _ = self.assemble(True, False, False)
        assert A is not None
        return A

    def b(self) -> AssembledVector:
        _, _, b = self.assemble(False, False, True)
        assert b is not None
        return b

    def A_and_b(self) -> tuple[AssembledMatrix, AssembledVector]:
        A, _, b = self.assemble(True, False, True)
        assert A is not None and b is not None
        return A, b

    def A_and_B(self) -> tuple[AssembledMatrix, AssembledMatrix]:
        A, B, _ = self.assemble(True, True, False)
        assert A is not None and B is not None
        return A, B

    def A_and_B_and_b(self) -> tuple[AssembledMatrix, AssembledMatrix, AssembledVector]:
        A, B, b = self.assemble(True, True, True)
        assert A is not None and B is not None and b is not None
        return A, B, b


class NonlinearSystem(LinearSystem):
    """Nonlinear system with C++-matching Newton sign convention."""


@register_neml2_object("NonlinearSystem")
class ModelNonlinearSystem(NonlinearSystem):
    """A nonlinear system defined by a Model."""

    hit = HitSchema(
        dependency("model", "get_model", "The Model defining this nonlinear system."),
        option(
            "unknowns",
            list,
            "Ordering and grouping of unknowns. Each inner list defines one variable group.",
        ),
        option(
            "residuals",
            list,
            "Ordering and grouping of residual variables. Each inner list defines one "
            "variable group.",
            default=[],
        ),
        option(
            "structure",
            list,
            "Per-group SubBatchStructure ('block' or 'dense'). One token per group, "
            "in group order. Mirrors v2's `istructure = 'BLOCK DENSE'`. "
            "BLOCK preserves sub_batch axes as intmd dims on the per-group "
            "assembled tensor (per-grain / per-cell / per-bin independence "
            "stays compact). DENSE folds sub_batch into base. Defaults to "
            "all 'dense' when omitted.",
            default=[],
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> ModelNonlinearSystem:
        model = factory.get_model(node.param_str("model"))
        # HIT `unknowns = 'a b; c'` -> two groups [['a','b'], ['c']]; a single
        # whitespace-separated list with no `;` collapses to one group. Matches
        # the C++ side (which also uses `;` as the group separator).
        unknowns = [list(group) for group in node.param_list_list_str("unknowns")]
        residuals: list[list[str]] | None = None
        if node.find("residuals") is not None:
            residuals = [list(group) for group in node.param_list_list_str("residuals")]
        structure: list[str] | None = None
        if node.find("structure") is not None:
            # HIT `structure = 'block dense'` -> ['block', 'dense'] (one per group).
            raw = node.param_list_str("structure")
            structure = [k.lower() for k in raw]
        return cls(model, unknowns=unknowns, residuals=residuals, structure=structure)

    def __init__(
        self,
        model: Model,
        unknowns: list[list[str]],
        residuals: list[list[str]] | None = None,
        structure: list[str] | None = None,
    ) -> None:
        self.model = model
        self.unknown_groups = [list(group) for group in unknowns]
        self.residual_groups = (
            [list(group) for group in residuals]
            if residuals is not None
            else self._infer_residual_groups(self.unknown_groups)
        )
        self.unknown_names = [name for group in self.unknown_groups for name in group]
        self.residual_names = [name for group in self.residual_groups for name in group]
        self.given_names = [name for name in model.input_spec if name not in self.unknown_names]
        # Per-group SubBatchStructure. Default to all 'dense' when caller doesn't
        # specify. Length must match the unknown_groups count.
        if structure is None or not structure:
            self._structure: tuple[SubBatchStructure, ...] = ("dense",) * len(self.unknown_groups)
        else:
            if len(structure) != len(self.unknown_groups):
                raise ValueError(
                    f"ModelNonlinearSystem: structure has {len(structure)} entries, expected "
                    f"{len(self.unknown_groups)} (one per unknown group)."
                )
            lowered = tuple(k.lower() for k in structure)
            for k in lowered:
                if k not in ("block", "dense"):
                    raise ValueError(
                        f"ModelNonlinearSystem: structure entries must be "
                        f"'block' or 'dense', got {k!r}."
                    )
            self._structure = cast(tuple[SubBatchStructure, ...], lowered)
        #: State store keyed by variable name. Typed wrappers
        #: (``Scalar``/``SR2``/...) -- the wrapper-discipline rule
        #: (CLAUDE.md) forbids stashing raw ``torch.Tensor`` here.
        #: ``_call_model``, ``_identity_seed``, and the assembly path all
        #: read typed values from this dict.
        self._state: dict[str, TensorWrapper] = {}
        # Populated deterministically at ``initialize`` time from the
        # caller-supplied per-variable ``sub_batch_ndim`` dict.
        self._dynamic_batch_ndim: dict[str, int] = {}
        self._sub_batch_shapes: dict[str, torch.Size] = {}
        super().__init__()

    def _infer_residual_groups(self, unknowns: list[list[str]]) -> list[list[str]]:
        # Default residual-name convention matches C++ `residual_name()`:
        # `<variable>_residual`.
        residual_groups: list[list[str]] = []
        for group in unknowns:
            residual_group: list[str] = []
            for name in group:
                residual = f"{name}_residual"
                if residual not in self.model.output_spec:
                    raise KeyError(
                        f"Could not infer residual for unknown {name!r}; "
                        f"expected output {residual!r}. Provide explicit "
                        "`residuals = ...` in the [EquationSystems] block."
                    )
                residual_group.append(residual)
            residual_groups.append(residual_group)
        return residual_groups

    def setup_ulayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        # ``sub_batch_shapes=None`` reads from the system's current state
        # (the default used by ``__init__`` and the post-``initialize``
        # rebuild). Callers that want to pre-build a layout for
        # :class:`SparseVector` construction *before* ``initialize`` pass
        # an explicit dict.
        sb = self._sub_batch_shapes if sub_batch_shapes is None else sub_batch_shapes
        return AxisLayout(
            self.unknown_groups,
            self.model.input_spec,
            sub_batch_shapes={k: v for k, v in sb.items() if k in self.unknown_names},
            structure=self._structure,
        )

    def setup_glayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        # Auto-mirror: split givens into groups whose structure mirrors the
        # unknown groups. A given with sub_batch matching a BLOCK unknown
        # group's sub_batch lands in a BLOCK col group with that same
        # sub_batch; everything else collapses into a trailing DENSE group.
        # This keeps the per-(grain row, grain col) cross-block in the IFT
        # B matrix at O(N) storage instead of the O(N^2) you get when a
        # per-grain given is folded DENSE.
        sb = self._sub_batch_shapes if sub_batch_shapes is None else sub_batch_shapes
        given_sb = {k: v for k, v in sb.items() if k in self.given_names}
        block_groups: list[list[str]] = []
        block_structures: list[SubBatchStructure] = []
        used: set[str] = set()
        for ugroup, ustructure in zip(self.unknown_groups, self._structure, strict=True):
            if ustructure != "block":
                continue
            u_sb = next(
                (sb[u] for u in ugroup if u in sb),
                None,
            )
            if u_sb is None:
                continue
            matched = [g for g in self.given_names if g not in used and given_sb.get(g) == u_sb]
            if matched:
                block_groups.append(matched)
                block_structures.append("block")
                used.update(matched)
        dense_remainder = [g for g in self.given_names if g not in used]
        all_groups = block_groups + ([dense_remainder] if dense_remainder else [])
        dense_tail: list[SubBatchStructure] = ["dense"] if dense_remainder else []
        all_structures: tuple[SubBatchStructure, ...] = tuple(block_structures + dense_tail)
        if not all_groups:
            # No givens at all -- single empty DENSE group keeps invariants.
            all_groups = [[]]
            all_structures = ("dense",)
        return AxisLayout(
            all_groups,
            self.model.input_spec,
            sub_batch_shapes=given_sb,
            structure=all_structures,
        )

    def setup_blayout(self, sub_batch_shapes: Mapping[str, torch.Size] | None = None) -> AxisLayout:
        # Residuals inherit per-variable sub-batch AND per-group structure from the
        # matching unknown (residual_groups[i][j] corresponds to unknown_groups[i][j]).
        sb = self._sub_batch_shapes if sub_batch_shapes is None else sub_batch_shapes
        residual_sb: dict[str, torch.Size] = {}
        for ugroup, rgroup in zip(self.unknown_groups, self.residual_groups, strict=True):
            for uname, rname in zip(ugroup, rgroup, strict=True):
                if uname in sb:
                    residual_sb[rname] = sb[uname]
        return AxisLayout(
            self.residual_groups,
            self.model.output_spec,
            sub_batch_shapes=residual_sb,
            structure=self._structure,
        )

    def to(self, *args, **kwargs) -> ModelNonlinearSystem:
        """Move the underlying ``Model`` and any populated state to a new
        device / dtype.

        Matches torch's ``nn.Module.to`` signature and convention: forwards
        ``*args`` / ``**kwargs`` to ``self.model.to(...)`` (covering
        ``to(device='cuda')``, ``to(dtype=torch.float32)``,
        ``to('cuda', non_blocking=True)`` etc.) and additionally walks
        ``self._state`` -- the per-variable typed wrappers populated by
        :meth:`initialize` -- moving each one through
        :meth:`TensorWrapper.to`. Returns ``self`` so call chains like
        ``system = neml2.load_nonlinear_system(...).to('cuda')`` work the
        same way they do for ``nn.Module``.

        ``ModelNonlinearSystem`` is intentionally not an ``nn.Module``
        (it composes one rather than being one), so torch's
        ``Module.to`` semantics don't reach the system automatically --
        this method is the bridge.
        """
        self.model.to(*args, **kwargs)
        for name, tw in self._state.items():
            self._state[name] = tw.to(*args, **kwargs)
        return self

    def set_u(self, u: AssembledVector | SparseVector) -> None:
        if isinstance(u, SparseVector):
            u = u.assemble()
        self._state.update(u.disassemble().values)

    def set_u_from_group_raws(self, u_raws: list[torch.Tensor]) -> None:
        """Commit per-group raw unknown tensors (the solver boundary).

        Inverse of ``_vector_to_per_group_raws(self.u())``: each per-group raw
        is re-typed via :func:`~neml2.es.assembled.wrap_group_raw` using the
        unknown layout's structure, then committed through :meth:`set_u`. Used
        by the C++-backed Newton solver to write the converged iterate back
        into the system state.
        """
        tensors = [
            wrap_group_raw(raw, self.ulayout.groups[gi], self.ulayout.structure[gi], self.ulayout)
            for gi, raw in enumerate(u_raws)
        ]
        self.set_u(AssembledVector(self.ulayout, tensors))

    def set_g(self, g: AssembledVector | SparseVector) -> None:
        if isinstance(g, SparseVector):
            g = g.assemble()
        self._state.update(g.disassemble().values)

    def u(self) -> AssembledVector:
        return AssembledVector.from_dict(self.ulayout, self._state)

    def g(self) -> AssembledVector:
        return AssembledVector.from_dict(self.glayout, self._state)

    def to_sparse(
        self,
        u: Mapping[str, TensorWrapper],
        g: Mapping[str, TensorWrapper],
        sub_batch_ndim: Mapping[str, int] | None = None,
    ) -> tuple[SparseVector, SparseVector]:
        """Convert typed ``(u_dict, g_dict, sub_batch_ndim)`` into ``(u_sv, g_sv)``
        :class:`SparseVector` pair ready to pass to :meth:`initialize`.

        Use this helper at internal call sites that already build per-variable
        typed-wrapper dicts + a per-variable ``sub_batch_ndim`` count dict
        (the classic :class:`~neml2.models.common.ImplicitUpdate` / pyzag
        adapter shape). Per-variable ``sub_batch_shapes`` are derived from
        each wrapper's trailing batch dims so the resulting
        :class:`SparseVector` layouts encode the same sub-batch structure
        that the wrappers carry.

        Per CLAUDE.md rule 1, raw ``torch.Tensor`` inputs are rejected --
        wrap at the boundary first. For user-facing surfaces with no
        sub-batch (the typical test / notebook shape), construct
        ``SparseVector(system.ulayout, {name: typed_wrapper, ...})``
        directly instead -- no helper needed.
        """
        sbn = sub_batch_ndim if sub_batch_ndim is not None else {}

        def _shape_of(value: TensorWrapper, name: str) -> torch.Size:
            n = sbn.get(name, 0)
            if n <= 0:
                return torch.Size(())
            type_cls = self.model.input_spec[name]
            batch = _batch_shape(value, type_cls)
            if n > len(batch):
                raise ValueError(
                    f"sub_batch_ndim[{name!r}]={n} exceeds batch_ndim={len(batch)} "
                    f"of value with shape {tuple(value.shape)}"
                )
            return torch.Size(batch[-n:])

        u_sb = {name: _shape_of(value, name) for name, value in u.items()}
        g_sb = {name: _shape_of(value, name) for name, value in g.items()}
        u_layout = self.setup_ulayout(sub_batch_shapes=u_sb)
        g_layout = self.setup_glayout(sub_batch_shapes=g_sb)
        return SparseVector(u_layout, u), SparseVector(g_layout, g)

    def initialize(
        self,
        *,
        u: SparseVector,
        g: SparseVector,
        dyn_shape: tuple[int, ...] = (),
    ) -> None:
        """Set the state and per-variable layout from typed :class:`SparseVector` inputs.

        ``u`` and ``g`` carry their own :class:`AxisLayout`, which pins
        each variable's ``sub_batch_shape``. The system trusts these
        layouts directly -- no separate ``sub_batch_ndim`` dict is needed
        because the wrappers + layout already encode the same information
        consistently.

        Each :class:`SparseVector`'s ``values`` may pass typed wrappers
        (preferred per the wrapper-discipline rule) or raw ``torch.Tensor``
        (auto-wrapped with the input_spec's type for caller convenience at
        construction sites that haven't migrated yet -- these wrap-on-entry,
        not wrap-on-exit, so no metadata is lost).

        Call sites that have raw dicts + a ``sub_batch_ndim`` count dict
        should funnel through :meth:`to_sparse` to construct the typed
        :class:`SparseVector` pair.
        """
        # Per-variable sub_batch_shapes come straight from the typed layouts.
        # No re-derivation from value shapes -- the caller committed to a
        # layout when they built the SparseVector.
        self._sub_batch_shapes = {
            **dict(u.layout.sub_batch_shapes),
            **dict(g.layout.sub_batch_shapes),
        }

        # Normalize every typed wrapper against the layout-pinned sub_batch_ndim
        # so wrappers whose own sub_batch_ndim happens to disagree get
        # rewrapped to the layout's source of truth (predictor /
        # initial-unknown wrappers that lost their sub axis count upstream
        # would otherwise leak that loss into per-site unknowns like
        # dislocation_density).
        merged: dict[str, TensorWrapper] = {}
        for name, value in {**dict(g.values), **dict(u.values)}.items():
            sbn = len(self._sub_batch_shapes.get(name, ()))
            merged[name] = value if value.sub_batch_ndim == sbn else value.with_sub_batch_ndim(sbn)
        self._state = merged
        # Caller-declared system-wide dynamic-batch shape.
        self._dyn_shape: tuple[int, ...] = tuple(dyn_shape)

        self._dynamic_batch_ndim = {}
        for name, wrapper in self._state.items():
            type_cls = self.model.input_spec[name]
            batch = _batch_shape(wrapper, type_cls)
            sbn = len(self._sub_batch_shapes.get(name, ()))
            if sbn > 0 and sbn > len(batch):
                raise ValueError(
                    f"sub_batch_ndim[{name!r}]={sbn} exceeds batch_ndim={len(batch)} "
                    f"of value with shape {tuple(wrapper.shape)}"
                )
            self._dynamic_batch_ndim[name] = len(batch) - sbn

        # Mirror per-unknown sub-batch + dyn-ndim onto the matching residual.
        for ugroup, rgroup in zip(self.unknown_groups, self.residual_groups, strict=True):
            for uname, rname in zip(ugroup, rgroup, strict=True):
                if uname in self._sub_batch_shapes:
                    self._sub_batch_shapes[rname] = self._sub_batch_shapes[uname]
                self._dynamic_batch_ndim[rname] = self._dynamic_batch_ndim[uname]

        self._ulayout = self.setup_ulayout()
        self._glayout = self.setup_glayout()
        self._blayout = self.setup_blayout()

    def _call_model(self, v: ChainRuleDict | None = None) -> tuple[Any, ...]:
        missing = [name for name in self.model.input_spec if name not in self._state]
        if missing:
            raise KeyError(f"ModelNonlinearSystem state is missing inputs: {missing}")
        args = tuple(
            self.model.input_spec[name](
                self._state[name],
                sub_batch_ndim=len(self._sub_batch_shapes.get(name, ())),
            )
            for name in self.model.input_spec
        )
        if v is not None:
            from neml2 import log  # noqa: PLC0415

            if log.enabled("model", "debug"):
                for name, arg in zip(self.model.input_spec, args, strict=True):
                    log.emit(
                        "model",
                        "debug",
                        f"_call_model arg[{name!r}]: "
                        f"data.shape={tuple(arg.data.shape)} sub_ndim={arg.sub_batch_ndim} "
                        f"sub_state={arg.sub_batch_state} k_ndim={arg.k_ndim}",
                    )
                for name, seeds in v.items():
                    for leaf, s in seeds.items():
                        log.emit(
                            "model",
                            "debug",
                            f"_call_model v[{name!r}][{leaf!r}]: "
                            f"data.shape={tuple(s.data.shape)} sub_ndim={s.sub_batch_ndim} "
                            f"sub_state={s.sub_batch_state} k_ndim={s.k_ndim} "
                            f"k_state={s.k_state} k_pairing={s.k_pairing}",
                        )
        result = self.model(*args, v=v) if v is not None else self.model(*args)
        return result if isinstance(result, tuple) else (result,)

    def _identity_seed(self, names: list[str]) -> ChainRuleDict:
        """Per-(residual_group, unknown) expanded identity seeds.

        For each (input, residual_group) pair we emit ONE seed leaf
        whose compactness is the intersection of per-pair compactness
        across the group's residuals -- the chain-rule analog of
        :meth:`_preserved_labels_per_group`'s ``candidate − touched``
        rule applied at the seed boundary. This guarantees the chain
        rule is never finer-grained than the storage layer that
        consumes it (``AxisLayout.preserved_labels[gi]`` is the per-
        group set, so per-pair compactness only differs from per-
        group when some pair preserves an axis the rest of the group
        touches -- and that finer-grained gain is wiped out at
        ``_normalize_block_to_cell_canonical`` because the cell's
        canonical preserved set is empty).

        Per-group seeds collapse the redundant per-pair-within-a-group
        multiplier:

        * **Single-residual groups** (isoharden, taylor / mxpc):
          equivalent to the historical per-pair seeds (one residual per
          group → group == pair).
        * **Multi-residual groups, no preservation surviving the
          intersection** (chaboche2, scpcoup as configured): single
          seed per input at the group's full ``K`` -- equivalent to the
          v3-OLD per-input column seed and ~N_residuals× less chain-
          rule work than per-pair.
        * **Multi-residual groups with surviving preservation**: single
          compact seed per (input, group). Same storage end as per-pair
          on the diagonal pair, with no redundant full-K cross-pair
          seeds.

        Compactness rule (per-axis): an axis is group-compact iff NO
        residual in the group touches its label (``reduces ∪
        introduces``). Symmetric with how the storage layer derives
        ``preserved_labels[gi]``.

        Leaf keys use the ``"{col}:rgroup{gi}"`` format where ``gi`` is
        the residual group index. ``_build_group_block`` reads the
        tangent at ``v_out[row_name][f"{col_name}:rgroup{i_of_row_group}"]``.

        Bundling per-group leaves into a single ``model.forward(*args,
        v=seed)`` call works because ``apply_chain_rule`` propagates
        each leaf independently and K-equalizes per leaf, never across
        leaves -- the chain rule treats seed keys opaquely.

        Seeds for variables whose state value has fewer dynamic-batch
        axes than the system's max are LEFT-padded with size-1
        placeholders so the leading-K tangent contract ``(K, *dyn, *sub,
        *base)`` lines up position-wise across all variables. Without
        these placeholders, a tangent for a base-shape-only unknown
        would have shape ``(K, base)`` and collide positionally with a
        force primal's ``(dyn, base)`` during the typed-wrapper algebra
        -- torch's right-aligned broadcast would then misalign K with
        dyn.
        """
        # V2P-5 canonical seed, delegated to the shared builder so the native
        # eager path and the AOTI export wrappers cannot drift (the padding +
        # per-(input, residual_group) expansion live in one place now).
        return build_identity_seed(
            self._state,
            names,
            len(self.residual_groups),
            self.model.input_spec,
            self._sub_batch_shapes,
        )

    def _assemble_matrix(
        self,
        row_layout: AxisLayout,
        col_layout: AxisLayout,
        v_out: ChainRuleDict,
        like_by_row: Mapping[str, Tensor],
    ) -> AssembledMatrix:
        from neml2 import log  # noqa: PLC0415

        if log.enabled("model", "debug"):
            for r, leaves in v_out.items():
                for leaf, w in leaves.items():
                    log.emit(
                        "model",
                        "debug",
                        f"_assemble_matrix v_out[{r!r}][{leaf!r}]: "
                        f"data.shape={tuple(w.data.shape)} k_ndim={w.k_ndim} "
                        f"k_state={w.k_state} k_pairing={w.k_pairing} "
                        f"sub={w.sub_batch_ndim}/{w.sub_batch_state}",
                    )
        M = _build_block_matrix(self.model, row_layout, col_layout, v_out, like_by_row)
        return M

    def assemble(
        self,
        need_A: bool,
        need_B: bool,
        need_b: bool,
    ) -> tuple[AssembledMatrix | None, AssembledMatrix | None, AssembledVector | None]:
        seed_names: list[str] = []
        if need_A:
            seed_names.extend(self.unknown_names)
        if need_B:
            seed_names.extend(self.given_names)
        seed = self._identity_seed(seed_names) if seed_names else None
        result = self._call_model(v=seed)

        if seed is None:
            output_values = result
            v_out: ChainRuleDict = {}
        else:
            output_values = result[:-1]
            v_out = result[-1]

        # Keep model outputs as typed wrappers per rule 1; if a leaf
        # returned raw (shouldn't happen post-cleanup but tolerate),
        # rewrap with the declared output type.
        output_state: dict[str, TensorWrapper] = {}
        for name, value in zip(self.model.output_spec, output_values, strict=True):
            if isinstance(value, TensorWrapper):
                output_state[name] = value
            else:
                output_state[name] = self.model.output_spec[name](value)
        residual_values = {name: output_state[name] for name in self.residual_names}
        like_by_row = {
            name: _flatten_base(residual_values[name], self.model.output_spec[name])
            for name in self.residual_names
        }

        A = (
            self._assemble_matrix(self.blayout, self.ulayout, v_out, like_by_row)
            if need_A
            else None
        )
        B = (
            self._assemble_matrix(self.blayout, self.glayout, v_out, like_by_row)
            if need_B
            else None
        )
        b = -AssembledVector.from_dict(self.blayout, residual_values) if need_b else None
        return A, B, b


__all__ = ["LinearSystem", "NonlinearSystem", "ModelNonlinearSystem"]
