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

"""pyzag ↔ NEML2 interface.

Wires NEML2's :mod:`neml2.equation_systems` runtime (``AxisLayout``,
``AssembledVector``, ``AssembledMatrix``) into the pyzag
``NonlinearRecursiveFunction`` protocol. The underlying NEML2 ``Model``
is itself an :class:`torch.nn.Module`, so ``named_parameters()`` works
without any wrapping.
"""

from __future__ import annotations

import math

import torch
from pyzag import nonlinear

from neml2 import load_nonlinear_system  # noqa: F401 — re-export for users
from neml2.equation_systems import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    ModelNonlinearSystem,
)


def lag_order(var: str) -> tuple[str, int]:
    """Split a variable name into its base name and lag order.

    ``var`` is either ``"name"`` (lag 0) or ``"name~n"`` (lag ``n``). Used by
    the interface to identify *old-state* / *old-force* variables — pyzag
    wants a contiguous-in-time state and forces tensor, while the NEML2
    nonlinear system keeps current values on the unknown axis and old values
    on the given axis with a ``~1`` suffix.
    """
    tokens = var.split("~")
    if len(tokens) == 1:
        return tokens[0], 0
    if len(tokens) == 2:
        return tokens[0], int(tokens[1])
    raise ValueError(
        f"Variable {var!r} has invalid format; expected 'name' or 'name~n' where n is an integer"
    )


def change_lag_order(var: str, new_order: int) -> str:
    """Re-tag a variable name to a different lag order. Inverse of :func:`lag_order`."""
    base, _ = lag_order(var)
    return f"{base}~{new_order}" if new_order != 0 else base


def _filter_layout_by_lag(
    layout: AxisLayout,
    order: int,
    new_order: int | None = None,
    *,
    keep_only_if: set[str] | None = None,
) -> AxisLayout:
    """Single-group sub-layout containing only variables whose lag matches ``order``.

    Optionally renames the surviving variables to ``new_order`` (so old-state
    seeded as ``"x~1"`` can be re-emitted as ``"x"`` for the pyzag-side
    representation).

    ``keep_only_if`` further filters the survivors by the *post-rename* name:
    only entries whose new name is in the set are kept. Used to drop
    rate-like unknowns from snlayout when the underlying NEML2 model has no
    matching ``~1`` glayout entry (rate variables solved instantaneously
    each step, e.g. ``gamma_rate_ri`` for rate-independent flow).

    Native ``AxisLayout`` is groups-based; pyzag's filter is name-based and
    always single-group, so we collapse to one group on the way out. The
    surviving names keep their ``specs`` / ``sub_batch_shapes`` from the
    parent layout.
    """
    survivors: list[str] = []
    new_specs: dict = {}
    new_sub: dict = {}
    for name in layout.vars():
        base, lag = lag_order(name)
        if lag != order:
            continue
        suffix = f"~{new_order}" if new_order is not None and new_order != 0 else ""
        new_name = f"{base}{suffix}"
        if keep_only_if is not None and new_name not in keep_only_if:
            continue
        survivors.append(new_name)
        new_specs[new_name] = layout.type_of(name)
        new_sub[new_name] = layout.sub_batch_shape(name)
    return AxisLayout([survivors], new_specs, new_sub)


class NEML2PyzagModel(nonlinear.NonlinearRecursiveFunction):
    """Wrap a native NEML2 :class:`NonlinearSystem` as a pyzag
    ``NonlinearRecursiveFunction``.

    Args:
        sys: the native NEML2 nonlinear system to wrap.

    Keyword Args:
        exclude_parameters (list of str): NEML2 parameters to *not* mirror
            as torch parameters on the wrapper (and therefore not optimize
            against).

    Additional ``args`` and ``kwargs`` are forwarded to
    :class:`torch.nn.Module` verbatim.
    """

    def __init__(
        self,
        sys: ModelNonlinearSystem,
        *args,
        exclude_parameters: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(sys, ModelNonlinearSystem):
            raise TypeError(
                f"sys should be a neml2.equation_systems.ModelNonlinearSystem, "
                f"instead got {type(sys)}. Use neml2.load_nonlinear_system "
                "or load_input(path).get_equation_system(name) to obtain one."
            )

        # ``self.sys`` is not an ``nn.Module`` — plain attribute assignment is fine.
        # ``sys.model`` IS an ``nn.Module``, and the default
        # ``nn.Module.__setattr__`` would silently auto-register it as a
        # submodule. That doubles every parameter under both
        # ``wrapper.<flat>`` (our mirror) and ``wrapper.model.<dotted>``
        # (the underlying), so any code iterating ``solver.named_parameters()``
        # sees each leaf twice. After our parameter-swap binds the underlying
        # ``_parameters[leaf]`` to the wrapper's Parameter, the two entries
        # become the same object — pyzag's adjoint accumulator then writes
        # the gradient through both positions, which interferes with itself
        # for at least the ``offset`` case (off-by-a-sign / cancellation).
        # ``object.__setattr__`` bypasses ``nn.Module.__setattr__`` and leaves
        # ``self.model`` as a plain instance attribute.
        self.sys = sys
        object.__setattr__(self, "model", sys.model)
        self._lookback = 1

        self._setup_maps()
        self._check_model()
        self._setup_parameters(exclude_parameters if exclude_parameters is not None else [])

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
        return sum(math.prod(self.slayout.specs[v].BASE_SHAPE or (1,)) for v in self.slayout.vars())

    @property
    def nforce(self) -> int:
        return sum(math.prod(self.flayout.specs[v].BASE_SHAPE or (1,)) for v in self.flayout.vars())

    def forward(
        self, state: torch.Tensor, forces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the residual and stacked Jacobians for one pyzag block step.

        Args:
            state: flat tensor with the current+previous state, shape
                ``(n_block + lookback, ..., nstate)``.
            forces: flat tensor with the current+previous forces, shape
                ``(n_block + lookback, ..., nforce)``.
        """
        # Sync the wrapper-owned torch.nn.Parameters back into the underlying
        # NEML2 model (the wrapper is what torch reparameterization decorates,
        # the underlying model is what actually evaluates).
        self._update_parameter_values()

        # Disassemble the pyzag-shaped state/forces into the per-variable
        # current/old dicts the native NonlinearSystem consumes.
        self._update_sys(state, forces)

        # Native assembly:
        #   A := dr/du   (Jacobian w.r.t. unknowns)
        #   B := dr/dg   (Jacobian w.r.t. given variables; includes old state,
        #                forces, old forces)
        #   b := -r
        A, B, b = self.sys.A_and_B_and_b()
        return self._adapt_for_pyzag(A, B, b)

    # ── one-time setup ───────────────────────────────────────────────────────

    def _check_model(self):
        """Sanity-check the model layout: every old-state (or old-force)
        variable must have a corresponding current variable."""
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
        """Build the layouts and the snlayout→glayout name map used by
        :meth:`_adapt_for_pyzag` to extract the old-state Jacobian columns
        from $B$.

        Native ``ModelNonlinearSystem`` distinguishes between unknowns ($u$)
        and given variables ($g$); pyzag wants a contiguous-in-time
        ``(state, forces)`` representation where current+previous state both
        live in the state tensor and current+previous forces both live in the
        forces tensor. So old unknowns and old forces (lag 1) need to be
        moved from $g$ back into a state/force lane on the pyzag side.
        """
        # `g` from native includes old unknowns, current forces, and old forces.
        glayout = self.sys.glayout
        gvars = glayout.vars()
        gvar_set = set(gvars)

        # Layouts the wrapper exposes to pyzag.
        self.rlayout = self.sys.blayout  # residuals
        self.slayout = self.sys.ulayout  # current state (the unknowns)
        # snlayout / fnlayout = old state / forces. Built from ulayout /
        # glayout entries with their lag retagged ``~1``. The renames stay
        # in step with the pyzag-side state and forces tensors, which carry
        # one slot per unknown / per force regardless of whether the
        # underlying NEML2 model actually requests a previous value for
        # that variable. The previous value is *fed into the model* (in
        # ``_update_sys``) only when the matching ``~1`` is in glayout —
        # see ``_relevant_old_state`` / ``_relevant_old_forces`` for the
        # filtered names used at evaluation time. Rate-like unknowns (e.g.
        # ``gamma_rate_ri`` for rate-independent flow complementarity) and
        # steady forces (e.g. ``temperature``) thus get a snlayout /
        # fnlayout slot but no glayout-side push.
        self.snlayout = _filter_layout_by_lag(self.slayout, 0, new_order=1)
        self.flayout = _filter_layout_by_lag(glayout, 0)
        self.fnlayout = _filter_layout_by_lag(glayout, 0, new_order=1)
        # Names whose lag-1 form is consumed by the model — used by
        # ``_update_sys`` to filter the old-state / old-force pushes.
        self._sn_in_glayout: set[str] = {v for v in self.snlayout.vars() if v in gvar_set}
        self._fn_in_glayout: set[str] = {v for v in self.fnlayout.vars() if v in gvar_set}

        # Surface names used by NEML2 callers (matches pre-port behavior).
        self.svars = self.slayout.vars()
        self.fvars = self.flayout.vars()

        # Per-snvar lookup of the matching glayout name (so we can pick the
        # right B column when assembling Jn). Old-state names in snlayout
        # were re-tagged to lag 1, matching their original spelling in
        # glayout — the lookup is straightforward.
        self._sn_to_g_name: dict[str, str] = {}
        for snv in self.snlayout.vars():
            if snv in gvars:
                self._sn_to_g_name[snv] = snv

    def _setup_parameters(self, exclude_parameters: list[str]):
        """Mirror each HIT-named NEML2 model parameter as a
        ``torch.nn.Parameter`` on the wrapper with a flat ``<hit_name>_<leaf>``
        key (e.g. ``elasticity_E``, ``flow_rate_eta``, ``Eerate_weight_0``).

        Why not just expose ``self.sys.model.named_parameters()`` verbatim?

        - Native parameters live under dotted paths that include framework
          boilerplate (``normality._inner.yield.sy``). pyzag callers (and the
          test contract) identify parameters by their HIT block
          name (``yield_sy``), so we collapse the dotted path to the deepest
          enclosing module that carries a ``_hit_name`` (set by the factory)
          plus the parameter's storage attribute name.

        - pyzag's reparametrization (:mod:`pyzag.reparametrization`) hooks
          ``torch.nn.utils.parametrize`` onto whichever module owns the
          parameter. The wrapper owning the (mirrored) parameter lets the
          user reparametrize without reaching into the NEML2 model.

        The storage attribute name **is** the user-facing identifier — leaf
        models on the native side are responsible for picking sane
        attributes (``K``, ``sy``, ``eta``, ``weight``, ``offset``).
        Anything that starts with an underscore is framework plumbing and
        should be renamed at the source rather than special-cased here.

        Each forward, :meth:`_update_parameter_values` copies the wrapper's
        (possibly reparameterized) value back into the underlying model's
        parameter so the model evaluation sees it.
        """
        # Build a {hit_name → submodule} index once. ``modules`` walks
        # the whole submodule tree in depth-first preorder so the
        # deepest-matching HIT name wins when a parameter sits inside a
        # nested wrapper (e.g. Normality's ``_inner.yield`` keeps the
        # ``yield`` HIT name, not the outer ``normality``).
        hit_modules: dict[int, str] = {}
        for module in self.sys.model.modules():
            hit_name = getattr(module, "_hit_name", None)
            if hit_name and hit_name.isidentifier():
                hit_modules[id(module)] = hit_name

        # ``_param_targets`` is the runtime sync table: wrapper-side flat
        # name → (owning submodule, leaf attribute name on that submodule).
        # ``_update_parameter_values`` walks this to push values back.
        self.parameter_names: list[str] = []
        self._param_targets: dict[str, tuple[torch.nn.Module, str]] = {}
        registered: set[str] = set()

        for module in self.sys.model.modules():
            mod_hit = hit_modules.get(id(module))
            if mod_hit is None:
                continue
            for leaf_name, param in module.named_parameters(recurse=False):
                flat = f"{mod_hit}_{leaf_name}"
                if flat in exclude_parameters or flat in registered:
                    # ``in exclude_parameters`` handles the user's opt-out;
                    # ``in registered`` guards against accidental collisions
                    # if two HIT blocks share a name + parameter (pathological,
                    # but cheap to defend against).
                    continue
                registered.add(flat)
                self.parameter_names.append(flat)
                param.requires_grad_(True)
                self.register_parameter(flat, torch.nn.Parameter(param.detach().clone()))
                self._param_targets[flat] = (module, leaf_name)

    def _update_parameter_values(self):
        """Bind the wrapper's (possibly reparameterized) parameter tensors
        onto the underlying NEML2 model so the next forward sees them and
        gradients flow back to the wrapper.

        Key point: ``module._parameters[leaf].data = new_value`` would *only*
        copy values — the underlying model's parameter stays a fresh autograd
        leaf, so a backward pass through the model's output gives the
        gradient to that internal Parameter, not to ``self.<pname>``. The
        wrapper's mirror parameter then sees a zero ``.grad`` and the
        adjoint method returns garbage.

        For *plain* (non-parametrized) leaves we therefore *swap* the
        underlying ``nn.Parameter`` for the wrapper's own ``nn.Parameter``
        (same object). The model's normal attribute lookup still returns a
        Parameter through ``_parameters``; the native
        ``Model.__getattr__`` re-wraps it into the registered typed class;
        and any backward through the model's output deposits ``.grad`` on
        the wrapper's Parameter directly.

        For *reparametrized* leaves (``torch.nn.utils.parametrize``
        replaced ``self.<pname>`` with a transform-producing property),
        ``getattr(self, pname)`` returns a fresh non-leaf tensor each call.
        That can't be plugged back into ``_parameters`` (a Parameter must
        be a leaf), so we fall back to ``object.__setattr__`` of a
        typed-wrapped tensor in the instance dict — same mechanism as
        before but only for the reparametrized case.
        """
        for pname in self.parameter_names:
            # ``getattr(self, pname)`` triggers any registered parametrize
            # transform — that's the natural-scale tensor the model wants.
            new_value = getattr(self, pname)
            module, leaf = self._param_targets[pname]
            typed_cls = getattr(module, "_typed_storage_classes", {}).get(leaf)
            sbn = getattr(module, "_typed_storage_sub_batch_ndim", {}).get(leaf, 0)

            if isinstance(new_value, torch.nn.Parameter):
                # Plain leaf: swap. Drop any instance-dict shim left from a
                # previous reparametrized cycle so __getattr__ resolves to
                # _parameters[leaf] again.
                module.__dict__.pop(leaf, None)
                module._parameters[leaf] = new_value
            else:
                # Reparametrized leaf: ``new_value`` is a non-leaf, can't go
                # in ``_parameters``. Bind as instance attr (wrapped to the
                # typed class so ``_get_param``'s isinstance check passes).
                module._parameters.pop(leaf, None)
                bound = (
                    typed_cls(new_value, sub_batch_ndim=sbn) if typed_cls is not None else new_value
                )
                object.__setattr__(module, leaf, bound)

    # ── per-forward plumbing ─────────────────────────────────────────────────

    def _update_sys(self, state: torch.Tensor, forces: torch.Tensor):
        """Disassemble the pyzag-shaped state/forces tensors into the
        per-variable current/old dicts the native NonlinearSystem consumes.

        Native typed-wrapper convention is dynamic batch axes leading; pyzag
        gives us ``(n_block + lookback, ..., nstate|nforce)`` where dim 0 is
        the time axis we slice with ``lookback``.
        """
        assert state.shape[:-1] == forces.shape[:-1]

        # Slice along the time axis to get current (``[lookback:]``) and
        # previous (``[:-lookback]``) windows.
        state_now = state[self.lookback :]
        state_prev = state[: -self.lookback]
        forces_now = forces[self.lookback :]
        forces_prev = forces[: -self.lookback]

        # Per-variable splits along the trailing axis. Each slice is a flat
        # ``(*batch, var_size)`` chunk that the native AssembledVector
        # disassemble path expects.
        u = self._split_by_layout(state_now, self.slayout)
        sn = self._split_by_layout(state_prev, self.snlayout, rename_to_lag=0)
        f = self._split_by_layout(forces_now, self.flayout)
        fn = self._split_by_layout(forces_prev, self.fnlayout, rename_to_lag=0)

        # Sub-batch ndim per variable. The pyzag interface only handles
        # variables without sub-batch (sub-batch is a per-crystal CP concept
        # and pyzag's training surface predates it), so always 0.
        sub_batch_ndim: dict[str, int] = {}
        for name in u:
            sub_batch_ndim[name] = 0
        for name in sn:
            sub_batch_ndim[name] = 0
        for name in f:
            sub_batch_ndim[name] = 0
        for name in fn:
            sub_batch_ndim[name] = 0

        # The underlying NEML2 NonlinearSystem doesn't know the
        # rename_to_lag=0 trick — it expects the original lag-tagged names
        # ('stress~1' etc.) for old state and old forces. Only push entries
        # the model actually requests: rate-like unknowns (gamma_rate_ri)
        # and steady forces (temperature) have a pyzag-side state/force
        # slot but no model-side ``~1`` input, so we'd KeyError if we tried
        # to push them.
        g_state: dict[str, torch.Tensor] = {}
        for sn_name, tensor in sn.items():
            sn_full = change_lag_order(sn_name, 1)
            if sn_full in self._sn_in_glayout:
                g_state[sn_full] = tensor
        for f_name, tensor in f.items():
            g_state[f_name] = tensor
        for fn_name, tensor in fn.items():
            fn_full = change_lag_order(fn_name, 1)
            if fn_full in self._fn_in_glayout:
                g_state[fn_full] = tensor

        # Compute the system-wide dyn_shape from the state tensor (drop the
        # trailing var_size dim). All inputs share this batch shape because
        # we just sliced state and forces alike.
        dyn_shape = tuple(state_now.shape[:-1])
        self.sys.initialize(u=u, g=g_state, sub_batch_ndim=sub_batch_ndim, dyn_shape=dyn_shape)

    def _split_by_layout(
        self,
        flat: torch.Tensor,
        layout: AxisLayout,
        rename_to_lag: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Split a flat ``(*batch, nflat)`` tensor into a per-variable dict
        keyed by the layout's variable names, optionally re-tagging each
        name to a different lag order.

        Variable sizes come from ``layout.var_size(name)``; offsets are the
        running sum. Each per-variable slice is reshaped from ``(*batch,
        var_size)`` back to ``(*batch, *BASE_SHAPE)`` so the typed wrappers
        the model uses can read it without further shape gymnastics.
        """
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for name in layout.vars():
            type_cls = layout.type_of(name)
            size = layout.var_size(name)
            slab = flat[..., offset : offset + size]
            if type_cls.BASE_NDIM == 0:
                # Scalar: drop the trailing size-1 axis.
                slab = slab.squeeze(-1)
            elif type_cls.BASE_NDIM > 1:
                slab = slab.reshape(*slab.shape[:-1], *type_cls.BASE_SHAPE)
            # For BASE_NDIM == 1 the shape is already correct.
            out_name = change_lag_order(name, rename_to_lag) if rename_to_lag is not None else name
            out[out_name] = slab
            offset += size
        return out

    def _adapt_for_pyzag(
        self, A: AssembledMatrix, B: AssembledMatrix, b: AssembledVector
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the residual and stacked Jacobians pyzag expects.

        - Native:
            * $A$ = ``dr/du`` (unknown side)
            * $B$ = ``dr/dg`` (given side; includes old state, forces, old forces)
            * $b$ = ``-r``
        - pyzag wants:
            * $r$ = residual (so flip the sign on $b$)
            * $J$ = ``dr/du`` (the current-state Jacobian)
            * ``Jn`` = ``dr/du_old`` (the old-state Jacobian, extracted from $B$)
            * Return value: ``(r, torch.stack([Jn, J]))``

        Jn is built by selecting the subset of B's per-(row, col) blocks that
        correspond to the snlayout vars (via :meth:`AssembledMatrix.select_blocks`,
        which replaces the C++ SparseMatrix round-trip).
        """
        r = -b.tensors[0]
        J = A.tensors[0][0]

        # Pull the per-(row, col) chain-rule blocks out of B and pick the
        # ones whose col_var matches a snlayout var (by g-name lookup).
        B_blocks = B.disassemble()
        picked: dict[str, dict[str, torch.Tensor]] = {}
        for row_name, cols in B_blocks.items():
            picked[row_name] = {}
            for sn_name in self.snlayout.vars():
                g_name = self._sn_to_g_name.get(sn_name)
                if g_name is not None and g_name in cols:
                    picked[row_name][sn_name] = cols[g_name]

        Jn = AssembledMatrix.select_blocks(self.rlayout, self.snlayout, picked).tensors[0][0]

        # Broadcast both Jacobians to the residual's batch shape so pyzag's
        # downstream stack sees consistent leading dims.
        target_batch = r.shape[:-1]
        J = J.expand(*target_batch, *J.shape[-2:])
        Jn = Jn.expand(*target_batch, *Jn.shape[-2:])

        assert J.shape[-1] == J.shape[-2], f"J must be square, got shape {tuple(J.shape)}"
        assert Jn.shape[-1] == Jn.shape[-2], f"Jn must be square, got shape {tuple(Jn.shape)}"

        return r, torch.stack([Jn, J])
