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

"""Python-native mirror of the C++ ``BilinearTraction`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import Scalar, Vec, gt, lt, vec_from_scalars, where


@register_native("BilinearTraction")
class BilinearTraction(Model):
    r"""Bilinear cohesive-zone traction with internal damage state. Computes the
    bilinear damage variable from $(\delta_m, \delta_c, \delta_f)$
    (effective separation, critical / damage-onset separation, full / failure
    separation), caps it for irreversibility against the previous-step value
    (auto-declared via `history_name`), and assembles
    $T_n = K(1-d)\delta_n^+ + K\delta_n^-,\;T_{si} = K(1-d)\delta_{si}$. Damage is
    exposed as a secondary output for diagnostics; the irreversibility cap is
    internal to this model and does not require an external
    `IrreversibleScalar`.
    """

    hit = HitSchema(
        input("effective_separation", Scalar, "Effective separation"),
        input(
            "normal_separation",
            Scalar,
            "Normal separation (typically the Macaulay-positive part of the normal jump)",
        ),
        input(
            "normal_penetration",
            Scalar,
            "Optional normal penetration (typically the Macaulay-negative part of the "
            "normal jump). When supplied, K times this is added to the normal traction "
            "as a penalty term resisting interpenetration.",
            default=None,
            attr="_dn_pen_name",
        ),
        input("tangential_separation_1", Scalar, "First tangential separation"),
        input("tangential_separation_2", Scalar, "Second tangential separation"),
        output("traction", Vec, "Traction Vec"),
        output(
            "damage",
            Scalar,
            "Damage scalar (current step, irreversibility-capped)",
        ),
        # Previous-step damage is auto-declared by appending the ``~1`` history
        # suffix to the resolved ``damage`` output name (matches the C++
        # ``history_name(_d.name(), /*nstep=*/1)`` plumbing). No HIT knob.
        derived_input("damage", Scalar, attr="_d_old_name", suffix="~1"),
        parameter(
            "penalty_stiffness",
            Scalar,
            "Penalty stiffness",
            attr="K",
        ),
        parameter(
            "critical_separation",
            Scalar,
            "Critical (damage-onset) separation. May be wired to an upstream "
            "`CamanhoDavilaCriticalSeparation` (nonlinear-capable).",
            attr="delta_c",
            allow_nonlinear=True,
        ),
        parameter(
            "full_separation",
            Scalar,
            "Full (failure) separation. May be wired to an upstream "
            "`BenzeggaghKenaneFullSeparation` or `PowerLawFullSeparation` "
            "(nonlinear-capable).",
            attr="delta_f",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the three Scalar parameters via
    # ``declare_typed_parameter``. The optional ``normal_penetration`` input
    # resolves to a HIT name (string) when the user supplies it, ``None``
    # otherwise; the ``derived_input("damage", ..., suffix="~1")`` resolves
    # ``self._d_old_name`` to ``<resolved damage>~1``.
    K: Scalar
    delta_c: Scalar
    delta_f: Scalar
    _dn_pen_name: str | None
    _d_old_name: str

    def forward(  # type: ignore[override]
        self,
        *args,
        v: ChainRuleDict | None = None,
        **_,
    ):
        # ``input_spec`` order: structural inputs first (declaration order from
        # the schema, minus optional inputs that HIT omitted), then promoted
        # nl-parameters appended in declaration order. Split accordingly.
        names = list(self.input_spec)
        if len(args) != len(names):
            raise AssertionError(
                f"BilinearTraction.forward: got {len(args)} args, expected {len(names)}"
            )
        n_nl = len(self._nl_params)
        n_struct = len(names) - n_nl
        inputs, nl_params = args[:n_struct], args[n_struct:]
        struct_names = names[:n_struct]
        bound = dict(zip(struct_names, inputs, strict=True))

        # ``input_spec`` already holds the *resolved* (post-rename) names, so
        # walk the rename map in reverse to pick out each canonical role.
        renames = getattr(self, "_var_renames", {})

        def _resolved(canon: str) -> str:
            return renames.get(canon, canon)

        delta_m = bound[_resolved("effective_separation")]
        dn_sep = bound[_resolved("normal_separation")]
        ds1 = bound[_resolved("tangential_separation_1")]
        ds2 = bound[_resolved("tangential_separation_2")]
        d_old = bound[self._d_old_name]
        dn_pen_name = self._dn_pen_name
        dn_pen = bound[dn_pen_name] if dn_pen_name is not None and dn_pen_name in bound else None

        # Nonlinear-capable parameters: read via ``_get_param`` so the same
        # code path works whether ``delta_c`` / ``delta_f`` are static
        # parameters or promoted to runtime nl inputs (mode 3/4).
        # ``penalty_stiffness`` is always static (allow_nonlinear=False).
        # The ``_get_param`` lookup key is the schema ``attr`` (which is the
        # ``ctor_name`` used by ``declare_typed_parameter`` in pass 2).
        K = self._get_param("K", nl_params, Scalar)
        delta_c = self._get_param("delta_c", nl_params, Scalar)
        delta_f = self._get_param("delta_f", nl_params, Scalar)

        # -------- Bilinear damage trial value from (delta_m, delta_c, delta_f).
        # Detached masks for ``where()`` (matches C++ ``.detach()``).
        one = Scalar.from_value(1.0, like=delta_m)
        zero = Scalar.from_value(0.0, like=delta_m)
        df_minus_di = delta_f - delta_c
        # ``safe_df_minus_di`` prevents 0/0 at the degenerate ``delta_c ==
        # delta_f`` edge before the ``where`` mask zeros the contribution.
        df_minus_di_pos = gt(df_minus_di, 0.0)
        safe_df_minus_di = where(df_minus_di_pos, df_minus_di, one)
        bilinear_d = delta_f * (delta_m - delta_c) / (delta_m * safe_df_minus_di)
        dm_lt_init = lt(delta_m, delta_c)
        dm_lt_final = lt(delta_m, delta_f)
        # Interior: delta_c < delta_m < delta_f.
        dm_gt_init_data = (delta_m.data > delta_c.data).detach()
        dm_lt_final_data = (delta_m.data < delta_f.data).detach()
        interior_data = (dm_gt_init_data & dm_lt_final_data).detach()
        interior = Scalar(interior_data, sub_batch_ndim=delta_m.sub_batch_ndim)
        d_trial = where(dm_lt_init, zero, where(dm_lt_final, bilinear_d, one))

        # -------- Irreversibility cap: damage = max(d_trial, d_old).
        advance = gt(d_trial, d_old)
        d = where(advance, d_trial, d_old)

        # -------- Assemble traction.
        active_scale = K * (1.0 - d)
        T_n = active_scale * dn_sep
        if dn_pen is not None:
            T_n = T_n + K * dn_pen
        T_s1 = active_scale * ds1
        T_s2 = active_scale * ds2
        T = vec_from_scalars(T_n, T_s1, T_s2)

        if v is None:
            if dn_pen is None:
                # Two outputs: (traction, damage).
                return T, d
            return T, d

        # -------- D-062 pushforward.
        #
        # Trial-value partials in the interior (zero outside):
        #   d(d_t)/d(delta_m) =  delta_f * delta_c / (delta_m^2 (delta_f - delta_c))
        #   d(d_t)/d(delta_c) =  delta_f * (delta_m - delta_f) / (delta_m * (delta_f - delta_c)^2)
        #   d(d_t)/d(delta_f) = -delta_c * (delta_m - delta_c) / (delta_m * (delta_f - delta_c)^2)
        # Irreversibility cap freezes them when d_trial does not advance.
        inv_dm = 1.0 / delta_m
        inv_diff = 1.0 / safe_df_minus_di
        inv_diff_sq = inv_diff * inv_diff

        dt_ddm_int = delta_f * delta_c * inv_dm * inv_dm * inv_diff
        dt_ddm = where(interior, dt_ddm_int, zero)
        # After max(d_t, d_old): partial collapses to 0 on the frozen branch.
        dd_ddm = where(advance, dt_ddm, zero)
        dd_dd_old = where(advance, zero, one)

        # ``-K`` prefactor of d(T)/d(damage); the per-component Vec coefficient
        # is ``(-K * dn_sep, -K * ds1, -K * ds2)`` (zero in the penetration
        # column because the K*dn_pen penalty does not depend on damage).
        neg_K = -K
        dT_dd_n = neg_K * dn_sep
        dT_dd_s1 = neg_K * ds1
        dT_dd_s2 = neg_K * ds2

        # ----- Actions for the damage output. Inputs not listed (dn_sep,
        # dn_pen, ds1, ds2) push forward to structural zero.
        damage_actions: dict[str, ChainRuleAction] = {
            "effective_separation": lambda V, c=dd_ddm: c * V,
            self._d_old_name: lambda V, c=dd_dd_old: c * V,
        }

        # ----- Actions for the traction output. Each direct-jump partial is a
        # Vec with the active_scale (or K) on one component and zero on the
        # other two; damage-mediated partials route through ``-K*[jumps] *
        # dd/d(scalar)``.
        def _traction_via_damage(dd_dscalar: Scalar):
            # d(T)/d(scalar via damage) = (-K * jumps) * dd/d(scalar)
            def action(V: Scalar) -> Vec:
                Vs = V
                return vec_from_scalars(
                    dT_dd_n * dd_dscalar * Vs,
                    dT_dd_s1 * dd_dscalar * Vs,
                    dT_dd_s2 * dd_dscalar * Vs,
                )

            return action

        def _dn_sep_action(V: Scalar) -> Vec:
            Vs = V
            return vec_from_scalars(active_scale * Vs, zero * Vs, zero * Vs)

        def _ds1_action(V: Scalar) -> Vec:
            Vs = V
            return vec_from_scalars(zero * Vs, active_scale * Vs, zero * Vs)

        def _ds2_action(V: Scalar) -> Vec:
            Vs = V
            return vec_from_scalars(zero * Vs, zero * Vs, active_scale * Vs)

        def _dn_pen_action(V: Scalar) -> Vec:
            Vs = V
            return vec_from_scalars(K * Vs, zero * Vs, zero * Vs)

        traction_actions: dict[str, ChainRuleAction] = {
            "effective_separation": _traction_via_damage(dd_ddm),
            self._d_old_name: _traction_via_damage(dd_dd_old),
            "normal_separation": _dn_sep_action,
            "tangential_separation_1": _ds1_action,
            "tangential_separation_2": _ds2_action,
        }
        if dn_pen_name is not None:
            traction_actions[dn_pen_name] = _dn_pen_action

        # ----- Nonlinear-parameter contributions (delta_c, delta_f) — only
        # emit when the user wired them to an upstream variable (allow_nonlinear
        # promoted them into the input set). When supplied as literals the
        # promoted-input name is absent from ``v`` and the action is silently
        # skipped by ``apply_chain_rule``.
        delta_c_nlp = self._nl_params.get("delta_c")
        if delta_c_nlp is not None:
            dt_dinit_int = delta_f * (delta_m - delta_f) * inv_dm * inv_diff_sq
            dt_dinit = where(interior, dt_dinit_int, zero)
            dd_dinit = where(advance, dt_dinit, zero)
            damage_actions[delta_c_nlp.input_name] = lambda V, c=dd_dinit: c * V
            traction_actions[delta_c_nlp.input_name] = _traction_via_damage(dd_dinit)

        delta_f_nlp = self._nl_params.get("delta_f")
        if delta_f_nlp is not None:
            dt_dfinal_int = -delta_c * (delta_m - delta_c) * inv_dm * inv_diff_sq
            dt_dfinal = where(interior, dt_dfinal_int, zero)
            dd_dfinal = where(advance, dt_dfinal, zero)
            damage_actions[delta_f_nlp.input_name] = lambda V, c=dd_dfinal: c * V
            traction_actions[delta_f_nlp.input_name] = _traction_via_damage(dd_dfinal)

        v_T = self.apply_chain_rule(v, "traction", traction_actions, output=T)
        v_d = self.apply_chain_rule(v, "damage", damage_actions, output=d)
        return T, d, {**v_T, **v_d}


__all__ = ["BilinearTraction"]
