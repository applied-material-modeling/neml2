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

"""Python-native mirror of the C++ ``SalehaniIraniTraction`` model."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import Scalar, Vec, exp, gt, vec_from_scalars, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SalehaniIraniTraction")
class SalehaniIraniTraction(Model):
    r"""3D coupled exponential cohesive law of Salehani & Irani with internal damage state.
    Computes $d_\text{trial} = 1 - \exp(-x)$ where $x = b_n + b_{s1}^2 + b_{s2}^2$, caps it for
    irreversibility against the previous-step value
    (auto-declared via `history_name`), and assembles $T_i = a_i b_i (1 - d)$.
    For monotonic loading this is exactly equivalent to the original $T_i = a_i b_i \exp(-x)$; under
    unloading the damage cap freezes the softness at its historical
    peak. The internal tangential characteristic length is $\sqrt{2}\,\delta_{u0,t}$. If
    `normal_penetration` is supplied, $K_\text{pen}\,\delta_n^\text{pen}$
    is added to $T_n$ (`penalty_stiffness` becomes required).
    """

    hit = HitSchema(
        input(
            "normal_separation",
            Scalar,
            "Normal separation (typically the Macaulay-positive part of the normal jump)",
        ),
        input(
            "normal_penetration",
            Scalar,
            "Optional normal penetration. When supplied, the penalty stiffness times this "
            "is added to the normal traction as a penalty term resisting interpenetration. "
            "Requires `penalty_stiffness` to be supplied as well.",
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
            "normal_characteristic_length",
            Scalar,
            "Normal characteristic length (raw user input)",
            attr="delta_u0_n",
        ),
        parameter(
            "tangential_characteristic_length",
            Scalar,
            "Tangential characteristic length (raw user input; the internal value is "
            "sqrt(2) times this)",
            attr="delta_u0_t",
        ),
        parameter(
            "normal_strength",
            Scalar,
            "Normal strength (peak normal traction)",
            attr="Tmax_n",
        ),
        parameter(
            "shear_strength",
            Scalar,
            "Shear strength (peak shear traction)",
            attr="Tmax_t",
        ),
        # Optional with default "0". The C++ asserts this is supplied iff
        # ``normal_penetration`` is; here we let the schema default carry the
        # zero value and simply skip the penalty term when ``normal_penetration``
        # is not wired.
        parameter(
            "penalty_stiffness",
            Scalar,
            "Penalty stiffness used to resist interpenetration. Required when "
            "`normal_penetration` is supplied; ignored otherwise.",
            attr="Kpen",
            default="0",
        ),
    )

    # ``from_hit`` auto-declares the five Scalar parameters via
    # ``declare_typed_parameter``. The optional ``normal_penetration`` input
    # resolves to a HIT name (string) when the user supplies it, ``None``
    # otherwise; the ``derived_input("damage", ..., suffix="~1")`` resolves
    # ``self._d_old_name`` to ``<resolved damage>~1``.
    delta_u0_n: Scalar
    delta_u0_t: Scalar
    Tmax_n: Scalar
    Tmax_t: Scalar
    Kpen: Scalar
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
                f"SalehaniIraniTraction.forward: got {len(args)} args, expected {len(names)}"
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

        dn_sep = bound[_resolved("normal_separation")]
        ds1 = bound[_resolved("tangential_separation_1")]
        ds2 = bound[_resolved("tangential_separation_2")]
        d_old = bound[self._d_old_name]
        dn_pen_name = self._dn_pen_name
        dn_pen = bound[dn_pen_name] if dn_pen_name is not None and dn_pen_name in bound else None

        # Parameters: all five are static (allow_nonlinear defaults to False).
        delta_u0_n = self._get_param("delta_u0_n", nl_params, Scalar)
        delta_u0_t = self._get_param("delta_u0_t", nl_params, Scalar)
        Tmax_n = self._get_param("Tmax_n", nl_params, Scalar)
        Tmax_t = self._get_param("Tmax_t", nl_params, Scalar)
        Kpen = self._get_param("Kpen", nl_params, Scalar)

        # -------- Internal characteristic-length vector: tangential is sqrt(2) * raw.
        sqrt2 = math.sqrt(2.0)
        e_const = math.exp(1.0)
        sqrt2e = math.sqrt(2.0 * e_const)
        delta_u0_t_int = sqrt2 * delta_u0_t

        # Prefactors a_i (normal enters linearly, shear enters quadratically).
        a_n = e_const * Tmax_n
        a_t = sqrt2e * Tmax_t

        b_n = dn_sep / delta_u0_n
        b_s1 = ds1 / delta_u0_t_int
        b_s2 = ds2 / delta_u0_t_int

        x = b_n + b_s1 * b_s1 + b_s2 * b_s2
        exp_x = exp(-x)

        # -------- Trial damage and irreversibility cap.
        one = Scalar.from_value(1.0, like=dn_sep)
        zero = Scalar.from_value(0.0, like=dn_sep)
        d_trial = one - exp_x
        advance = gt(d_trial, d_old)
        d = where(advance, d_trial, d_old)
        factor = one - d

        # -------- Assemble traction.
        T_n = a_n * b_n * factor
        if dn_pen is not None:
            T_n = T_n + Kpen * dn_pen
        T_s1 = a_t * b_s1 * factor
        T_s2 = a_t * b_s2 * factor
        T = vec_from_scalars(T_n, T_s1, T_s2)

        if v is None:
            return T, d

        # -------- D-062 pushforward.
        #
        # Trial-value partials (only nonzero on the advancing branch):
        #   d(d_trial)/d(dn_sep) = exp_x * (1/delta_u0_n)
        #   d(d_trial)/d(ds1)    = exp_x * (2 * ds1 / delta_u0_t_int^2)
        #   d(d_trial)/d(ds2)    = exp_x * (2 * ds2 / delta_u0_t_int^2)
        # The irreversibility cap freezes d when d_trial does not advance:
        #   d(damage)/d(input) = where(advance, d(d_trial)/d(input), 0)
        #   d(damage)/d(d_old) = where(advance, 0, 1)
        inv_dun = one / delta_u0_n
        inv_dut2 = one / (delta_u0_t_int * delta_u0_t_int)
        dx_dn = inv_dun
        dx_ds1 = (one + one) * ds1 * inv_dut2
        dx_ds2 = (one + one) * ds2 * inv_dut2

        dd_ddn = where(advance, exp_x * dx_dn, zero)
        dd_dds1 = where(advance, exp_x * dx_ds1, zero)
        dd_dds2 = where(advance, exp_x * dx_ds2, zero)
        dd_dd_old = where(advance, zero, one)

        # T_i = a_i * b_i * factor where factor = (1 - damage). For input dj:
        #   dT_i/dj = a_i * (db_i/dj) * factor - a_i * b_i * d(damage)/dj
        # For input d_old (only contributes through factor):
        #   dT_i/dd_old = -a_i * b_i * d(damage)/dd_old
        # db_n/dn = 1 / delta_u0_n; db_t/dt = 1 / delta_u0_t_int (Kronecker).
        db_n_dn = inv_dun
        db_t_dt = one / delta_u0_t_int

        a_n_b_n = a_n * b_n
        a_t_b_s1 = a_t * b_s1
        a_t_b_s2 = a_t * b_s2

        def _dn_sep_action(V: Scalar) -> Vec:
            Vs = V
            T0 = a_n * (factor * db_n_dn - b_n * dd_ddn) * Vs
            T1 = -a_t_b_s1 * dd_ddn * Vs
            T2 = -a_t_b_s2 * dd_ddn * Vs
            return vec_from_scalars(T0, T1, T2)

        def _ds1_action(V: Scalar) -> Vec:
            Vs = V
            T0 = -a_n_b_n * dd_dds1 * Vs
            T1 = a_t * (factor * db_t_dt - b_s1 * dd_dds1) * Vs
            T2 = -a_t_b_s2 * dd_dds1 * Vs
            return vec_from_scalars(T0, T1, T2)

        def _ds2_action(V: Scalar) -> Vec:
            Vs = V
            T0 = -a_n_b_n * dd_dds2 * Vs
            T1 = -a_t_b_s1 * dd_dds2 * Vs
            T2 = a_t * (factor * db_t_dt - b_s2 * dd_dds2) * Vs
            return vec_from_scalars(T0, T1, T2)

        def _d_old_action(V: Scalar) -> Vec:
            Vs = V
            T0 = -a_n_b_n * dd_dd_old * Vs
            T1 = -a_t_b_s1 * dd_dd_old * Vs
            T2 = -a_t_b_s2 * dd_dd_old * Vs
            return vec_from_scalars(T0, T1, T2)

        def _dn_pen_action(V: Scalar) -> Vec:
            # The penalty term Kpen * dn_pen contributes only to T_n.
            Vs = V
            zero_Vs = zero * Vs
            return vec_from_scalars(Kpen * Vs, zero_Vs, zero_Vs)

        damage_actions: dict[str, ChainRuleAction] = {
            "normal_separation": lambda V, c=dd_ddn: c * V,
            "tangential_separation_1": lambda V, c=dd_dds1: c * V,
            "tangential_separation_2": lambda V, c=dd_dds2: c * V,
            self._d_old_name: lambda V, c=dd_dd_old: c * V,
        }

        traction_actions: dict[str, ChainRuleAction] = {
            "normal_separation": _dn_sep_action,
            "tangential_separation_1": _ds1_action,
            "tangential_separation_2": _ds2_action,
            self._d_old_name: _d_old_action,
        }
        if dn_pen_name is not None:
            traction_actions[dn_pen_name] = _dn_pen_action
            # dn_pen does not contribute to damage (structural zero).

        v_T = self.apply_chain_rule(v, "traction", traction_actions, output=T)
        v_d = self.apply_chain_rule(v, "damage", damage_actions, output=d)
        return T, d, {**v_T, **v_d}


__all__ = ["SalehaniIraniTraction"]
