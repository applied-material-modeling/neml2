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

"""Python-native mirror of the C++ ``OrthotropicLinearTraction`` model."""

from __future__ import annotations

from typing import cast

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, Vec, vec_from_scalars


@register_native("OrthotropicLinearTraction")
class OrthotropicLinearTraction(Model):
    r"""Orthotropic linear-elastic interface traction:
    $T_n = K_n \delta_n^\text{sep}$, $T_{si} = K_t \delta_{si}$.
    If `normal_penetration` is supplied,
    $K_\text{pen} \delta_n^\text{pen}$ is added to $T_n$ as a
    penalty term resisting interpenetration (`penalty_stiffness` becomes
    required); otherwise interpenetration produces zero normal traction.
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
            "Optional normal penetration (typically the Macaulay-negative part of the "
            "normal jump). When supplied, the penalty stiffness times this is added to "
            "the normal traction as a penalty term resisting interpenetration. Requires "
            "`penalty_stiffness` to be supplied as well.",
            default=None,
            attr="_dn_pen_name",
        ),
        input("tangential_separation_1", Scalar, "First tangential separation"),
        input("tangential_separation_2", Scalar, "Second tangential separation"),
        output("traction", Vec, "Traction Vec"),
        parameter("normal_stiffness", Scalar, "Normal stiffness", attr="Kn"),
        parameter(
            "tangential_stiffness",
            Scalar,
            "Tangential stiffness (isotropic)",
            attr="Kt",
        ),
        parameter(
            "penalty_stiffness",
            Scalar,
            "Penalty stiffness used to resist interpenetration. Required when "
            "`normal_penetration` is supplied; ignored otherwise.",
            attr="Kpen",
            default=0.0,
        ),
    )

    # ``from_hit`` auto-declares the three Scalar parameters via
    # ``declare_typed_parameter``. The optional ``normal_penetration`` input
    # resolves to a HIT name (string) when the user supplies it, ``None``
    # otherwise.
    Kn: Scalar
    Kt: Scalar
    Kpen: Scalar
    _dn_pen_name: str | None

    def forward(  # type: ignore[override]
        self,
        *args,
        v: ChainRuleDict | None = None,
        **_,
    ):
        # ``input_spec`` order: structural inputs first (declaration order from
        # the schema, minus optional inputs that HIT omitted), then promoted
        # nl-parameters appended. ``Kn``/``Kt``/``Kpen`` are static (no
        # ``allow_nonlinear``) so no nl-promotions occur in practice.
        names = list(self.input_spec)
        if len(args) != len(names):
            raise AssertionError(
                f"OrthotropicLinearTraction.forward: got {len(args)} args, expected {len(names)}"
            )
        n_nl = len(self._nl_params)
        n_struct = len(names) - n_nl
        inputs, nl_params = args[:n_struct], args[n_struct:]
        struct_names = names[:n_struct]
        bound = dict(zip(struct_names, inputs, strict=True))

        # ``input_spec`` holds the resolved (post-rename) names; map canonical
        # roles through the rename table to pluck each typed input.
        renames = getattr(self, "_var_renames", {})

        def _resolved(canon: str) -> str:
            return renames.get(canon, canon)

        dn_sep = cast(Scalar, bound[_resolved("normal_separation")])
        ds1 = cast(Scalar, bound[_resolved("tangential_separation_1")])
        ds2 = cast(Scalar, bound[_resolved("tangential_separation_2")])
        dn_pen_name = self._dn_pen_name
        dn_pen = (
            cast(Scalar, bound[dn_pen_name])
            if dn_pen_name is not None and dn_pen_name in bound
            else None
        )

        Kn = self._get_param("Kn", nl_params, Scalar)
        Kt = self._get_param("Kt", nl_params, Scalar)
        # ``Kpen`` is always declared (default 0); only used when penetration is wired.
        Kpen = self._get_param("Kpen", nl_params, Scalar)

        # -------- Assemble traction.
        T_n = Kn * dn_sep
        if dn_pen is not None:
            T_n = T_n + Kpen * dn_pen
        T_s1 = Kt * ds1
        T_s2 = Kt * ds2
        T = vec_from_scalars(T_n, T_s1, T_s2)

        if v is None:
            return T

        # -------- D-062 pushforward.
        # Linear leaf: each action is the forward applied to the tangent on
        # the relevant component. ``zero * V`` keeps sub-batch metadata aligned.
        zero = Scalar.from_value(0.0, like=dn_sep)

        def _dn_sep_action(V: Scalar) -> Vec:
            Vs = cast(Scalar, V)
            return vec_from_scalars(Kn * Vs, zero * Vs, zero * Vs)

        def _ds1_action(V: Scalar) -> Vec:
            Vs = cast(Scalar, V)
            return vec_from_scalars(zero * Vs, Kt * Vs, zero * Vs)

        def _ds2_action(V: Scalar) -> Vec:
            Vs = cast(Scalar, V)
            return vec_from_scalars(zero * Vs, zero * Vs, Kt * Vs)

        def _dn_pen_action(V: Scalar) -> Vec:
            Vs = cast(Scalar, V)
            return vec_from_scalars(Kpen * Vs, zero * Vs, zero * Vs)

        traction_actions: dict[str, ChainRuleAction] = {
            "normal_separation": _dn_sep_action,
            "tangential_separation_1": _ds1_action,
            "tangential_separation_2": _ds2_action,
        }
        if dn_pen_name is not None:
            traction_actions[dn_pen_name] = _dn_pen_action

        return T, self.apply_chain_rule(v, "traction", traction_actions, output=T)


__all__ = ["OrthotropicLinearTraction"]
