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

"""Python-native mirror of the C++ ``AdvectiveStress`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import R2, Scalar, inner, pow
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("AdvectiveStress")
class AdvectiveStress(Model):
    r"""Variational advective stress associated with swelling-induced volume change.

    Implements $p_s = -c/(3 J_s^{5/3} J_t^{2/3}) * P_{ij} F_{ij}$ where
    $J_s$ (swelling/phase-change Jacobian) and $J_t$ (thermal-expansion
    Jacobian) are optional inputs defaulting to ``1``, $P$ is the first
    Piola-Kirchhoff stress, $F$ is the deformation gradient, and $c$ is
    the volume-change coefficient. Mirrors the C++
    ``models/porous_flow/AdvectiveStress`` set_value math 1:1.
    """

    hit = HitSchema(
        # Optional sub-Jacobian inputs (default=None pops them from input_spec
        # when HIT is silent, mirroring the C++ ``add_optional_input`` knob).
        input(
            "js",
            Scalar,
            "The Jacobian of the deformation gradient associated with the "
            "swelling and phase change",
            default=None,
            attr="_js_name",
        ),
        input(
            "jt",
            Scalar,
            "The Jacobian of the deformation gradient associated with the "
            "thermal and volume expansion",
            default=None,
            attr="_jt_name",
        ),
        input(
            "deformation_gradient",
            R2,
            "The deformation gradient",
            attr="_F_name",
        ),
        input(
            "pk1_stress",
            R2,
            "1st Piola-Kirchhoff stress",
            attr="_P_name",
        ),
        output("advective_stress", Scalar, "The average advective stress"),
        parameter(
            "coefficient",
            Scalar,
            "Coefficient c",
            attr="coeff",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the ``coefficient`` parameter (stored as
    # ``coeff``). The ``_*_name`` attrs hold the resolved HIT names (or
    # ``None`` when the optional input was not specified). Annotate so pyright
    # sees the typed wrapper that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    coeff: Scalar
    _js_name: str | None
    _jt_name: str | None
    _F_name: str
    _P_name: str

    def forward(  # type: ignore[override]
        self,
        *inputs,
        v: ChainRuleDict | None = None,
        **_,
    ):
        # Inputs arrive positionally in ``input_spec`` declaration order. The
        # optional ``js`` / ``jt`` entries are popped from ``input_spec`` when
        # HIT didn't name them, so the present subset is exactly the present
        # inputs — pair them with the surviving names.
        names = list(self.input_spec)
        if len(inputs) != len(names):
            raise AssertionError(
                f"AdvectiveStress.forward: got {len(inputs)} inputs, expected {len(names)}"
            )
        bound = dict(zip(names, inputs, strict=True))

        F = bound[self._F_name]
        P = bound[self._P_name]
        js_name = self._js_name
        jt_name = self._jt_name
        Js = (
            bound[js_name]
            if js_name is not None and js_name in bound
            else Scalar.from_value(1.0, like=F)
        )
        Jt = (
            bound[jt_name]
            if jt_name is not None and jt_name in bound
            else Scalar.from_value(1.0, like=F)
        )

        coeff = self._get_param("coeff", (), Scalar)

        # ``s = -c/3 * Js^(-5/3) * Jt^(-2/3)`` — the shared scalar prefactor.
        Js_p = pow(Js, -5.0 / 3.0)
        Jt_p = pow(Jt, -2.0 / 3.0)
        s = (-coeff / 3.0) * Js_p * Jt_p

        # ``q = P : F`` Frobenius inner product (full ``(3,3)`` contraction).
        q = inner(P, F)
        ps = s * q

        if v is None:
            return ps

        # Differential pushforward. The forward is bilinear in (P, F)
        # and a smooth power in (Js, Jt); each partial below is the analytical
        # coefficient times the corresponding tangent — no Jacobian formed.
        #
        # d ps / d Js = -s * (5 / (3 Js)) * q   →   +c/9 * Js^(-8/3) * Jt^(-2/3) * q
        # d ps / d Jt = -s * (2 / (3 Jt)) * q   →   +c/9 * Js^(-5/3) * Jt^(-5/3) * q  *(2/2 factor)
        # d ps / d P  = s * F                     (R2 → Scalar via inner)
        # d ps / d F  = s * P                     (R2 → Scalar via inner)
        actions: dict[str, ChainRuleAction] = {}

        if js_name is not None and js_name in bound:
            dps_dJs = (5.0 / 9.0) * coeff * pow(Js, -8.0 / 3.0) * Jt_p * q
            actions[js_name] = lambda V, c=dps_dJs: c * V
        if jt_name is not None and jt_name in bound:
            dps_dJt = (2.0 / 9.0) * coeff * Js_p * pow(Jt, -5.0 / 3.0) * q
            actions[jt_name] = lambda V, c=dps_dJt: c * V

        coeffP = s * P
        coeffF = s * F
        actions[self._P_name] = lambda V, A=coeffF: inner(A, V)
        actions[self._F_name] = lambda V, A=coeffP: inner(A, V)

        return ps, self.apply_chain_rule(
            v,
            "advective_stress",
            actions,
            output=ps,
        )


__all__ = ["AdvectiveStress"]
