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

"""Python-native mirror of the C++ ``GTNYieldFunction`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, cosh, sinh
from ...chain_rule import (
    ChainRuleAction,
    ChainRuleDict,
    SecondOrderChainRuleAction,
    SecondOrderChainRuleDict,
)
from ...model import Model


@register_neml2_object("GTNYieldFunction")
class GTNYieldFunction(Model):
    r"""Gurson-Tvergaard-Needleman yield function for poroplasticity. The yield
    function is defined as
    $f = \left( \frac{\bar{\sigma}}{\sigma_y + k} \right)^2 + 2 q_1 \phi \cosh \left( \frac{1}{2} q_2 \frac{3\sigma_h-\sigma_s}{\sigma_y + k} \right) - \left( q_3 \phi^2 + 1 \right)$,
    where $\bar{\sigma}$ is the von Mises stress, $\sigma_y$ is the yield stress, $k$ is isotropic
    hardening, $\phi$ is the porosity, $\sigma_h$ is the hydrostatic stress, and
    $\sigma_s$ is the void growth back stress (sintering stress). $q_1$, $q_2$, and $q_3$ are
    parameters controlling the yield
    mechanisms.
    """  # noqa: E501

    # Second-order chain-rule support — required so this leaf may sit inside a
    # ``Normality`` wrap (rate_independent_plasticity/gurson uses
    # ``Normality(yield_function)`` over ``mandel_stress + isotropic_hardening``).
    SUPPORTS_SECOND_ORDER = True

    # Variable names match the C++ defaults in
    # ``include/neml2/models/solid_mechanics/plasticity/GTNYieldFunction.h``;
    # HIT renames cascade via the schema's option-name == canonical-key
    # convention. ``isotropic_hardening`` is optional (default=None pops it
    # from input_spec when HIT is silent — mirrors the C++ ``add_optional_input``).
    hit = HitSchema(
        input("flow_invariant", Scalar, "Effective stress driving plastic flow"),
        input("poro_invariant", Scalar, "Effective stress driving porous flow"),
        input("void_fraction", Scalar, "Void fraction (porosity)"),
        input(
            "isotropic_hardening",
            Scalar,
            "Isotropic hardening",
            default=None,
            attr="_h_name",
        ),
        output("yield_function", Scalar, "Yield function"),
        parameter(
            "yield_stress",
            Scalar,
            "Yield stress",
            attr="sy",
            allow_promotion=True,
        ),
        parameter(
            "q1",
            Scalar,
            "Parameter controlling the balance/competition between plastic flow and "
            "void evolution.",
            attr="q1",
            allow_promotion=True,
        ),
        parameter(
            "q2",
            Scalar,
            "Void evolution rate",
            attr="q2",
            allow_promotion=True,
        ),
        parameter(
            "q3",
            Scalar,
            "Pore pressure",
            attr="q3",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the four parameters; ``_h_name`` is the
    # resolved name of the optional isotropic_hardening input (or ``None``).
    sy: Scalar
    q1: Scalar
    q2: Scalar
    q3: Scalar
    _h_name: str | None

    def forward(  # type: ignore[override]
        self,
        *args,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Inputs arrive positionally in ``input_spec`` declaration order, then
        # the *promoted_params pack. The optional ``isotropic_hardening`` entry is
        # popped from ``input_spec`` when HIT didn't name it; pair the present
        # subset with the surviving names.
        # Promotion bookkeeping: when a structural parameter (``q1``/``q2``/
        # ``q3``/``sy``) is promoted to a runtime input via the
        # HIT ``q1 = '<model>'`` form, the promoted input is appended to
        # ``input_spec`` (see ``Model.declare_typed_parameter``). Subtract
        # the promoted count so the structural inputs and the promoted_params tail
        # land in their respective slices.
        names = list(self.input_spec)
        n_in = len(names) - len(self._promoted_params)
        names = names[:n_in]
        inputs, promoted_params = args[:n_in], args[n_in:]
        if len(inputs) != n_in:
            raise AssertionError(
                f"GTNYieldFunction.forward: got {len(inputs)} inputs, expected {n_in}"
            )
        bound = dict(zip(names, inputs, strict=True))

        # Mirrors ``GTNYieldFunction::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/GTNYieldFunction.cxx``.
        se: Scalar = bound["flow_invariant"]
        sp: Scalar = bound["poro_invariant"]
        phi: Scalar = bound["void_fraction"]
        h_name = self._h_name
        h: Scalar | None = bound[h_name] if h_name is not None and h_name in bound else None

        sy = self._get_param("sy", promoted_params, Scalar)
        q1 = self._get_param("q1", promoted_params, Scalar)
        q2 = self._get_param("q2", promoted_params, Scalar)
        q3 = self._get_param("q3", promoted_params, Scalar)

        # Flow stress (depending on whether isotropic hardening is provided).
        sf = sy + h if h is not None else sy

        # arg = q2/2 * sp / sf appears in every cosh/sinh below.
        arg = (q2 / 2.0) * sp / sf
        cosh_a = cosh(arg)
        sinh_a = sinh(arg)

        f = (se / sf) ** 2 + 2.0 * q1 * phi * cosh_a - (1.0 + q3 * phi**2)
        if v is None:
            return f

        # Differential pushforward. Each action takes a Scalar tangent
        # V and returns the Scalar tangent of f. Coefficients match the dense
        # C++ Jacobian in set_value:
        #   df/dse  = 2 * se / sf**2
        #   df/dsp  = q1 * phi * q2 / sf * sinh(arg)
        #   df/dphi = 2 * q1 * cosh(arg) - 2 * q3 * phi
        #   df/dh   = -2 * se**2 / sf**3 - q1*phi*q2*sp/sf**2 * sinh(arg)
        #   df/dsy  = same as df/dh
        #   df/dq1  = 2 * phi * cosh(arg)
        #   df/dq2  = q1 * phi * sp / sf * sinh(arg)
        #   df/dq3  = -phi**2
        coef_se = 2.0 * se / sf**2
        coef_sp = q1 * phi * q2 / sf * sinh_a
        coef_phi = 2.0 * q1 * cosh_a - 2.0 * q3 * phi
        coef_h = -2.0 * se**2 / sf**3 - q1 * phi * q2 * sp / sf**2 * sinh_a

        actions_1: dict[str, ChainRuleAction] = {
            "flow_invariant": lambda V, c=coef_se: c * V,
            "poro_invariant": lambda V, c=coef_sp: c * V,
            "void_fraction": lambda V, c=coef_phi: c * V,
        }
        if h is not None and h_name is not None:
            actions_1[h_name] = lambda V, c=coef_h: c * V

        # Promoted-parameter contributions: each parameter that was promoted to a
        # runtime input gets its own action keyed on the resolved input name.
        sy_name: str | None = None
        q1_name: str | None = None
        q2_name: str | None = None
        q3_name: str | None = None
        if "sy" in self._promoted_params:
            sy_name = self._promoted_params["sy"].input_name
            # ``sy`` shares the same coefficient as the hardening derivative.
            actions_1[sy_name] = lambda V, c=coef_h: c * V
        if "q1" in self._promoted_params:
            q1_name = self._promoted_params["q1"].input_name
            coef_q1 = 2.0 * phi * cosh_a
            actions_1[q1_name] = lambda V, c=coef_q1: c * V
        if "q2" in self._promoted_params:
            q2_name = self._promoted_params["q2"].input_name
            coef_q2 = q1 * phi * sp / sf * sinh_a
            actions_1[q2_name] = lambda V, c=coef_q2: c * V
        if "q3" in self._promoted_params:
            q3_name = self._promoted_params["q3"].input_name
            coef_q3 = -(phi**2)
            actions_1[q3_name] = lambda V, c=coef_q3: c * V

        if v2 is None and vh is None:
            return f, *self.propagate_tangents(v, "yield_function", actions_1, output=f)

        # Second-order pushforward. Each
        # ``action_2(Va, Vb)`` receives primal-shape Scalar tangents and returns
        # a primal-shape Scalar bilinear; the framework handles the (N_a, N_b)
        # seed-pair iteration. Pure typed-wrapper algebra — no .data, no torch.
        #
        # Hessian entries (mirrors C++ ``set_value`` ``d2out_din2`` branch in
        # ``GTNYieldFunction.cxx``). Variables: se, sp, phi (always); h, sy,
        # q1, q2, q3 (when promoted / hardening is present). Cross terms
        # registered symmetrically in BOTH orders — the framework iterates the
        # outer-product (a, b) over actions_2 and treats absent pairs as zero
        # (rather than recovering symmetry); upstream chain expressions stay
        # simple this way (see ``chain_rule.py`` docstring on v2 storage).
        actions_2: dict[tuple[str, str], SecondOrderChainRuleAction] = {}

        def _add_symmetric(a: str, b: str, coef) -> None:
            actions_2[(a, b)] = lambda Va, Vb, c=coef: c * Va * Vb
            if a != b:
                actions_2[(b, a)] = lambda Va, Vb, c=coef: c * Va * Vb

        # Precompute shared sub-expressions matching the C++ source's repeated
        # algebraic groupings.
        # d2f/dse2 = 2 / sf**2
        d2_se_se = 2.0 / sf**2
        # d2f/dse dh = d2f/dse dsy = -4 * se / sf**3
        d2_se_h = -4.0 * se / sf**3
        # d2f/dsp2 = phi * q1 * q2**2 / (2 * sf**2) * cosh(arg)
        d2_sp_sp = phi * q1 * q2**2 / (2.0 * sf**2) * cosh_a
        # d2f/dsp dphi = q1 * q2 * sinh(arg) / sf
        d2_sp_phi = q1 * q2 / sf * sinh_a
        # d2f/dsp dh = d2f/dsp dsy =
        #    -phi*q1*q2 * (q2*sp*cosh + 2*sf*sinh) / (2 * sf**3)
        d2_sp_h = -phi * q1 * q2 * (q2 * sp * cosh_a + 2.0 * sf * sinh_a) / (2.0 * sf**3)
        # d2f/dsp dq1 = phi * q2 * sinh / sf
        d2_sp_q1 = phi * q2 / sf * sinh_a
        # d2f/dsp dq2 = phi * q1 * (q2*sp*cosh + 2*sf*sinh) / (2*sf**2)
        d2_sp_q2 = phi * q1 * (q2 * sp * cosh_a + 2.0 * sf * sinh_a) / (2.0 * sf**2)
        # d2f/dphi2 = -2 * q3
        d2_phi_phi = -2.0 * q3
        # d2f/dphi dh = d2f/dphi dsy = -q1 * q2 * sp * sinh / sf**2
        d2_phi_h = -q1 * q2 * sp / sf**2 * sinh_a
        # d2f/dphi dq1 = 2 * cosh
        d2_phi_q1 = 2.0 * cosh_a
        # d2f/dphi dq2 = q1 * sp * sinh / sf
        d2_phi_q2 = q1 * sp / sf * sinh_a
        # d2f/dphi dq3 = -2 * phi
        d2_phi_q3 = -2.0 * phi
        # d2f/dh2 = d2f/dsy2 = d2f/dh dsy =
        #     (12*se**2 + phi*q1*q2*sp*(q2*sp*cosh + 4*sf*sinh)) / (2*sf**4)
        d2_h_h = (12.0 * se**2 + phi * q1 * q2 * sp * (q2 * sp * cosh_a + 4.0 * sf * sinh_a)) / (
            2.0 * sf**4
        )
        # d2f/dh dq1 = d2f/dsy dq1 = -phi * q2 * sp * sinh / sf**2 (== d2_phi_h)
        d2_h_q1 = -phi * q2 * sp / sf**2 * sinh_a
        # d2f/dh dq2 = d2f/dsy dq2 =
        #     -phi * q1 * sp * (q2*sp*cosh + 2*sf*sinh) / (2*sf**3)
        d2_h_q2 = -phi * q1 * sp * (q2 * sp * cosh_a + 2.0 * sf * sinh_a) / (2.0 * sf**3)
        # d2f/dq1 dq2 = phi * sp * sinh / sf
        d2_q1_q2 = phi * sp / sf * sinh_a
        # d2f/dq2 dq2 = phi * q1 * sp**2 * cosh / (2 * sf**2)
        d2_q2_q2 = phi * q1 * sp**2 * cosh_a / (2.0 * sf**2)

        # se row
        _add_symmetric("flow_invariant", "flow_invariant", d2_se_se)
        if h is not None and h_name is not None:
            _add_symmetric("flow_invariant", h_name, d2_se_h)
        if sy_name is not None:
            _add_symmetric("flow_invariant", sy_name, d2_se_h)

        # sp row (sp,sp; sp,phi)
        _add_symmetric("poro_invariant", "poro_invariant", d2_sp_sp)
        _add_symmetric("poro_invariant", "void_fraction", d2_sp_phi)
        if h is not None and h_name is not None:
            _add_symmetric("poro_invariant", h_name, d2_sp_h)
        if sy_name is not None:
            _add_symmetric("poro_invariant", sy_name, d2_sp_h)
        if q1_name is not None:
            _add_symmetric("poro_invariant", q1_name, d2_sp_q1)
        if q2_name is not None:
            _add_symmetric("poro_invariant", q2_name, d2_sp_q2)

        # phi row (phi,phi)
        _add_symmetric("void_fraction", "void_fraction", d2_phi_phi)
        if h is not None and h_name is not None:
            _add_symmetric("void_fraction", h_name, d2_phi_h)
        if sy_name is not None:
            _add_symmetric("void_fraction", sy_name, d2_phi_h)
        if q1_name is not None:
            _add_symmetric("void_fraction", q1_name, d2_phi_q1)
        if q2_name is not None:
            _add_symmetric("void_fraction", q2_name, d2_phi_q2)
        if q3_name is not None:
            _add_symmetric("void_fraction", q3_name, d2_phi_q3)

        # h-h, h-sy, sy-sy share the same coefficient
        if h is not None and h_name is not None:
            _add_symmetric(h_name, h_name, d2_h_h)
            if sy_name is not None:
                _add_symmetric(h_name, sy_name, d2_h_h)
            if q1_name is not None:
                _add_symmetric(h_name, q1_name, d2_h_q1)
            if q2_name is not None:
                _add_symmetric(h_name, q2_name, d2_h_q2)
        if sy_name is not None:
            _add_symmetric(sy_name, sy_name, d2_h_h)
            if q1_name is not None:
                _add_symmetric(sy_name, q1_name, d2_h_q1)
            if q2_name is not None:
                _add_symmetric(sy_name, q2_name, d2_h_q2)

        # q1-q2 cross
        if q1_name is not None and q2_name is not None:
            _add_symmetric(q1_name, q2_name, d2_q1_q2)

        # q2-q2
        if q2_name is not None:
            _add_symmetric(q2_name, q2_name, d2_q2_q2)

        return f, *self.propagate_tangents(
            v,
            "yield_function",
            actions_1,
            output=f,
            v2=v2,
            actions_2=actions_2,
            vh=vh,
        )


__all__ = ["GTNYieldFunction"]
