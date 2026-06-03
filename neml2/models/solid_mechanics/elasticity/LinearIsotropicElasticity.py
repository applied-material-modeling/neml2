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

"""Python-native mirror of the C++ ``LinearIsotropicElasticity`` model.

Stiffness form: ``σ = 3K · vol(ε) + 2G · dev(ε)`` with K, G derived from
(E, ν). Compliance form (``compliance = true``): ``ε = (1/3K) · vol(σ) +
(1/2G) · dev(σ)`` — the inverse, mapping stress to strain. Rate form
(``rate_form = true``): identical math, but the input/output names are
suffixed with ``_rate`` (so e.g. ``strain_rate → stress_rate``).

Variable names: ``"strain"`` → ``"stress"`` by default; swapped when
``compliance`` is set; both suffixed when ``rate_form`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, option, output
from ....types import SR2, Scalar, dev, vol

if TYPE_CHECKING:
    import nmhit

    from ....factory import _NativeInputFile


@register_native("LinearIsotropicElasticity")
class LinearIsotropicElasticity(Model):
    r"""Relate elastic strain to stress (or stress to strain, in compliance form)
    for a linear isotropic material.

    Stiffness form (default): :math:`\sigma = 3K\,\mathrm{vol}(\varepsilon) +
    2G\,\mathrm{dev}(\varepsilon)`.
    Compliance form: :math:`\varepsilon = (1/3K)\,\mathrm{vol}(\sigma) +
    (1/2G)\,\mathrm{dev}(\sigma)`.
    Rate form adds the ``_rate`` suffix to the input/output variable names; the
    math is identical (the relation between rates is linear-in-rate just like
    the relation between totals).
    """

    hit = HitSchema(
        input("strain", SR2, "Elastic strain"),
        output("stress", SR2, "Stress"),
        option(
            "compliance",
            bool,
            "Whether the model defines the compliance relationship (stress -> strain). "
            "Default false: maps strain -> stress.",
            default=False,
            attr="_compliance",
        ),
        option(
            "rate_form",
            bool,
            "Whether the model defines the stress-strain relationship in rate form. "
            "When true the input/output variable names are suffixed with '_rate'.",
            default=False,
            attr="_rate_form",
        ),
        option(
            "coefficients",
            list,
            "Coefficients used to define the elasticity tensor",
            reader=lambda node, name: node.param_list_str(name),
        ),
        option(
            "coefficient_types",
            list,
            "Types for each parameter, options are: INVALID, P_WAVE_MODULUS, "
            "POISSONS_RATIO, YOUNGS_MODULUS, SHEAR_MODULUS, BULK_MODULUS, "
            "LAME_LAMBDA",
            reader=lambda node, name: node.param_list_str(name),
        ),
    )

    E: Scalar
    nu: Scalar
    _compliance: bool
    _rate_form: bool

    def __init__(
        self, E=None, nu=None, *, factory: _NativeInputFile | None = None, **hit_values
    ) -> None:
        # Forward schema-driven hit_values (input/output rename, etc.) to the
        # base; E and nu are extracted from HIT's coefficients list by from_hit
        # and declared explicitly here.
        super().__init__(**hit_values)
        # After super().__init__, input_spec and output_spec have the user-
        # resolved names for ``strain`` and ``stress``. Apply the ``_rate``
        # suffix (rate_form) and ``compliance`` swap to produce the final
        # variable names. Mirrors C++ Elasticity.cxx:59-62.
        strain_name = next(iter(self.input_spec))
        stress_name = next(iter(self.output_spec))
        if self._rate_form:
            strain_name = strain_name + "_rate"
            stress_name = stress_name + "_rate"
        if self._compliance:
            self.input_spec = {stress_name: SR2}
            self.output_spec = {strain_name: SR2}
        else:
            self.input_spec = {strain_name: SR2}
            self.output_spec = {stress_name: SR2}
        if E is not None:
            self.declare_typed_parameter("E", E, Scalar, factory=factory, allow_nonlinear=True)
        if nu is not None:
            self.declare_typed_parameter("nu", nu, Scalar, factory=factory, allow_nonlinear=True)

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> LinearIsotropicElasticity:
        coeffs = node.param_list_str("coefficients")
        types = node.param_list_str("coefficient_types")
        e_idx = types.index("YOUNGS_MODULUS")
        nu_idx = types.index("POISSONS_RATIO")
        schema_kwargs = cls.hit.kwargs_from_hit(node, factory)
        # coefficients/coefficient_types are consumed manually above.
        schema_kwargs.pop("coefficients", None)
        schema_kwargs.pop("coefficient_types", None)
        return cls(E=coeffs[e_idx], nu=coeffs[nu_idx], factory=factory, **schema_kwargs)

    def _moduli(self, E: Scalar, nu: Scalar) -> tuple[Scalar, Scalar]:
        K = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        return K, G

    def forward(  # type: ignore[override]
        self,
        x: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # ``x`` is whatever input the schema landed at the first positional
        # slot: ``strain`` in stiffness form, ``stress`` in compliance form
        # (possibly with the ``_rate`` suffix when rate_form is set). The math
        # is the same shape — vol and dev coefficients flip between K- and
        # 1/K-based when compliance is on. Mirrors C++ LinearIsotropicElasticity.cxx:52-82.
        E = self._get_param("E", nl_params, Scalar)
        nu = self._get_param("nu", nl_params, Scalar)
        K, G = self._moduli(E, nu)
        vf = 1.0 / (3.0 * K) if self._compliance else 3.0 * K
        df = 1.0 / (2.0 * G) if self._compliance else 2.0 * G
        y = vf * vol(x) + df * dev(x)
        if v is None:
            return y

        in_name = next(iter(self.input_spec))
        out_name = next(iter(self.output_spec))

        def x_action(V: SR2) -> SR2:
            return vf * vol(V) + df * dev(V)

        actions: dict = {in_name: x_action}

        # Closed-form partials w.r.t. E / ν. The sign factor ``s`` unifies the
        # two cases: in stiffness mode vf = 3K (so dvf/dK = 3 = vf/K · 1), and
        # in compliance mode vf = 1/(3K) (so dvf/dK = -vf/K). Both modes give
        #   dvf/dE     = s · vf / E
        #   dvf/dnu    = s · 2·vf / (1 - 2·nu)
        #   ddf/dE     = s · df / E
        #   ddf/dnu    = -s · df / (1 + nu)
        # with s = +1 (stiffness) or -1 (compliance).
        if "E" in self._nl_params or "nu" in self._nl_params:
            s = -1.0 if self._compliance else 1.0
            if "E" in self._nl_params:
                dy_dE = s * (vf / E) * vol(x) + s * (df / E) * dev(x)
                actions[self._nl_params["E"].input_name] = lambda V, c=dy_dE: c * V
            if "nu" in self._nl_params:
                dy_dnu = s * (2.0 * vf / (1.0 - 2.0 * nu)) * vol(x) + (-s) * (
                    df / (1.0 + nu)
                ) * dev(x)
                actions[self._nl_params["nu"].input_name] = lambda V, c=dy_dnu: c * V

        return y, self.apply_chain_rule(v, out_name, actions, output=y)
