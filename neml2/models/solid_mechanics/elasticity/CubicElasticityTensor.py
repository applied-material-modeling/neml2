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

"""Python-native mirror of the C++ ``CubicElasticityTensor`` model.

$C = C1 * I_C1 + C2 * I_C2 + C3 * I_C3$ with (C1, C2, C3) the three
independent cubic elastic constants and ``I_C1/I_C2/I_C3`` the corresponding
Mandel-basis projectors (``SSR4.identity_C1/C2/C3``).

The only conversion table supported by the C++ ``CubicElasticityConverter`` is
(SHEAR_MODULUS, YOUNGS_MODULUS, POISSONS_RATIO) -> (C1, C2, C3) with

* $C1 = E (1 - nu) / ((1 + nu) (1 - 2 nu))$
* $C2 = E nu / ((1 + nu) (1 - 2 nu))$
* ``C3 = 2 G``

The output is a single ``SSR4`` whose external name defaults to the HIT block
name (matching the C++ ``declare_output_variable<SSR4>(name())``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import BLOCK_NAME, HitSchema, option, output
from ....types import SSR4, Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    import nmhit

    from ....factory import _NativeInputFile


@register_neml2_object("CubicElasticityTensor")
class CubicElasticityTensor(Model):
    """This class defines a cubic anisotropic elasticity tensor using three
    parameters.  Various options are available for which three parameters to
    provide.
    """

    hit = HitSchema(
        # Output name defaults to the HIT block name (the C++
        # ``declare_output_variable<SSR4>(name())`` convention).
        output(
            "C",
            SSR4,
            "Cubic elasticity tensor",
            default=BLOCK_NAME,
            attr="_C_name",
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

    # Output variable name (resolved via ``BLOCK_NAME`` default).
    _C_name: str
    # Static-mode storage for the three cubic-symmetry coefficients.
    G: Scalar
    E: Scalar
    nu: Scalar

    def __init__(
        self,
        *,
        G=None,
        E=None,
        nu=None,
        factory: _NativeInputFile | None = None,
        **hit_values,
    ) -> None:
        # ``coefficients`` and ``coefficient_types`` are consumed by
        # ``from_hit`` below; the resolved (G, E, nu) triple is forwarded here.
        # Direct Python construction without HIT is not supported for this
        # model (the conversion-table dispatch needs the explicit triple).
        hit_values.pop("coefficients", None)
        hit_values.pop("coefficient_types", None)
        super().__init__(**hit_values)
        if G is None:
            return
        # Declare all three coefficients as nonlinear-capable parameters so
        # they can independently be promoted to runtime inputs (mode 3/4);
        # literal HIT values stay as static buffers (mode 1).
        self.declare_typed_parameter("G", G, Scalar, factory=factory, allow_nonlinear=True)
        self.declare_typed_parameter("E", E, Scalar, factory=factory, allow_nonlinear=True)
        self.declare_typed_parameter("nu", nu, Scalar, factory=factory, allow_nonlinear=True)

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> CubicElasticityTensor:
        coeffs = node.param_list_str("coefficients")
        types = node.param_list_str("coefficient_types")
        if len(coeffs) != 3 or len(types) != 3:
            raise ValueError(
                f"CubicElasticityTensor expects exactly 3 coefficients and 3 types; "
                f"got {len(coeffs)} coefficients and {len(types)} types."
            )
        if frozenset(types) != frozenset(("SHEAR_MODULUS", "YOUNGS_MODULUS", "POISSONS_RATIO")):
            raise NotImplementedError(
                "CubicElasticityTensor native re-impl only supports the "
                "(SHEAR_MODULUS, YOUNGS_MODULUS, POISSONS_RATIO) parameterization; "
                f"got {types}."
            )
        g_idx = types.index("SHEAR_MODULUS")
        e_idx = types.index("YOUNGS_MODULUS")
        nu_idx = types.index("POISSONS_RATIO")
        schema_kwargs = cls.hit.kwargs_from_hit(node, factory)
        schema_kwargs.pop("coefficients", None)
        schema_kwargs.pop("coefficient_types", None)
        return cls(
            G=coeffs[g_idx],
            E=coeffs[e_idx],
            nu=coeffs[nu_idx],
            factory=factory,
            **schema_kwargs,
        )

    def forward(  # type: ignore[override]
        self,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        G = self._get_param("G", nl_params, Scalar)
        E = self._get_param("E", nl_params, Scalar)
        nu = self._get_param("nu", nl_params, Scalar)

        # CubicElasticityConverter G_E_nu_to_C{1,2,3}:
        #   C1 = E (1 - nu) / ((1 + nu) (1 - 2 nu))
        #   C2 = E nu      / ((1 + nu) (1 - 2 nu))
        #   C3 = 2 G
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        C1 = E * (1.0 - nu) / denom
        C2 = E * nu / denom
        C3 = 2.0 * G

        # Pick dtype/device from any of the typed scalars (all three share
        # them; G is the only one always guaranteed nonzero in the buffer).
        I1 = SSR4.identity_C1(dtype=G.data.dtype, device=G.data.device)
        I2 = SSR4.identity_C2(dtype=G.data.dtype, device=G.data.device)
        I3 = SSR4.identity_C3(dtype=G.data.dtype, device=G.data.device)

        # Build the elasticity tensor with typed wrapper algebra: no .data
        # access on inputs, no SSR4 ever materialized as a raw torch tensor.
        C = C1 * I1 + C2 * I2 + C3 * I3

        if v is None:
            return C

        # Differential pushforward. The forward is linear in each
        # coefficient through the (C1, C2, C3) closed forms, so the action on
        # the input scalar tangent V is the same linear combination with the
        # closed-form partial as the coefficient. We never form the full
        # n_out x n_in Jacobian; the action is one SSR4 per scalar input.
        #
        # CubicElasticityConverter partials:
        #   dC1/dG = 0,   dC1/dE = C1/E,   dC1/dnu = -2 (nu - 2) nu E / D^2
        #   dC2/dG = 0,   dC2/dE = C2/E,   dC2/dnu = (2 nu^2 E + E) / D^2
        #   dC3/dG = 2,   dC3/dE = 0,      dC3/dnu = 0
        # where D = 2 nu^2 + nu - 1 == (2 nu - 1) (nu + 1) == -((1+nu)(1-2nu)).
        D = 2.0 * nu * nu + nu - 1.0
        D2 = D * D
        dC1_dnu = -2.0 * (nu - 2.0) * nu * E / D2
        dC2_dnu = (2.0 * nu * nu * E + E) / D2

        actions: dict = {}
        if "G" in self._nl_params:
            # dC = 2 I3 * V
            def action_G(V: Scalar, _I3=I3) -> SSR4:
                return 2.0 * V * _I3

            actions[self._nl_params["G"].input_name] = action_G
        if "E" in self._nl_params:
            dsig_dE = (C1 / E) * I1 + (C2 / E) * I2

            def action_E(V: Scalar, _c=dsig_dE) -> SSR4:
                return _c * V

            actions[self._nl_params["E"].input_name] = action_E
        if "nu" in self._nl_params:
            dsig_dnu = dC1_dnu * I1 + dC2_dnu * I2

            def action_nu(V: Scalar, _c=dsig_dnu) -> SSR4:
                return _c * V

            actions[self._nl_params["nu"].input_name] = action_nu

        return C, self.apply_chain_rule(v, "C", actions, output=C)
