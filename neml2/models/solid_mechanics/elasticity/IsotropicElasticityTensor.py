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

"""Python-native mirror of the C++ ``IsotropicElasticityTensor`` model.

$C = 3K * I_vol + 2G * I_dev$ with (K, G) derived from a user-selected pair of
elastic constants. Two parameterizations are supported, matching the C++
``IsotropicElasticityConverter`` conversion table:

* (BULK_MODULUS, SHEAR_MODULUS) -> (K, G) directly
* (YOUNGS_MODULUS, POISSONS_RATIO) -> K = E / (3(1 - 2 nu)), G = E / (2(1 + nu))

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


@register_neml2_object("IsotropicElasticityTensor")
class IsotropicElasticityTensor(Model):
    """This class defines an isotropic elasticity tensor using two parameters.
    Various options are available for which two parameters to provide.
    """

    hit = HitSchema(
        # Output name defaults to the HIT block name (the C++
        # ``declare_output_variable<SSR4>(name())`` convention).
        output(
            "C",
            SSR4,
            "Isotropic elasticity tensor",
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
    # Static-mode storage for the two coefficients. The actual attribute names
    # used depend on the parameterization selected via ``coefficient_types``:
    # either (K, G) for K_G or (E, nu) for E_nu.
    K: Scalar
    G: Scalar
    E: Scalar
    nu: Scalar
    # Records which two slot names ``__init__`` populated, e.g. ("E", "nu") or
    # ("K", "G"); used by ``forward`` to dispatch on the parameterization.
    _param_pair: tuple[str, str]

    def __init__(
        self,
        *,
        param_pair: tuple[str, str] | None = None,
        coef0=None,
        coef1=None,
        factory: _NativeInputFile | None = None,
        **hit_values,
    ) -> None:
        # ``coefficients`` and ``coefficient_types`` are consumed by ``from_hit``
        # below; the resolved (param_pair, coef0, coef1) triple is forwarded
        # here. Direct Python construction without HIT is not supported for
        # this model (the conversion-table dispatch needs the explicit pair).
        hit_values.pop("coefficients", None)
        hit_values.pop("coefficient_types", None)
        super().__init__(**hit_values)
        if param_pair is None:
            return
        a, b = param_pair
        # Declare both coefficients as promotion-capable parameters so they can
        # independently be promoted to runtime inputs (mode 3/4); literal HIT
        # values stay as static buffers (mode 1).
        self.declare_typed_parameter(a, coef0, Scalar, factory=factory, allow_promotion=True)
        self.declare_typed_parameter(b, coef1, Scalar, factory=factory, allow_promotion=True)
        # Remember which two slots are populated so ``forward`` can dispatch.
        self._param_pair = param_pair

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> IsotropicElasticityTensor:
        coeffs = node.param_list_str("coefficients")
        types = node.param_list_str("coefficient_types")
        if len(coeffs) != 2 or len(types) != 2:
            raise ValueError(
                f"IsotropicElasticityTensor expects exactly 2 coefficients and 2 types; "
                f"got {len(coeffs)} coefficients and {len(types)} types."
            )
        type_set = frozenset(types)
        if type_set == frozenset(("YOUNGS_MODULUS", "POISSONS_RATIO")):
            e_idx = types.index("YOUNGS_MODULUS")
            nu_idx = types.index("POISSONS_RATIO")
            pair = ("E", "nu")
            c0, c1 = coeffs[e_idx], coeffs[nu_idx]
        elif type_set == frozenset(("BULK_MODULUS", "SHEAR_MODULUS")):
            k_idx = types.index("BULK_MODULUS")
            g_idx = types.index("SHEAR_MODULUS")
            pair = ("K", "G")
            c0, c1 = coeffs[k_idx], coeffs[g_idx]
        else:
            raise NotImplementedError(
                "IsotropicElasticityTensor native re-impl only supports the "
                "(YOUNGS_MODULUS, POISSONS_RATIO) and (BULK_MODULUS, SHEAR_MODULUS) "
                f"parameterizations; got {types}."
            )
        schema_kwargs = cls.hit.kwargs_from_hit(node, factory)
        schema_kwargs.pop("coefficients", None)
        schema_kwargs.pop("coefficient_types", None)
        return cls(param_pair=pair, coef0=c0, coef1=c1, factory=factory, **schema_kwargs)

    def forward(  # type: ignore[override]
        self,
        *promoted_params,
        v: ChainRuleDict | None = None,
    ):
        a, b = self._param_pair
        c0 = self._get_param(a, promoted_params, Scalar)
        c1 = self._get_param(b, promoted_params, Scalar)

        # Conversion table -> (K, G) and their partials w.r.t. (c0, c1).
        if (a, b) == ("K", "G"):
            K, G = c0, c1

            # K_G to K: dK/dK = 1, dK/dG = 0; K_G to G: dG/dK = 0, dG/dG = 1.
            # The pushforward action is trivial (linear-leaf shape).
            def dK_action(V: Scalar, _src: str) -> Scalar:
                # K depends only on c0 (K itself).
                return V if _src == "K" else 0.0 * V

            def dG_action(V: Scalar, _src: str) -> Scalar:
                return V if _src == "G" else 0.0 * V

        else:  # ("E", "nu")
            E, nu = c0, c1
            # K = E / (3 (1 - 2 nu)),  G = E / (2 (1 + nu))
            K = E / (3.0 * (1.0 - 2.0 * nu))
            G = E / (2.0 * (1.0 + nu))

            # Closed-form partials (same as the IsotropicElasticityConverter
            # E_nu_to_K / E_nu_to_G table entries):
            #   dK/dE = K/E,           dK/dnu = 6 K^2 / E  ==  2K / (1 - 2 nu)
            #   dG/dE = G/E,           dG/dnu = -G / (1 + nu)
            def dK_action(V: Scalar, _src: str) -> Scalar:
                if _src == "E":
                    return (K / E) * V
                return (6.0 * K * K / E) * V

            def dG_action(V: Scalar, _src: str) -> Scalar:
                if _src == "E":
                    return (G / E) * V
                return (-G / (1.0 + nu)) * V

        # Build the elasticity tensor with the typed wrapper algebra: no .data
        # access, no SSR4 ever materialized as a raw torch tensor.
        Iv = SSR4.identity_vol(dtype=K.data.dtype, device=K.data.device)
        Id = SSR4.identity_dev(dtype=G.data.dtype, device=G.data.device)
        C = 3.0 * K * Iv + 2.0 * G * Id

        if v is None:
            return C

        # Differential pushforward:
        #   dC = 3 dK * I_vol + 2 dG * I_dev
        # where dK and dG are scalar tangents in c0/c1 obtained from
        # dK_action / dG_action above. The action is pure typed-wrapper algebra
        # in SSR4; we never form the full n_out x n_in Jacobian.
        actions: dict = {}
        if a in self._promoted_params:
            nl_name_a = self._promoted_params[a].input_name

            def action_a(V: Scalar, _a=a, _Iv=Iv, _Id=Id) -> SSR4:
                return 3.0 * dK_action(V, _a) * _Iv + 2.0 * dG_action(V, _a) * _Id

            actions[nl_name_a] = action_a
        if b in self._promoted_params:
            nl_name_b = self._promoted_params[b].input_name

            def action_b(V: Scalar, _b=b, _Iv=Iv, _Id=Id) -> SSR4:
                return 3.0 * dK_action(V, _b) * _Iv + 2.0 * dG_action(V, _b) * _Id

            actions[nl_name_b] = action_b

        return C, self.apply_chain_rule(v, "C", actions, output=C)
