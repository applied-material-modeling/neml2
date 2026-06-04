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

"""Python-native mirror of the C++ ``LinearIsotropicStrainEnergyDensity`` model.

Computes the active and inactive parts of the elastic strain energy density
for a linear isotropic elastic response, with optional volumetric/deviatoric
(``VOLDEV``) split for phase-field fracture. The ``SPECTRAL`` split (which on
the C++ side requires ``linalg::eigh``/``ieigh`` of an SR2) is not yet ported.

Variable names mirror the C++ defaults so the same ``.i`` fixtures drive
both backends: ``strain`` -> (``active_strain_energy_density``,
``inactive_strain_energy_density``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import SR2, Scalar, dev, heaviside, inner, macaulay, tr

if TYPE_CHECKING:
    import nmhit

    from ...factory import _NativeInputFile


@register_native("LinearIsotropicStrainEnergyDensity")
class LinearIsotropicStrainEnergyDensity(Model):
    """Calculates elastic strain energy density based on linear elastic isotropic response"""

    hit = HitSchema(
        input("strain", SR2, "Elastic strain"),
        output(
            "active_strain_energy_density",
            Scalar,
            "Active part of the strain energy density",
        ),
        output(
            "inactive_strain_energy_density",
            Scalar,
            "Inactive part of the strain energy density",
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
        option(
            "decomposition",
            str,
            "Strain energy density decomposition types, options are: VOLDEV, SPECTRAL, NONE",
            default="NONE",
            attr="_decomposition",
        ),
    )

    # K (bulk modulus) and G (shear modulus) are the canonical (K, G) pair the
    # C++ IsotropicElasticityConverter always returns; extracted from the
    # (coefficients, coefficient_types) HIT list and declared explicitly in
    # from_hit so each may be promoted to a nonlinear input.
    K: Scalar
    G: Scalar
    _decomposition: str

    def __init__(
        self,
        K=None,
        G=None,
        *,
        factory: _NativeInputFile | None = None,
        **hit_values,
    ) -> None:
        # Schema-driven hit_values (input/output renames, decomposition) flow
        # to the base; K and G are pulled from coefficients/coefficient_types
        # in from_hit and declared explicitly here so each can be promoted.
        super().__init__(**hit_values)
        if K is not None:
            self.declare_typed_parameter("K", K, Scalar, factory=factory, allow_nonlinear=True)
        if G is not None:
            self.declare_typed_parameter("G", G, Scalar, factory=factory, allow_nonlinear=True)

    @classmethod
    def from_hit(
        cls, node: nmhit.Node, factory: _NativeInputFile
    ) -> LinearIsotropicStrainEnergyDensity:
        coeffs = node.param_list_str("coefficients")
        types = node.param_list_str("coefficient_types")
        # Mirror the C++ IsotropicElasticityConverter: accept (K, G) directly
        # or (E, nu) and convert at construction. Both parameterizations require
        # literal numeric coefficients here (the conversion needs scalar math).
        if "BULK_MODULUS" in types and "SHEAR_MODULUS" in types:
            K = coeffs[types.index("BULK_MODULUS")]
            G = coeffs[types.index("SHEAR_MODULUS")]
        elif "YOUNGS_MODULUS" in types and "POISSONS_RATIO" in types:
            E_spec = coeffs[types.index("YOUNGS_MODULUS")]
            nu_spec = coeffs[types.index("POISSONS_RATIO")]
            try:
                E = float(E_spec)
                nu = float(nu_spec)
            except ValueError as exc:
                raise NotImplementedError(
                    "LinearIsotropicStrainEnergyDensity: (E, nu) parameterization "
                    "currently requires literal numeric coefficients; got "
                    f"{(E_spec, nu_spec)!r}."
                ) from exc
            K = str(E / (3.0 * (1.0 - 2.0 * nu)))
            G = str(E / (2.0 * (1.0 + nu)))
        else:
            raise NotImplementedError(
                "LinearIsotropicStrainEnergyDensity native port supports "
                "coefficient_types = 'BULK_MODULUS SHEAR_MODULUS' or "
                f"'YOUNGS_MODULUS POISSONS_RATIO'; got {types!r}."
            )
        schema_kwargs = cls.hit.kwargs_from_hit(node, factory)
        schema_kwargs.pop("coefficients", None)
        schema_kwargs.pop("coefficient_types", None)
        return cls(K=K, G=G, factory=factory, **schema_kwargs)

    # Quadratic in strain (per branch) ⇒ second-order chain rule has a
    # closed-form bilinear (Hessian SSR4 contracted against (Va, Vb)).
    # Required for ``Normality(energy)`` wraps that consume this leaf
    # inside the inner composition (elastic-brittle-fracture scenario).
    SUPPORTS_SECOND_ORDER = True

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        K = self._get_param("K", nl_params, Scalar)
        G = self._get_param("G", nl_params, Scalar)

        if self._decomposition == "NONE":
            psie_active, psie_inactive, strain_action = self._no_decomposition(strain, K, G)
        elif self._decomposition == "VOLDEV":
            psie_active, psie_inactive, strain_action = self._voldev_decomposition(strain, K, G)
        elif self._decomposition == "SPECTRAL":
            raise NotImplementedError(
                "LinearIsotropicStrainEnergyDensity: SPECTRAL decomposition is not "
                "yet ported to the Python-native backend (requires typed linalg.eigh "
                "and linalg.ieigh primitives)."
            )
        else:
            raise ValueError(
                f"LinearIsotropicStrainEnergyDensity: Unsupported decomposition type "
                f"{self._decomposition!r}; expected one of 'NONE', 'VOLDEV', 'SPECTRAL'."
            )

        if v is None:
            return psie_active, psie_inactive

        # First-order: two outputs, one input direction — pushforward returns
        # one tangent per output. Merge the per-output ChainRuleDicts.
        actions_1_act = {"strain": strain_action[0]}
        actions_1_inact = {"strain": strain_action[1]}

        # Second-order pushforwards.
        #   NONE:   H_act = K · I⊗I + 2G · P_dev ⇒ bilinear
        #           H_act(Va, Vb) = K tr(Va) tr(Vb) + 2G inner(dev(Va), dev(Vb))
        #           H_inact = 0
        #   VOLDEV: H_act has K · H(tr(e)) · I⊗I in the volumetric part
        #           H_inact has K · H(-tr(e)) · I⊗I
        # Pure typed-wrapper algebra; framework handles the (N_a, N_b) outer.
        strain_action_2 = self._strain_action_2(strain, K, G)
        actions_2_act = {("strain", "strain"): strain_action_2[0]}
        actions_2_inact = {("strain", "strain"): strain_action_2[1]}

        if v2 is None and vh is None:
            v_act = self.apply_chain_rule(
                v, "active_strain_energy_density", actions_1_act, output=psie_active
            )
            v_inact = self.apply_chain_rule(
                v, "inactive_strain_energy_density", actions_1_inact, output=psie_inactive
            )
            return psie_active, psie_inactive, {**v_act, **v_inact}

        v2_in = v2 if v2 is not None else {}
        v_act = self.apply_chain_rule(
            v, "active_strain_energy_density", actions_1_act, output=psie_active
        )
        v2_act = self.apply_chain_rule_2(
            v, v2_in, "active_strain_energy_density", actions_1_act, actions_2_act, vh=vh
        )
        v_inact = self.apply_chain_rule(
            v, "inactive_strain_energy_density", actions_1_inact, output=psie_inactive
        )
        v2_inact = self.apply_chain_rule_2(
            v, v2_in, "inactive_strain_energy_density", actions_1_inact, actions_2_inact, vh=vh
        )
        if vh is None:
            return (
                psie_active,
                psie_inactive,
                {**v_act, **v_inact},
                {**v2_act, **v2_inact},
            )
        vh_act = self.apply_chain_rule(
            vh, "active_strain_energy_density", actions_1_act, output=psie_active
        )
        vh_inact = self.apply_chain_rule(
            vh, "inactive_strain_energy_density", actions_1_inact, output=psie_inactive
        )
        return (
            psie_active,
            psie_inactive,
            {**v_act, **v_inact},
            {**v2_act, **v2_inact},
            {**vh_act, **vh_inact},
        )

    def _strain_action_2(self, strain: SR2, K: Scalar, G: Scalar) -> tuple:
        """Build the (active, inactive) bilinear actions ``H(Va, Vb)`` for
        the chosen decomposition. Both lambdas receive primal-shape SR2
        tangents and return primal-shape Scalars — pure typed algebra.
        """
        if self._decomposition == "NONE":

            def act_action_2(Va: SR2, Vb: SR2) -> Scalar:
                return K * tr(Va) * tr(Vb) + 2.0 * G * inner(dev(Va), dev(Vb))

            def inact_action_2(Va: SR2, Vb: SR2) -> Scalar:
                return 0.0 * tr(Va) * tr(Vb)

            return act_action_2, inact_action_2

        # VOLDEV: indicator on which branch is active.
        etr = tr(strain)
        h_pos = heaviside(etr)
        h_neg = heaviside(-etr)

        def act_action_2(Va: SR2, Vb: SR2) -> Scalar:
            return K * h_pos * tr(Va) * tr(Vb) + 2.0 * G * inner(dev(Va), dev(Vb))

        def inact_action_2(Va: SR2, Vb: SR2) -> Scalar:
            return K * h_neg * tr(Va) * tr(Vb)

        return act_action_2, inact_action_2

    # ------------------------------------------------------------------
    # Decomposition kernels (forward value + closed-form strain pushforward)
    # ------------------------------------------------------------------
    #
    # Each helper returns ``(psie_active, psie_inactive, (act_action,
    # inact_action))`` where the actions are the differential-pushforward
    # closures that map an SR2 strain-tangent ``V`` to the matching Scalar
    # tangent. The closures capture K, G, and the already-evaluated forward
    # quantities (etr, edev, etc.) — pure typed wrapper algebra, no .data.

    def _no_decomposition(self, strain: SR2, K: Scalar, G: Scalar) -> tuple[Scalar, Scalar, tuple]:
        etr = tr(strain)
        edev = dev(strain)
        # 0.5 * K * tr(e)^2 + G * (dev(e) : dev(e)); inactive part is zero.
        psie_active = 0.5 * K * etr * etr + G * inner(edev, edev)
        psie_inactive = 0.0 * etr  # structural zero, sub-batch aligned

        # dpsi_active/de = K * tr(e) * I + 2 G dev(e) ; action on V is the
        # inner product of that gradient with V, written entirely in typed
        # wrapper algebra (tr / dev / inner / scalar multiply).
        def act_action(V: SR2) -> Scalar:
            return K * etr * tr(V) + 2.0 * G * inner(edev, dev(V))

        def inact_action(V: SR2) -> Scalar:
            return 0.0 * tr(V)

        return psie_active, psie_inactive, (act_action, inact_action)

    def _voldev_decomposition(
        self, strain: SR2, K: Scalar, G: Scalar
    ) -> tuple[Scalar, Scalar, tuple]:
        etr = tr(strain)
        edev = dev(strain)
        # Macaulay split on the trace:
        #   etr_pos = <etr>_+ ,  etr_neg = etr - etr_pos = -<-etr>_+
        etr_pos = macaulay(etr)
        etr_neg = etr - etr_pos
        psie_active = 0.5 * K * etr_pos * etr_pos + G * inner(edev, edev)
        psie_inactive = 0.5 * K * etr_neg * etr_neg

        # d/de [ 0.5 K <tr(e)>_+^2 ] = K <tr(e)>_+ * I  (the H(tr(e)) factor is
        # absorbed because either etr_pos == etr or etr_pos == 0 at the gradient
        # site; the resulting action picks the volumetric/deviatoric V-part).
        # d/de [ G dev(e):dev(e) ] = 2 G dev(e) ; combined ⇒ action on V:
        #   da = K * etr_pos * tr(V) + 2 G dev(e):dev(V)
        # Similarly for the inactive (negative) branch.
        def act_action(V: SR2) -> Scalar:
            return K * etr_pos * tr(V) + 2.0 * G * inner(edev, dev(V))

        def inact_action(V: SR2) -> Scalar:
            return K * etr_neg * tr(V)

        return psie_active, psie_inactive, (act_action, inact_action)


__all__ = ["LinearIsotropicStrainEnergyDensity"]
