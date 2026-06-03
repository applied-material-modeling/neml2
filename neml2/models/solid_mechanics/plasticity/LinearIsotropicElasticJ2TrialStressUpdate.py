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

"""Python-native mirror of the C++ ``LinearIsotropicElasticJ2TrialStressUpdate``.

Scalar return-mapping helper for J2 plasticity with isotropic linear
elasticity: $sigma_trial = sigma_e_trial - 3G * (ep - ep_n)$. G is derived
from the (E, nu) pair supplied via the C++ ``coefficients`` /
``coefficient_types`` knobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, derived_input, input, option, output

# Avoid the bare ``input`` shadowing the builtin in type hints below.
from ....types import Scalar

if TYPE_CHECKING:
    import nmhit

    from ....factory import _NativeInputFile


@register_native("LinearIsotropicElasticJ2TrialStressUpdate")
class LinearIsotropicElasticJ2TrialStressUpdate(Model):
    r"""Update the trial stress under the assumptions of J2 plasticity and
    isotropic linear elasticity.

    This allows the construction of fully scalar return mapping models for
    isotropic materials.
    """

    hit = HitSchema(
        input(
            "elastic_trial_stress",
            Scalar,
            "Initial trial stress assuming a purely elastic step",
        ),
        input(
            "equivalent_plastic_strain",
            Scalar,
            "Current guess for the equivalent plastic strain",
        ),
        # The previous-step value of ``equivalent_plastic_strain`` is
        # auto-declared by appending the ``~1`` history suffix (matches the
        # C++ ``history_name(_inelastic_strain.name(), /*nstep=*/1)``). No
        # HIT knob.
        derived_input(
            "equivalent_plastic_strain",
            Scalar,
            attr="_ep_old_name",
            suffix="~1",
        ),
        output(
            "updated_trial_stress",
            Scalar,
            "Trial stress corrected for the current increment of plastic deformation",
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

    # ``_ep_old_name`` carries the resolved ``<equivalent_plastic_strain>~1``
    # history input name.
    _ep_old_name: str
    E: Scalar
    nu: Scalar

    def __init__(
        self,
        E=None,
        nu=None,
        *,
        factory: _NativeInputFile | None = None,
        **hit_values,
    ) -> None:
        # ``E`` and ``nu`` are extracted from the HIT ``coefficients`` list by
        # ``from_hit``; declare them explicitly here. All other schema-derived
        # values (input/output renames) pass through to the base.
        super().__init__(**hit_values)
        if E is not None:
            self.declare_typed_parameter("E", E, Scalar, factory=factory, allow_nonlinear=True)
        if nu is not None:
            self.declare_typed_parameter("nu", nu, Scalar, factory=factory, allow_nonlinear=True)

    @classmethod
    def from_hit(
        cls,
        node: nmhit.Node,
        factory: _NativeInputFile,
    ) -> LinearIsotropicElasticJ2TrialStressUpdate:
        coeffs = node.param_list_str("coefficients")
        types = node.param_list_str("coefficient_types")
        # This native port mirrors the (E, nu) parameterisation used by the
        # sibling ``LinearIsotropicElasticity`` leaf — the canonical pair for
        # the existing scalar return-map fixtures.
        e_idx = types.index("YOUNGS_MODULUS")
        nu_idx = types.index("POISSONS_RATIO")
        schema_kwargs = cls.hit.kwargs_from_hit(node, factory)
        # ``coefficients`` and ``coefficient_types`` were consumed manually.
        schema_kwargs.pop("coefficients", None)
        schema_kwargs.pop("coefficient_types", None)
        return cls(
            E=coeffs[e_idx],
            nu=coeffs[nu_idx],
            factory=factory,
            **schema_kwargs,
        )

    @staticmethod
    def _shear_modulus(E: Scalar, nu: Scalar) -> Scalar:
        # G = E / (2 (1 + nu)) -- typed Scalar algebra, no .data.
        return E / (2.0 * (1.0 + nu))

    def forward(  # type: ignore[override]
        self,
        elastic_trial_stress: Scalar,
        equivalent_plastic_strain: Scalar,
        equivalent_plastic_strain_old: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        E = self._get_param("E", nl_params, Scalar)
        nu = self._get_param("nu", nl_params, Scalar)
        G = self._shear_modulus(E, nu)
        three_G = 3.0 * G
        dep = equivalent_plastic_strain - equivalent_plastic_strain_old
        sigma = elastic_trial_stress - three_G * dep

        if v is None:
            return sigma

        # D-062 pushforward. The forward is linear in (sigma_e_trial, ep, ep_n)
        # and rational in (E, nu). Closed-form coefficients (Scalars in K):
        #   d sigma / d sigma_e_trial   = +1
        #   d sigma / d ep              = -3G
        #   d sigma / d ep_n            = +3G
        #   d sigma / d E               = -3 * dG/dE * dep      with dG/dE = G/E
        #   d sigma / d nu              = -3 * dG/dnu * dep     with
        #                                 dG/dnu = -G / (1 + nu)
        actions = {
            "elastic_trial_stress": lambda V: V,
            "equivalent_plastic_strain": lambda V, c=-three_G: c * V,
            self._ep_old_name: lambda V, c=three_G: c * V,
        }
        if "E" in self._nl_params:
            dG_dE = G / E
            coef_E = -3.0 * dG_dE * dep
            actions[self._nl_params["E"].input_name] = lambda V, c=coef_E: c * V
        if "nu" in self._nl_params:
            dG_dnu = -G / (1.0 + nu)
            coef_nu = -3.0 * dG_dnu * dep
            actions[self._nl_params["nu"].input_name] = lambda V, c=coef_nu: c * V

        return sigma, self.apply_chain_rule(v, "updated_trial_stress", actions, output=sigma)


__all__ = ["LinearIsotropicElasticJ2TrialStressUpdate"]
