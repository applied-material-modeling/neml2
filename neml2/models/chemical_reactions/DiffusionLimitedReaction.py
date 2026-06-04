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

"""Python-native mirror of the C++ ``DiffusionLimitedReaction`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar


@register_neml2_object("DiffusionLimitedReaction")
class DiffusionLimitedReaction(Model):
    r"""Diffusion-limited reaction rate for a shrinking-core product phase.

    Given the dimensionless inner radius :math:`r_i` and outer radius
    :math:`r_o` of the product phase, the liquid and solid reactivities
    :math:`R_l` and :math:`R_s` (both in :math:`[0, 1]`), the characteristic
    diffusion coefficient :math:`D` of the rate-limiting species, and the
    molar volume :math:`\omega` of that species, the volumetric reaction
    rate is

    .. math::

        \dot{\alpha} = \frac{2 D R_l R_s}{\omega}
                       \cdot \frac{r_o}{r_o - r_i + \delta}

    where :math:`\delta` is a small "dummy thickness" added to the product
    thickness :math:`r_o - r_i` to keep the rate finite at the start of the
    reaction.
    """

    hit = HitSchema(
        input("product_inner_radius", Scalar, "Inner radius of the product phase"),
        input("solid_inner_radius", Scalar, "Inner radius of the solid phase"),
        input("liquid_reactivity", Scalar, "Reactivity of the liquid phase, between 0 and 1"),
        input("solid_reactivity", Scalar, "Reactivity of the solid phase, between 0 and 1"),
        output("reaction_rate", Scalar, "Product phase substance volumetric rate of change"),
        parameter(
            "diffusion_coefficient",
            Scalar,
            "Diffusion coefficient of the rate-limiting species in the product phase",
            attr="D",
            allow_nonlinear=True,
        ),
        option(
            "product_dummy_thickness",
            float,
            "Minimum product thickness to avoid division by 0",
            default=0.01,
            attr="delta",
        ),
        option(
            "molar_volume",
            float,
            "Molar volume of the rate-limiting (liquid) species",
            attr="omega",
        ),
    )

    # ``from_hit`` auto-declares the ``diffusion_coefficient`` parameter
    # (stored as ``D``) and lands the ``delta`` / ``omega`` options on
    # ``self``. The annotations let pyright see the typed Scalar wrapper /
    # plain floats that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s ``Module`` hint.
    D: Scalar
    delta: float
    omega: float

    def forward(  # type: ignore[override]
        self,
        product_inner_radius: Scalar,
        solid_inner_radius: Scalar,
        liquid_reactivity: Scalar,
        solid_reactivity: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        ri = product_inner_radius
        ro = solid_inner_radius
        R_l = liquid_reactivity
        R_s = solid_reactivity
        D = self._get_param("D", nl_params, Scalar)
        delta = self.delta
        omega = self.omega

        factor = 2.0 * D * R_l * R_s / omega
        thickness = ro - ri + delta  # ro - ri + delta (Scalar)
        ratio = ro / thickness
        rate = factor * ratio
        if v is None:
            return rate

        # Differential pushforward. The factor / thickness / ratio
        # algebra above is linear in each contribution, so each ``∂rate/∂x``
        # is a Scalar coefficient that scales the incoming Scalar tangent ``V``.
        # No Jacobian is materialized.
        drate = factor / thickness / thickness
        # ∂rate/∂r_i = factor * r_o / thickness^2
        drate_dri = drate * ro
        # ∂rate/∂r_o = factor * (thickness - r_o) / thickness^2
        #            = factor * (delta - r_i) / thickness^2
        drate_dro = drate * (-ri + delta)
        # ∂rate/∂R_l = 2 D R_s / omega * ratio
        drate_dRl = 2.0 * D * R_s / omega * ratio
        # ∂rate/∂R_s = 2 D R_l / omega * ratio
        drate_dRs = 2.0 * D * R_l / omega * ratio
        # ∂rate/∂D   = 2 R_l R_s / omega * ratio  (only flows if D promoted)
        drate_dD = 2.0 * R_l * R_s / omega * ratio

        actions = {
            "product_inner_radius": lambda V, c=drate_dri: c * V,
            "solid_inner_radius": lambda V, c=drate_dro: c * V,
            "liquid_reactivity": lambda V, c=drate_dRl: c * V,
            "solid_reactivity": lambda V, c=drate_dRs: c * V,
        }
        if "D" in self._nl_params:
            actions[self._nl_params["D"].input_name] = lambda V, c=drate_dD: c * V
        return rate, self.apply_chain_rule(v, "reaction_rate", actions, output=rate)


__all__ = ["DiffusionLimitedReaction"]
