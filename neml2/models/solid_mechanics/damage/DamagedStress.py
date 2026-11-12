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

"""Python-native port of the v2 C++ ``DamagedStress`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("DamagedStress")
class DamagedStress(Model):
    r"""Apply isotropic scalar damage to a stress tensor:

    .. math::
        \boldsymbol{\sigma} = (1 - D)\,\tilde{\boldsymbol{\sigma}}

    where :math:`D \in [0, 1)` is the scalar damage variable and
    :math:`\tilde{\boldsymbol{\sigma}}` is the undamaged (effective) stress.
    Used as the final stage of any continuum-damage-mechanics composition;
    paired with :class:`MazarsDamage` or :class:`MazarsDamageStressAlpha`
    in Mazars CDM input files.
    """

    hit = HitSchema(
        input("damage", Scalar, "Scalar damage variable D"),
        input("effective_stress", SR2, "Undamaged effective stress sigma_tilde"),
        output("stress", SR2, "Nominal (damaged) stress sigma"),
    )

    def forward(  # type: ignore[override]
        self,
        damage: Scalar,
        effective_stress: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Primal: σ = (1 − D) · σ̃
        one_minus_D = Scalar.from_value(1.0, like=damage) - damage
        stress = one_minus_D * effective_stress

        if v is None:
            return stress

        # Closed-form first-order JVPs.
        # ∂σ/∂D   = -σ̃         (Scalar tangent V_D → SR2 result: -V_D · σ̃)
        # ∂σ/∂σ̃   = (1-D)·I⁴ˢ   (SR2 tangent V_σ̃ → SR2 result: (1-D) · V_σ̃)
        # Default-arg binding defeats Python late-binding (CLAUDE.md guidance).
        actions = {
            "damage": (lambda V_D, st=effective_stress: -V_D * st),
            "effective_stress": (lambda V_S, c=one_minus_D: c * V_S),
        }
        return stress, self.apply_chain_rule(v, "stress", actions, output=stress)
