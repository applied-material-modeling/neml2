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

"""Python-native mirror of the C++ ``GeneralElasticity`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import (
    SR2,
    SSR4,
    Rot,
    euler_rodrigues,
    jvp_euler_rodrigues,
    jvp_rotate,
    rotate,
)


@register_native("GeneralElasticity")
class GeneralElasticity(Model):
    """``stress = (T.rotate(R)) : strain`` with ``T : SSR4`` the lab-frame stiffness.

    $R$ is the orientation matrix derived from the input ``orientation : Rot``
    via Euler-Rodrigues. Mirrors C++ ``GeneralElasticity`` in
    ``src/neml2/models/solid_mechanics/elasticity/GeneralElasticity.cxx``.
    $T$ is a parameter (HIT option ``elastic_stiffness_tensor``); declared
    nonlinear-capable to mirror the C++ ``declare_parameter`` flag, but Taylor
    uses it as a plain buffer.

    Chain-rule actions cover ``strain``, ``orientation``, and the parameter
    $T$ (the last only when promoted to a nonlinear input).
    """

    hit = HitSchema(
        input("strain", SR2, "Elastic strain"),
        input("orientation", Rot, "Active convention orientation from reference to current"),
        output("stress", SR2, "Stress"),
        parameter(
            "elastic_stiffness_tensor",
            SSR4,
            "Elastic stiffness tensor",
            attr="T",
            allow_nonlinear=True,
        ),
    )

    # Constructed via the schema: ``from_hit`` auto-declares the
    # ``elastic_stiffness_tensor`` parameter (stored as ``T``), so no __init__.
    T: SSR4

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        orientation: Rot,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        T = self._get_param("T", nl_params, SSR4)
        R = euler_rodrigues(orientation)
        T_rot = rotate(T, R)
        stress = T_rot @ strain
        if v is None:
            return stress

        # Differential pushforwards, pure typed-wrapper algebra:
        #   strain:      dσ = T_rot : dε
        #   orientation: dσ = (∂T_rot/∂R : ε) pushed through dR = ∂R/∂r · dr,
        #                via the typed JVP primitives jvp_euler_rodrigues /
        #                jvp_rotate (which hide the only irreducible
        #                geometric contractions). No leaf-level Jacobian formed.
        def strain_action(V: SR2) -> SR2:
            return T_rot @ V

        def orientation_action(V: Rot) -> SR2:
            dR = jvp_euler_rodrigues(orientation, V)
            return jvp_rotate(T, R, dR) @ strain

        actions: dict = {"strain": strain_action, "orientation": orientation_action}
        return stress, self.apply_chain_rule(v, "stress", actions, output=stress)
