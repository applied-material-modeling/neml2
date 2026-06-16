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

"""Python-native mirror of the C++ ``LinearKinematicHardening`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import SR2, Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("LinearKinematicHardening")
class LinearKinematicHardening(Model):
    r"""Map kinematic plastic strain to back stress following a linear
    relationship, i.e., $\boldsymbol{X} = H \boldsymbol{K}_p$ where
    $H$ is the hardening modulus.
    """

    hit = HitSchema(
        input("kinematic_plastic_strain", SR2, "Kinematic plastic strain"),
        output("back_stress", SR2, "Back stress"),
        parameter(
            "hardening_modulus",
            Scalar,
            "Hardening modulus",
            attr="H",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``hardening_modulus`` parameter (stored as
    # ``H``) -- no __init__ needed.
    H: Scalar

    def forward(  # type: ignore[override]
        self,
        kinematic_plastic_strain: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        Kp = kinematic_plastic_strain
        H = self._get_param("H", nl_params, Scalar)
        X = H * Kp
        if v is None:
            return X
        # Differential pushforward: X = H * Kp is linear in both Kp and
        # H. The action in Kp is just H scaling the SR2 tangent; the action in
        # H (when promoted to a nonlinear parameter input) is Kp scaling the
        # incoming Scalar tangent.
        actions = {"kinematic_plastic_strain": lambda V, c=H: c * V}
        if "H" in self._nl_params:
            actions[self._nl_params["H"].input_name] = lambda V, c=Kp: c * V
        return X, self.apply_chain_rule(v, "back_stress", actions, output=X)
