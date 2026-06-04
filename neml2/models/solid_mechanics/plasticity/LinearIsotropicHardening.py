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

"""Python-native mirror of the C++ ``LinearIsotropicHardening`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar


@register_neml2_object("LinearIsotropicHardening")
class LinearIsotropicHardening(Model):
    r"""Map equivalent plastic strain to isotropic hardening following a linear
    relationship, i.e., $h = K \bar{\varepsilon}_p$ where $K$ is
    the hardening modulus.
    """

    hit = HitSchema(
        input("equivalent_plastic_strain", Scalar, "Equivalent plastic strain"),
        output("isotropic_hardening", Scalar, "Isotropic hardening"),
        parameter(
            "hardening_modulus",
            Scalar,
            "Hardening modulus",
            attr="K",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``hardening_modulus`` parameter (stored as
    # ``K``) — no __init__ needed.
    K: Scalar

    def forward(  # type: ignore[override]
        self,
        equivalent_plastic_strain: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        eps = equivalent_plastic_strain
        K = self._get_param("K", nl_params, Scalar)
        h = K * eps
        if v is None:
            return h
        # Differential pushforward: ∂h/∂eps = K, ∂h/∂K = eps — each a
        # Scalar coefficient scaling the incoming Scalar tangent. Leading-K means
        # K / eps broadcast against the tangent's K axis automatically.
        actions = {"equivalent_plastic_strain": lambda V, c=K: c * V}
        if "K" in self._nl_params:
            actions[self._nl_params["K"].input_name] = lambda V, c=eps: c * V
        return h, self.apply_chain_rule(v, "isotropic_hardening", actions, output=h)
