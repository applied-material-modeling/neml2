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

"""Python-native mirror of the C++ ``PowerLawPermeability`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, pow
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("PowerLawPermeability")
class PowerLawPermeability(Model):
    r"""Power-law porosity-permeability relation
    $K = K_0 \left( \frac{\varphi}{\varphi_0} \right)^p$ where
    $\varphi_0$ and $K_0$ are the reference porosity and
    permeability respectively, and $p$ is the power-law exponent.
    """

    hit = HitSchema(
        input("porosity", Scalar, "Porosity"),
        output("permeability", Scalar, "Permeability"),
        parameter(
            "reference_permeability",
            Scalar,
            "The reference permeability",
            attr="K0",
        ),
        parameter(
            "reference_porosity",
            Scalar,
            "The reference porosity",
            attr="phi0",
        ),
        parameter(
            "exponent",
            Scalar,
            "The exponent in the power law",
            attr="p",
        ),
    )

    # ``from_hit`` auto-declares the three parameters; annotate so pyright sees
    # the typed wrappers that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    K0: Scalar
    phi0: Scalar
    p: Scalar

    def forward(  # type: ignore[override]
        self,
        porosity: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        phi = porosity
        K0 = self._get_param("K0", nl_params, Scalar)
        phi0 = self._get_param("phi0", nl_params, Scalar)
        p = self._get_param("p", nl_params, Scalar)

        # ``K = K0 * (phi / phi0) ** p`` — typed Scalar algebra end to end.
        ratio = phi / phi0
        K = K0 * pow(ratio, p)

        if v is None:
            return K

        # Differential pushforward: dK/dphi = K0 * p * (phi/phi0)^(p-1) / phi0.
        dK_dphi = K0 * p * pow(ratio, p - 1.0) / phi0
        actions: dict[str, ChainRuleAction] = {
            "porosity": lambda V, c=dK_dphi: c * V,
        }

        return K, self.apply_chain_rule(
            v,
            "permeability",
            actions,
            output=K,
        )


__all__ = ["PowerLawPermeability"]
