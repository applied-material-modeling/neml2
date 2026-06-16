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

"""Python-native mirror of the C++ ``KozenyCarmanPermeability`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, pow
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("KozenyCarmanPermeability")
class KozenyCarmanPermeability(Model):
    r"""Define the relationship between non-dimensionalized porosity and permeability.

    The Kozeny-Carman porosity-permeability relation takes the form of
    $K = K_0 \frac{\varphi^n (1-\varphi_0^m)}{\varphi_0^m (1-\varphi)^n}$
    where $n$ and $m$ are shape parameters. $\varphi_0$ and
    $K_0$ are the reference porosity and permeability respectively.
    """

    hit = HitSchema(
        input("porosity", Scalar, "The porosity"),
        output("permeability", Scalar, "The porosity-dependent permeability"),
        parameter(
            "reference_permeability",
            Scalar,
            "the reference permeability",
            attr="K0",
        ),
        parameter(
            "reference_porosity",
            Scalar,
            "the reference porosity",
            attr="phi0",
        ),
        parameter("n", Scalar, "Shape parameter n", attr="n"),
        parameter("m", Scalar, "Shape parameter m", attr="m"),
    )

    # ``from_hit`` auto-declares the four parameters; annotate so pyright sees
    # the typed wrappers that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    K0: Scalar
    phi0: Scalar
    n: Scalar
    m: Scalar

    def forward(  # type: ignore[override]
        self,
        porosity: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        phi = porosity
        K0 = self.K0
        phi0 = self.phi0
        n = self.n
        m = self.m

        # Note from the C++ docstring (verbatim): the formula reads
        # K = K0 * phi^n * (1 - phi0^m) / (phi0^m * (1 - phi)^n). The actual
        # set_value implementation uses the algebraically equivalent
        # "(phi/(1-phi))^n / (phi0/(1-phi0))^m" form; mirror that since the
        # in-tree fixture's reference output values were computed against it.
        ratio = phi / (1.0 - phi)
        ratio0 = phi0 / (1.0 - phi0)
        K = K0 * pow(ratio, n) / pow(ratio0, m)

        if v is None:
            return K

        # Differential pushforward. With f(phi) = (phi/(1-phi))^n,
        #   df/dphi = n * (phi/(1-phi))^(n-1) * 1/(1-phi)^2,
        # so dK/dphi = K0 / ratio0^m * n * ratio^(n-1) / (1-phi)^2.
        one_minus_phi = 1.0 - phi
        dK_dphi = K0 / pow(ratio0, m) * (n * pow(ratio, n - 1.0)) / one_minus_phi / one_minus_phi
        actions: dict[str, ChainRuleAction] = {
            "porosity": lambda V, c=dK_dphi: c * V,
        }

        return K, self.apply_chain_rule(
            v,
            "permeability",
            actions,
            output=K,
        )


__all__ = ["KozenyCarmanPermeability"]
