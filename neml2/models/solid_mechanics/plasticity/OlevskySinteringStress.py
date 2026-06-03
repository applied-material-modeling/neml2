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

"""Python-native mirror of the C++ ``OlevskySinteringStress`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar


@register_native("OlevskySinteringStress")
class OlevskySinteringStress(Model):
    r"""Define the Olevsky-Skorohod sintering stress to be used in conjunction with poroplasticity
    yield functions such as the GTNYieldFunction. The sintering stress is defined as
    $\sigma_s = 3 \dfrac{\gamma}{r} \phi^2$, where $\gamma$ is the surface tension,
    $r$ is the size of the particles/powders, and $\phi$ is the void fraction.
    """

    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input("void_fraction", Scalar, "Void fraction"),
        output("sintering_stress", Scalar, "Sintering stress"),
        parameter("surface_tension", Scalar, "Surface tension", attr="gamma", allow_nonlinear=True),
        parameter("particle_radius", Scalar, "Particle radius", attr="r", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the ``surface_tension`` / ``particle_radius``
    # parameters (stored as ``gamma`` / ``r``) -- no __init__ needed.
    gamma: Scalar
    r: Scalar

    def forward(  # type: ignore[override]
        self,
        void_fraction: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Mirrors ``OlevskySinteringStress::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/OlevskySinteringStress.cxx``.
        phi = void_fraction
        gamma = self._get_param("gamma", nl_params, Scalar)
        r = self._get_param("r", nl_params, Scalar)

        # s = 3 * gamma * phi^2 / r
        s = 3.0 * gamma * phi * phi / r
        if v is None:
            return s

        # Differential pushforward.
        # First-order partials of s w.r.t. (phi, gamma, r):
        #   ds/dphi   = 6 * gamma * phi / r
        #   ds/dgamma = 3 * phi^2 / r
        #   ds/dr     = -3 * gamma * phi^2 / r^2
        coef_phi = 6.0 * gamma * phi / r
        actions_1 = {"void_fraction": lambda V, c=coef_phi: c * V}
        if "gamma" in self._nl_params:
            coef_gamma = 3.0 * phi * phi / r
            actions_1[self._nl_params["gamma"].input_name] = lambda V, c=coef_gamma: c * V
        if "r" in self._nl_params:
            coef_r = -3.0 * gamma * phi * phi / (r * r)
            actions_1[self._nl_params["r"].input_name] = lambda V, c=coef_r: c * V

        if v2 is None and vh is None:
            return s, *self.propagate_tangents(v, "sintering_stress", actions_1, output=s)

        # Second-order pushforward. Cross-bilinear forms (each takes two
        # leading-K tangents Va, Vb and returns the contribution to the
        # second-order tangent of s; the (N_a, N_b) outer-product axes are
        # intrinsic to a Hessian-times-(Va, Vb)). The Hessian entries are:
        #   d2s/dphi2         = 6 * gamma / r
        #   d2s/dphi dgamma   = 6 * phi / r
        #   d2s/dphi dr       = -6 * gamma * phi / r^2
        #   d2s/dgamma2       = 0
        #   d2s/dgamma dr     = -3 * phi^2 / r^2
        #   d2s/dr2           = 6 * gamma * phi^2 / r^3
        # each action_2 receives primal-shape
        # tangents and returns the primal-shape bilinear; the framework
        # iterates the (N_a, N_b) seed-pair outer + stacks.
        actions_2: dict = {}
        c_pp = 6.0 * gamma / r
        actions_2[("void_fraction", "void_fraction")] = lambda Va, Vb, c=c_pp: c * Va * Vb

        if "gamma" in self._nl_params:
            gname = self._nl_params["gamma"].input_name
            c_pg = 6.0 * phi / r
            actions_2[("void_fraction", gname)] = lambda Va, Vb, c=c_pg: c * Va * Vb
            actions_2[(gname, "void_fraction")] = lambda Va, Vb, c=c_pg: c * Va * Vb
        if "r" in self._nl_params:
            rname = self._nl_params["r"].input_name
            c_pr = -6.0 * gamma * phi / (r * r)
            actions_2[("void_fraction", rname)] = lambda Va, Vb, c=c_pr: c * Va * Vb
            actions_2[(rname, "void_fraction")] = lambda Va, Vb, c=c_pr: c * Va * Vb
            c_rr = 6.0 * gamma * phi * phi / (r * r * r)
            actions_2[(rname, rname)] = lambda Va, Vb, c=c_rr: c * Va * Vb
        if "gamma" in self._nl_params and "r" in self._nl_params:
            gname = self._nl_params["gamma"].input_name
            rname = self._nl_params["r"].input_name
            c_gr = -3.0 * phi * phi / (r * r)
            actions_2[(gname, rname)] = lambda Va, Vb, c=c_gr: c * Va * Vb
            actions_2[(rname, gname)] = lambda Va, Vb, c=c_gr: c * Va * Vb

        return s, *self.propagate_tangents(
            v,
            "sintering_stress",
            actions_1,
            output=s,
            v2=v2,
            actions_2=actions_2,
            vh=vh,
        )


__all__ = ["OlevskySinteringStress"]
