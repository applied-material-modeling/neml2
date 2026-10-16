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

"""Solute depletion rate, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, heaviside, macaulay, pow
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SoluteDepletion")
class SoluteDepletion(Model):
    r"""Depletion of a solid-solution strengthening species as it is taken out
    of the matrix by precipitation, defined by
    $\dot{c} = K \left\langle c_0 - c \right\rangle^{1/3} \left( c - c_{eq} \right)$
    where $c$ is the matrix concentration, $c_0$ is the initial
    concentration, $c_{eq}$ is the equilibrium concentration and $K$
    is the depletion coefficient.

    $K$ and $c_{eq}$ are promotable parameters, so either may be supplied
    as a constant or driven by another model -- an ``ArrheniusParameter`` or an
    interpolation for the usual temperature dependence of $c_{eq}$.

    The Macaulay bracket on $c_0 - c$ is a deliberate departure from the bare
    $\left( c_0 - c \right)^{1/3}$ of the specification: it keeps the cube
    root on its real branch if the state overshoots $c_0$, which is
    reachable mid-Newton even when the converged solution satisfies
    $c \le c_0$. The two agree wherever $c \le c_0$. See the
    ``numerical notes`` section of the model documentation.
    """

    hit = HitSchema(
        input("solute_concentration", Scalar, "Current matrix solute concentration"),
        derived_output("solute_concentration", Scalar, attr="_c_rate", suffix="_rate"),
        parameter("depletion_coefficient", Scalar, "Depletion coefficient", allow_promotion=True),
        parameter("initial_concentration", Scalar, "Initial concentration", allow_promotion=True),
        parameter(
            "equilibrium_concentration",
            Scalar,
            "Equilibrium concentration, generally a function of temperature",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the three parameters and the derived
    # ``solute_concentration_rate`` output name (stored on ``self._c_rate``).
    depletion_coefficient: Scalar
    initial_concentration: Scalar
    equilibrium_concentration: Scalar
    _c_rate: str

    def forward(  # type: ignore[override]
        self,
        c: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        K = self._get_param("depletion_coefficient", promoted_params, Scalar)
        c0 = self._get_param("initial_concentration", promoted_params, Scalar)
        ceq = self._get_param("equilibrium_concentration", promoted_params, Scalar)

        raw_gap = c0 - c
        gap = macaulay(raw_gap)
        cube_root = pow(gap, 1.0 / 3.0)
        driving = c - ceq
        c_dot = K * cube_root * driving

        if v is None:
            return c_dot

        # Differential pushforward. Closed-form coefficients of typed Scalars;
        # no Jacobian materialised.
        #
        # Write g = <c0 - c>, so dg/dc = -H(c0 - c) and dg/dc0 = +H(c0 - c),
        # with H the Heaviside step. Then with u = g^(1/3),
        #   du/dg = 1/3 g^(-2/3)
        # and
        #   d c_dot / d c   = K * (u - (driving / 3) * g^(-2/3) * H)
        #   d c_dot / d c0  = K * driving / 3 * g^(-2/3) * H
        #   d c_dot / d ceq = -K * u
        #   d c_dot / d K   = u * driving
        #
        # ``heaviside`` gives H directly. Offsetting the gap by (1 - H) makes the
        # base of the negative power 1.0 on the inactive branch, so g^(-2/3) stays
        # finite there and the H factor then zeroes the contribution -- this avoids
        # an inf * 0 = nan. At exactly c == c0 the repo's ``heaviside`` returns 0.5,
        # so the base is 0.5 and the result is finite: the true derivative is
        # unbounded at that single point, and this picks a finite subgradient
        # rather than propagating a nan into the Newton solve.
        H = heaviside(raw_gap)
        safe_gap = gap + (1.0 - H)
        dgap = H * pow(safe_gap, -2.0 / 3.0) / 3.0

        actions: dict[str, ChainRuleAction] = {
            "solute_concentration": lambda V, c_=K * (cube_root - driving * dgap): c_ * V,
        }
        if "depletion_coefficient" in self._promoted_params:
            actions[self._promoted_params["depletion_coefficient"].input_name] = (
                lambda V, c_=cube_root * driving: c_ * V
            )
        if "initial_concentration" in self._promoted_params:
            actions[self._promoted_params["initial_concentration"].input_name] = (
                lambda V, c_=K * driving * dgap: c_ * V
            )
        if "equilibrium_concentration" in self._promoted_params:
            actions[self._promoted_params["equilibrium_concentration"].input_name] = (
                lambda V, c_=-K * cube_root: c_ * V
            )

        return c_dot, self.apply_chain_rule(v, self._c_rate, actions, output=c_dot)


__all__ = ["SoluteDepletion"]
