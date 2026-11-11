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
    $\dot{c} = -K \left( \left\langle c_0 - c \right\rangle + \delta \right)^{1/3}
    \left( c - c_{eq} \right)$
    where $c$ is the matrix concentration, $c_0$ is the initial
    concentration, $c_{eq}$ is the equilibrium concentration, $K$ is the
    depletion coefficient and $\delta$ is the seed concentration.

    Starting from $c = c_0 > c_{eq}$ the rate is negative, so the matrix
    drains monotonically toward $c_{eq}$, the only fixed point.

    The factor $\left( c_0 - c \right)^{1/3}$ stands in for the precipitate
    radius: the solute removed from the matrix is proportional to the
    precipitate volume. Without a seed it vanishes at the physical initial
    condition $c = c_0$, which makes $c = c_0$ a spurious (non-Lipschitz)
    fixed point -- the integrator never leaves it. The seed $\delta > 0$ is
    the solute already held in precipitate nuclei at the start; it keeps the
    rate nonzero and its derivative finite at $c = c_0$. It must be
    strictly positive.

    The Macaulay bracket keeps the cube root on its real branch if the state
    overshoots $c_0$, which is reachable mid-Newton even though the solution
    itself satisfies $c \le c_0$.

    $K$, $c_0$, $c_{eq}$ and $\delta$ are promotable parameters, so any may
    be supplied as a constant or driven by another model -- an
    ``ArrheniusParameter`` or an interpolation for the usual temperature
    dependence of $K$ and $c_{eq}$.
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
        parameter(
            "seed_concentration",
            Scalar,
            "Solute held in precipitate nuclei at the start; must be strictly positive",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the four parameters and the derived
    # ``solute_concentration_rate`` output name (stored on ``self._c_rate``).
    depletion_coefficient: Scalar
    initial_concentration: Scalar
    equilibrium_concentration: Scalar
    seed_concentration: Scalar
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
        delta = self._get_param("seed_concentration", promoted_params, Scalar)

        # The seed keeps the base >= delta > 0, so the cube root and its
        # derivative are finite everywhere (including c >= c0) and no
        # branch masking is needed for either forward or reverse mode.
        raw_gap = c0 - c
        base = macaulay(raw_gap) + delta
        cube_root = pow(base, 1.0 / 3.0)
        driving = c - ceq
        c_dot = -K * cube_root * driving

        if v is None:
            return c_dot

        # Differential pushforward. Closed-form coefficients of typed Scalars;
        # no Jacobian materialised.
        #
        # Write b = <c0 - c> + delta and u = b^(1/3), so du/db = 1/3 b^(-2/3)
        # and db/dc = -H, db/dc0 = H, db/ddelta = 1 with H = H(c0 - c). H uses
        # the same (sign + 1)/2 convention as ``macaulay``, so H = 1/2 at the
        # kink c = c0 and the pushforward matches autograd of the primal there.
        # Then
        #   d c_dot / d c     = -K * (u - driving * du/db * H)
        #   d c_dot / d c0    = -K * driving * du/db * H
        #   d c_dot / d delta = -K * driving * du/db
        #   d c_dot / d ceq   =  K * u
        #   d c_dot / d K     = -u * driving
        H = heaviside(raw_gap)
        du = pow(base, -2.0 / 3.0) / 3.0

        actions: dict[str, ChainRuleAction] = {
            "solute_concentration": lambda V, k=-K * (cube_root - driving * du * H): k * V,
        }
        promoted = self._promoted_params
        if "depletion_coefficient" in promoted:
            actions[promoted["depletion_coefficient"].input_name] = (
                lambda V, k=-cube_root * driving: k * V
            )
        if "initial_concentration" in promoted:
            actions[promoted["initial_concentration"].input_name] = (
                lambda V, k=-K * driving * du * H: k * V
            )
        if "equilibrium_concentration" in promoted:
            actions[promoted["equilibrium_concentration"].input_name] = lambda V, k=K * cube_root: (
                k * V
            )
        if "seed_concentration" in promoted:
            actions[promoted["seed_concentration"].input_name] = lambda V, k=-K * driving * du: (
                k * V
            )

        return c_dot, self.apply_chain_rule(v, self._c_rate, actions, output=c_dot)


__all__ = ["SoluteDepletion"]
