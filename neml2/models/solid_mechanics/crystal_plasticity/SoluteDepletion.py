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
from ....types import Scalar, gt, macaulay, pow, where
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
    $c \le c_0$. The two agree wherever $c \le c_0$.

    On that inactive branch, and at the endpoint $c = c_0$, both the rate
    and its derivative are taken to be zero. The true derivative is unbounded
    at the endpoint; the right-hand limit is used instead so that neither the
    forward pushforward nor reverse-mode differentiation of this model emits
    a ``nan`` into the Newton solve, and so that the two agree.
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

        # On the inactive branch (c >= c0) the bracket is exactly zero, and
        # ``pow``'s backward at a zero base with a fractional exponent is
        # infinite. Multiplied by the bracket's own zero derivative that is a
        # nan, so the guard has to sit in the PRIMAL -- ``param_jacobian`` and
        # ``param_vjp`` in ``models/param_ad.py`` differentiate the primal with
        # autograd and never see ``v``, so a guard applied only to the
        # handwritten pushforward below would leave them broken.
        #
        # The fix is the double-``where``: feed ``pow`` a base of 1.0 wherever
        # the branch is inactive, so its local derivative stays finite, then
        # mask the result to zero. The value is unchanged (the bracket is zero
        # there anyway) and the gradient is a clean zero in both modes.
        #
        # The mask is strict (``c0 - c > 0``), which fixes the policy at the
        # endpoint ``c == c0``: value zero, derivative zero, forward and
        # reverse agreeing. The true derivative is unbounded at that single
        # point; taking the right-hand limit keeps a nan out of the Newton
        # solve, and the inactive branch is in any case only reachable by an
        # intermediate iterate.
        active = gt(raw_gap, 0.0)
        zero = gap * 0.0
        safe_gap = where(active, gap, zero + 1.0)
        cube_root = where(active, pow(safe_gap, 1.0 / 3.0), zero)
        driving = c - ceq
        c_dot = K * cube_root * driving

        if v is None:
            return c_dot

        # Differential pushforward. Closed-form coefficients of typed Scalars;
        # no Jacobian materialised.
        #
        # Write g = <c0 - c>, so dg/dc = -1 and dg/dc0 = +1 on the active
        # branch and both vanish off it. Then with u = g^(1/3),
        #   du/dg = 1/3 g^(-2/3)
        # and
        #   d c_dot / d c   = K * (u - driving * du/dg)
        #   d c_dot / d c0  = K * driving * du/dg
        #   d c_dot / d ceq = -K * u
        #   d c_dot / d K   = u * driving
        #
        # ``du/dg`` reuses the same ``safe_gap`` / ``active`` pair as the
        # primal, so the pushforward agrees with what autograd produces from
        # the primal on both branches and at the endpoint.
        dcube = where(active, pow(safe_gap, -2.0 / 3.0) / 3.0, zero)

        actions: dict[str, ChainRuleAction] = {
            "solute_concentration": lambda V, c_=K * (cube_root - driving * dcube): c_ * V,
        }
        if "depletion_coefficient" in self._promoted_params:
            actions[self._promoted_params["depletion_coefficient"].input_name] = (
                lambda V, c_=cube_root * driving: c_ * V
            )
        if "initial_concentration" in self._promoted_params:
            actions[self._promoted_params["initial_concentration"].input_name] = (
                lambda V, c_=K * driving * dcube: c_ * V
            )
        if "equilibrium_concentration" in self._promoted_params:
            actions[self._promoted_params["equilibrium_concentration"].input_name] = (
                lambda V, c_=-K * cube_root: c_ * V
            )

        return c_dot, self.apply_chain_rule(v, self._c_rate, actions, output=c_dot)


__all__ = ["SoluteDepletion"]
