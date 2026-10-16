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

"""Solid solution strengthening, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, pow, sqrt
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SolidSolutionStrength")
class SolidSolutionStrength(Model):
    r"""Solid solution strengthening contribution, defined by
    $\tau = \alpha \mu \varepsilon^{3/2} c^{1/2}$
    where $c$ is the matrix solute concentration, $\varepsilon$ is the
    misfit strain, $\alpha$ is an interaction coefficient and $\mu$ is
    the shear modulus.

    The output is a single strength contribution, intended to be combined with
    the other contributions through a ``ScalarLinearCombination``.
    """

    hit = HitSchema(
        input("solute_concentration", Scalar, "Current matrix solute concentration"),
        output("solid_solution_strength", Scalar, "The solid solution strength contribution"),
        parameter("alpha", Scalar, "Interaction coefficient", allow_promotion=True),
        parameter("shear_modulus", Scalar, "Shear modulus", allow_promotion=True),
        parameter("misfit_strain", Scalar, "Misfit strain", allow_promotion=True),
    )

    # ``from_hit`` auto-declares the three parameters via the schema.
    alpha: Scalar
    shear_modulus: Scalar
    misfit_strain: Scalar

    def forward(  # type: ignore[override]
        self,
        c: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        alpha = self._get_param("alpha", promoted_params, Scalar)
        mu = self._get_param("shear_modulus", promoted_params, Scalar)
        eps = self._get_param("misfit_strain", promoted_params, Scalar)

        sqrt_c = sqrt(c)
        # ``pow(eps, 1.5)`` rather than the algebraically equal ``eps *
        # sqrt(eps)``: the factored form's backward is ``inf * 0`` at
        # ``eps = 0``, so reverse mode (``param_jacobian`` / ``param_vjp``,
        # which differentiate the primal) returns nan for a misfit strain that
        # has been zeroed to switch this contribution off. ``torch.pow``'s
        # backward is ``1.5 * eps^0.5``, which is a clean 0 there.
        sqrt_eps = pow(eps, 0.5)
        eps_32 = pow(eps, 1.5)
        tau = alpha * mu * eps_32 * sqrt_c

        if v is None:
            return tau

        # Differential pushforward. Closed-form coefficients of typed Scalars.
        #   d tau / d c     = alpha mu eps^(3/2) / (2 sqrt(c))
        #   d tau / d alpha = mu eps^(3/2) sqrt(c)
        #   d tau / d mu    = alpha eps^(3/2) sqrt(c)
        #   d tau / d eps   = 3/2 alpha mu eps^(1/2) sqrt(c)
        # The three parameter coefficients differentiate the product directly
        # rather than forming ``tau / <param>``: zeroing a coefficient is a
        # valid way to switch this contribution off, and the quotient form is
        # 0/0 there.
        actions: dict[str, ChainRuleAction] = {
            "solute_concentration": lambda V, k=0.5 * alpha * mu * eps_32 / sqrt_c: k * V,
        }
        if "alpha" in self._promoted_params:
            actions[self._promoted_params["alpha"].input_name] = lambda V, k=mu * eps_32 * sqrt_c: (
                k * V
            )
        if "shear_modulus" in self._promoted_params:
            actions[self._promoted_params["shear_modulus"].input_name] = (
                lambda V, k=alpha * eps_32 * sqrt_c: k * V
            )
        if "misfit_strain" in self._promoted_params:
            actions[self._promoted_params["misfit_strain"].input_name] = (
                lambda V, k=1.5 * alpha * mu * sqrt_eps * sqrt_c: k * V
            )

        return tau, self.apply_chain_rule(v, "solid_solution_strength", actions, output=tau)


__all__ = ["SolidSolutionStrength"]
