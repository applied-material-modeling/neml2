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

"""Inverted-form Perzyna flow-rate residual (see the class docstring)."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, heaviside, lt, macaulay, where
from ....types import opaque_pow as wpow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("PerzynaPlasticFlowRateResidual")
class PerzynaPlasticFlowRateResidual(Model):
    r"""Perzyna flow rule stated as an implicit residual in *inverted* form,
    $r = \eta \dot{\gamma}^{1/n} - \left< f \right>$, rather than as the explicit
    map $\dot{\gamma} = \left( \left< f \right> / \eta \right)^n$ of
    :class:`PerzynaPlasticFlowRate`.

    Both have the same root. The difference is what Newton sees. Substituting
    the explicit map into the backward-Euler equations makes every residual a
    degree-$n$ polynomial in the stress, and Newton on a degree-$n$ monomial
    converges only *linearly*, contracting by $(1-1/n)^n$ per iteration until
    the iterate reaches the root -- 13-15 wasted iterations at a cold start for
    typical rate sensitivities. Carrying $\dot{\gamma}$ as an unknown of the
    implicit system and closing it with this residual instead relocates that
    nonlinearity to a single $1/n$ power, leaving the stress and
    internal-variable residuals affine in $\dot{\gamma}$.

    The gain grows with the stiffness of the step, which is where the rate form
    hurts most: for $n = 8$ the scalar model problem takes 12 iterations either
    way at $\Delta t = 1$, but 20 (rate form) against 8 (this form) at
    $\Delta t = 10^4$.

    **Regularization.** $\dot{\gamma}^{1/n}$ has infinite slope at
    $\dot{\gamma} = 0$, which is exactly where a cold start sits and exactly the
    root on an elastic step. Below ``cutoff`` the power is therefore replaced by
    its tangent line at the cutoff, giving a $C^1$ residual with a bounded
    derivative everywhere and no need for a positivity projection (which the
    Newton solver does not have). On an elastic step this converges in one
    iteration to $\dot{\gamma} = -(n-1)\,\mathrm{cutoff}$ instead of exactly
    zero; with the default cutoff that error is far below solver tolerance.

    Pair this with an initial guess for ``flow_rate`` -- the residual is well
    behaved at zero thanks to the regularization, but a strictly positive seed
    (e.g. an initial condition plus a ``ConstantExtrapolationPredictor``) costs
    nothing and saves a couple of iterations.
    """

    hit = HitSchema(
        input("yield_function", Scalar, "Yield function"),
        input(
            "flow_rate",
            Scalar,
            "Flow rate. Unlike PerzynaPlasticFlowRate this is an *input* -- it "
            "is carried as an unknown of the implicit system and solved for.",
        ),
        output("flow_rate_residual", Scalar, "Residual of the inverted flow rule"),
        parameter("reference_stress", Scalar, "Reference stress", attr="eta"),
        parameter("exponent", Scalar, "Power-law exponent", attr="n"),
        parameter(
            "cutoff",
            Scalar,
            "Flow rate below which the fractional power is replaced by its tangent line, "
            "bounding the derivative at the origin. Lower is more accurate and less "
            "well-conditioned.",
            default="1e-20",
            attr="gc",
        ),
    )

    eta: Scalar
    n: Scalar
    gc: Scalar

    def forward(  # type: ignore[override]
        self,
        yield_function: Scalar,
        flow_rate: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        f = yield_function
        g = flow_rate
        eta = self._get_param("eta", promoted_params, Scalar)
        n = self._get_param("n", promoted_params, Scalar)
        gc = self._get_param("gc", promoted_params, Scalar)

        below = lt(g, gc)
        # Evaluate the power at gc wherever we are below it: `where` evaluates
        # *both* branches, and a fractional power of a negative flow rate is NaN
        # -- which would poison the gradient even on the branch we discard.
        g_safe = where(below, gc, g)
        P = wpow(g_safe, 1.0 / n)
        # Slope of the power at g_safe. On the regularized branch g_safe == gc,
        # so this is already the tangent-line slope -- meaning dP/dg is this
        # expression on *both* branches, with no `where` needed below.
        dP = P / (n * g_safe)
        # Tangent line below the cutoff, the power itself above it.
        P_reg = where(below, P + dP * (g - gc), P)

        r = eta * P_reg - macaulay(f)

        if v is None:
            return r

        # dr/d(flow_rate)      = eta * dP   (both branches, see above)
        # dr/d(yield_function) = -H(f)
        coef_g = eta * dP
        coef_f = -heaviside(f)
        actions = {
            "flow_rate": lambda V, c=coef_g: c * V,
            "yield_function": lambda V, c=coef_f: c * V,
        }
        return r, self.apply_chain_rule(v, "flow_rate_residual", actions, output=r)


__all__ = ["PerzynaPlasticFlowRateResidual"]
