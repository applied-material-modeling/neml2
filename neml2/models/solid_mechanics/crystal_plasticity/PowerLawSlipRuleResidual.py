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

"""Inverted-form power-law slip-rule residual (see the class docstring)."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, lt, sign, where
from ....types import abs as tensor_abs
from ....types import opaque_pow as tensor_pow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("PowerLawSlipRuleResidual")
class PowerLawSlipRuleResidual(Model):
    r"""Power-law slip rule stated as an implicit residual in *inverted* form,
    $r_i = \hat{\tau}_i \operatorname{sgn}(\dot{\gamma}_i)
    \left| \dot{\gamma}_i / \dot{\gamma}_0 \right|^{1/n} - \tau_i$,
    rather than as the explicit map
    $\dot{\gamma}_i = \dot{\gamma}_0 \left| \tau_i / \hat{\tau}_i \right|^{n-1}
    (\tau_i / \hat{\tau}_i)$ of :class:`PowerLawSlipRule`.

    The crystal-plasticity counterpart of
    :class:`~neml2.models.solid_mechanics.plasticity.PerzynaPlasticFlowRateResidual`,
    and the same argument applies: substituting the explicit map into the
    backward-Euler equations makes the residuals degree-$n$ in the stress, and
    Newton on a degree-$n$ monomial converges only linearly, contracting by
    $(1-1/n)^n$. Carrying the slip rates as unknowns and closing them with this
    residual relocates that to a single $1/n$ power.

    **When this helps.** The gain tracks the overstress ratio
    $\tau / \hat{\tau}$ at the cold start, and it *crosses over*: on a scalar
    model of one slip system at $n = 20$, the rate form takes 19 iterations at
    $\tau/\hat{\tau} = 2$ against this form's 13, but 61 against 10 at
    $\tau/\hat{\tau} = 20$. Below roughly $\tau/\hat{\tau} \approx 2$ the
    explicit form is the better choice.

    **Regularization.** $\left|\dot{\gamma}\right|^{1/n}$ has infinite slope at
    the origin, which is exactly where a cold start sits. Unlike the Perzyna
    case there is no Macaulay bracket and the law is odd, so the extension must
    be odd too: below ``cutoff`` the power is replaced by the unique odd cubic
    matching its value and slope at the cutoff. That keeps the residual $C^1$
    and monotone with a bounded derivative through zero, so no positivity
    projection is needed (the solver has none) and either sign of slip works.

    **Cost.** The slip rates are per-slip-system, so this promotes a sub-batched
    quantity to an unknown -- for a cubic crystal with 12 systems the implicit
    system grows by 12 per grain and the layout becomes BLOCK. That is the
    trade: a much better-conditioned nonlinearity on a larger system.
    """

    hit = HitSchema(
        input("resolved_shears", Scalar, "Resolved shear on each slip system", attr="_rss"),
        input("slip_strengths", Scalar, "Slip system strengths", attr="_tau"),
        input(
            "slip_rates",
            Scalar,
            "Slip rates. Unlike PowerLawSlipRule these are *inputs* -- carried as "
            "unknowns of the implicit system and solved for.",
            attr="_g",
        ),
        derived_output("slip_rates", Scalar, attr="_resid", suffix="_residual"),
        parameter("gamma0", Scalar, "Reference slip rate"),
        parameter("n", Scalar, "Rate sensitivity exponent"),
        parameter(
            "cutoff",
            Scalar,
            "Slip-rate magnitude below which the fractional power is replaced by an odd "
            "cubic, bounding the derivative at the origin. Lower is more accurate and "
            "less well-conditioned.",
            default="1e-20",
            attr="gc",
        ),
    )

    _rss: str
    _tau: str
    _g: str
    _resid: str
    gamma0: Scalar
    n: Scalar
    gc: Scalar

    def forward(  # type: ignore[override]
        self,
        rss: Scalar,
        tau: Scalar,
        g: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        gamma0 = self._get_param("gamma0", promoted_params, Scalar)
        nv = self._get_param("n", promoted_params, Scalar)
        gc = self._get_param("gc", promoted_params, Scalar)

        a = tensor_abs(g)
        below = lt(a, gc)
        # Clamp before the fractional power: `where` evaluates both branches, and
        # a below-cutoff magnitude would otherwise drive the power's derivative
        # to infinity and poison the gradient on the branch we discard.
        a_safe = where(below, gc, a)
        # P = (|g|/gamma0)^(1/n) and its slope in |g|. On the regularized branch
        # a_safe == gc, so these double as the cubic's matching value and slope.
        P = tensor_pow(a_safe / gamma0, 1.0 / nv)
        dP = P / (nv * a_safe)

        # Odd cubic c1*g + c3*g^3 on |g| < gc, matching P and dP at g = gc. It is
        # the unique odd cubic that does so, and is monotone on the interval.
        c1 = P * (3.0 - 1.0 / nv) / (2.0 * gc)
        c3 = P * (1.0 / nv - 1.0) / (2.0 * gc * gc * gc)
        Q = where(below, c1 * g + c3 * g * g * g, sign(g) * P)
        dQ = where(below, c1 + 3.0 * c3 * g * g, dP)

        r = tau * Q - rss

        if v is None:
            return r

        # dr/d(slip_rates)     = tau * dQ
        # dr/d(resolved_shears) = -1
        # dr/d(slip_strengths)  = Q
        coef_g = tau * dQ
        actions = {
            self._g: lambda V, c=coef_g: c * V,
            self._rss: lambda V: -V,
            self._tau: lambda V, c=Q: c * V,
        }
        return r, self.apply_chain_rule(v, self._resid, actions, output=r)


__all__ = ["PowerLawSlipRuleResidual"]
