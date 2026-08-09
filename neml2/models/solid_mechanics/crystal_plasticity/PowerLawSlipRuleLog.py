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

"""Log-space form of the power-law slip rule: a matched residual/recovery pair.

The slip rates in a crystal-plasticity solve span an enormous dynamic range --
70 orders of magnitude between a well-oriented and a nearly-inactive system is
routine. Carrying ``log|gdot|`` instead of ``gdot`` compresses that onto a few
hundred units and turns the power law affine. See the two class docstrings.
"""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, option, output, parameter
from ....types import Scalar, clamp, exp, gt, heaviside, log, sign, where
from ....types import abs as tensor_abs
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("PowerLawSlipRuleLogResidual")
class PowerLawSlipRuleLogResidual(Model):
    r"""Power-law slip rule as an implicit residual in *log* space,
    $r_i = u_i - n \log \left( \left| \tau_i \right| / \hat{\tau}_i \right)$,
    where $u_i = \log \left( \left| \dot{\gamma}_i \right| / \dot{\gamma}_0 \right)$
    is carried as an unknown.

    Taking the log of the flow law's magnitude relation,
    $\left| \dot{\gamma} \right| / \dot{\gamma}_0 = \left| \tau / \hat{\tau} \right|^n$,
    turns a degree-$n$ monomial into something **exactly affine in the unknown**:
    $\partial r_i / \partial u_i = 1$, identically. What nonlinearity remains
    enters only through $\log \left| \tau \right|$, which is far gentler than
    either the $n$ power of the explicit form or the $1/n$ power of
    :class:`PowerLawSlipRuleResidual`.

    Two things this buys over the $1/n$-power form:

    * **No singular derivative at zero slip.** $\left| \dot{\gamma} \right|^{1/n}$
      has infinite slope at the origin and needs a regularization cutoff, which
      is awkward precisely because many slip systems legitimately sit near zero.
      In log space a dormant system is simply a large negative $u$ -- no cutoff,
      no special branch.
    * **The dynamic range collapses.** Slip rates spanning $10^{-63}$ to $10^{9}$
      become $u \in [-145, 21]$.

    The price is that $\log \left| \tau \right|$ diverges as $\tau \to 0$, so the
    resolved shear is floored at ``tau_floor`` times the slip strength. A system
    at the floor solves to $u = n \log(\texttt{tau\_floor})$, i.e. a slip rate of
    $\dot{\gamma}_0 \texttt{tau\_floor}^n$ -- utterly negligible at the default.

    **Line search is required.** The residual is nearly flat in $u$ far from the
    root and has a barrier where the slip absorbs the whole trial shear; an
    undamped Newton step vaults over it and cannot recover. With backtracking it
    is the best-conditioned of the three forms. Pair with
    :class:`SlipRateFromLog`, which recovers the signed slip rate.
    """

    hit = HitSchema(
        input("resolved_shears", Scalar, "Resolved shear on each slip system", attr="_rss"),
        input("slip_strengths", Scalar, "Slip system strengths", attr="_tau"),
        input(
            "log_slip_rates",
            Scalar,
            "log(|slip rate| / gamma0) on each system, carried as an unknown",
            attr="_u",
        ),
        derived_output("log_slip_rates", Scalar, attr="_resid", suffix="_residual"),
        parameter("n", Scalar, "Rate sensitivity exponent"),
        parameter(
            "tau_floor",
            Scalar,
            "Lower bound on |resolved shear| / slip strength, keeping log|tau| finite. A "
            "system at the floor solves to a slip rate of gamma0 * tau_floor^n.",
            default="1e-8",
            attr="tf",
        ),
    )

    _rss: str
    _tau: str
    _u: str
    _resid: str
    n: Scalar
    tf: Scalar

    def forward(  # type: ignore[override]
        self,
        rss: Scalar,
        tau: Scalar,
        u: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        nv = self._get_param("n", promoted_params, Scalar)
        tf = self._get_param("tf", promoted_params, Scalar)

        floor = tf * tau
        a = tensor_abs(rss)
        # Floor before the log. `where` evaluates both branches, so the guarded
        # value -- not the raw shear -- must go into the log; log of a zero shear
        # would poison the gradient even on the branch that discards it.
        above = gt(a, floor)
        a_safe = where(above, a, floor)
        r = u - nv * log(a_safe / tau)

        if v is None:
            return r

        # dr/du  = 1 exactly -- this is the whole point of the transform.
        # dr/drss = -n / rss, and 0 once floored (a_safe no longer tracks rss).
        # dr/dtau = +n / tau  (the floor scales with tau, so both terms agree).
        active = heaviside(a - floor)  # 1 above the floor, 0 at it
        coef_rss = -nv * active * sign(rss) / a_safe
        coef_tau = nv / tau
        actions = {
            self._u: lambda V: V,
            self._rss: lambda V, c=coef_rss: c * V,
            self._tau: lambda V, c=coef_tau: c * V,
        }
        return r, self.apply_chain_rule(v, self._resid, actions, output=r)


@register_neml2_object("SlipRateFromLog")
class SlipRateFromLog(Model):
    r"""Recover the signed slip rate from its log magnitude,
    $\dot{\gamma}_i = \dot{\gamma}_0 \operatorname{sgn}(\tau_i) e^{u_i}$.

    The companion to :class:`PowerLawSlipRuleLogResidual`: that residual
    constrains only the magnitude, since the log discards the sign. The sign is
    not a free quantity -- the flow law forces slip to follow its driving shear
    -- so it is taken from the resolved shear here rather than carried as extra
    state.

    ``u`` is clamped before exponentiating: the unknown is legitimately large and
    negative for a dormant slip system, and ``exp`` of anything below about
    $-745$ underflows to zero anyway (which is the physically right answer).
    """

    hit = HitSchema(
        input("log_slip_rates", Scalar, "log(|slip rate| / gamma0) on each system", attr="_u"),
        input("resolved_shears", Scalar, "Resolved shear, used for the sign", attr="_rss"),
        output("slip_rates", Scalar, "Signed slip rate on each system", attr="_g"),
        parameter("gamma0", Scalar, "Reference slip rate"),
        option(
            "u_max",
            float,
            "Clamp on |u| before exponentiating, guarding overflow/underflow.",
            default=700.0,
            attr="umax",
        ),
    )

    _u: str
    _rss: str
    _g: str
    gamma0: Scalar
    umax: float

    def forward(  # type: ignore[override]
        self,
        u: Scalar,
        rss: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        gamma0 = self._get_param("gamma0", promoted_params, Scalar)
        umax = self.umax

        uc = clamp(u, -umax, umax)
        g = gamma0 * sign(rss) * exp(uc)

        if v is None:
            return g

        # dgdot/du = gdot inside the clamp, 0 outside. sgn(rss) is piecewise
        # constant, so its derivative vanishes almost everywhere.
        inside = heaviside(-(tensor_abs(u) - umax))
        coef_u = g * inside
        actions = {
            self._u: lambda V, c=coef_u: c * V,
            self._rss: lambda V: V * 0.0,
        }
        return g, self.apply_chain_rule(v, self._g, actions, output=g)


__all__ = ["PowerLawSlipRuleLogResidual", "SlipRateFromLog"]
