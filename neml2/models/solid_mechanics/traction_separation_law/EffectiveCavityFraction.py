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

"""GTN-style coalescence map for the Sham-Needleman cavitation model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("EffectiveCavityFraction")
class EffectiveCavityFraction(Model):
    r"""GTN-style coalescence map from the physical cavitated area fraction $f$
    to the effective fraction $f^*$ that drives the softening functions,

    $$
    f^* = \begin{cases}
      f, & f \le f_c \\
      f_c + \kappa (f - f_c), & f_c < f < f_F \\
      f_u^*, & f \ge f_F
    \end{cases}, \qquad \kappa = \frac{f_u^* - f_c}{f_F - f_c}.
    $$

    $f_c$ is the coalescence onset (ligament rupture at $a/b \approx 0.7$ puts
    it near $0.5$), $f_F$ the physical fraction at final separation, and
    $f_u^*$ the effective fraction at which the interface carries no load.
    Because both softening functions, $q(f)$ and $(1-f)$, vanish at $f = 1$,
    $f_u^*$ must be set strictly below one so $q(f^*)$ stays positive and the
    rates stay finite right up to separation.

    The map **saturates** at $f_u^*$ for $f \ge f_F$ rather than extrapolating
    the accelerated branch. Without that cap a separated state, or a Newton
    iterate that overshoots, drives $f^*$ through one at
    $f = f_c + (1 - f_c) / \kappa$ -- a physical fraction below one and
    therefore reachable -- and both downstream denominators, $q(f^*)$ and
    $1 - f^*$, vanish together. Saturating bounds every consumer of $f^*$ by
    $f_u^* < 1$ for any $f$ whatsoever, so the rates are finite however far an
    iterate strays. The map is continuous at $f_F$: the accelerated branch
    reaches exactly $f_u^*$ there.

    Only the softening functions see $f^*$. The geometric factors ($f/N$,
    $\sqrt{f}$, $V(f,N)$) keep the physical $f$, which is what preserves the
    identity $[\![u]\!]_n^v = NV - N_0V_0$.
    """

    hit = HitSchema(
        input("cavity_fraction", Scalar, "Physical cavitated area fraction"),
        output("effective_cavity_fraction", Scalar, "Effective cavitated area fraction"),
        parameter(
            "coalescence_fraction",
            Scalar,
            "Area fraction at the onset of coalescence",
            attr="f_c",
            allow_promotion=True,
        ),
        parameter(
            "failure_fraction",
            Scalar,
            "Physical area fraction at final separation",
            attr="f_F",
            allow_promotion=True,
        ),
        parameter(
            "ultimate_fraction",
            Scalar,
            "Effective area fraction at which the interface carries no load. Keep "
            "strictly below one so the softening function q stays positive.",
            attr="f_u",
            allow_promotion=True,
        ),
    )

    f_c: Scalar
    f_F: Scalar
    f_u: Scalar

    def forward(  # type: ignore[override]
        self,
        f: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        f_c = self._get_param("f_c", promoted_params, Scalar)
        f_F = self._get_param("f_F", promoted_params, Scalar)
        f_u = self._get_param("f_u", promoted_params, Scalar)

        span = f_F - f_c
        kappa = (f_u - f_c) / span
        # Fraction of the way from coalescence onset to failure. Writing the
        # accelerated branch as f_c + (f_u - f_c) * t rather than
        # f_c + kappa * (f - f_c) is the same number, but it makes the
        # parameter partials below read straight off the expression.
        t = (f - f_c) / span
        # Detached branch masks, as elsewhere in this module: both the
        # coalescence switch and the failure cap are kinks in f*, not
        # differentiable functions of f, so their variation must not leak
        # into the pushforward.
        coalesced = gt(f, f_c)
        failed = gt(f, f_F)
        f_star = where(failed, f_u, where(coalesced, f_c + (f_u - f_c) * t, f))

        if v is None:
            return f_star

        zero = Scalar.from_value(0.0, like=f)
        one = Scalar.from_value(1.0, like=f)
        df_df = where(failed, zero, where(coalesced, kappa, one))
        actions: dict[str, ChainRuleAction] = {
            "cavity_fraction": lambda V, c=df_df: c * V,
        }
        # On the intact branch f* == f carries no parameter dependence at all;
        # on the saturated branch it is f_u exactly, so only that one partial
        # survives. Both are masked here rather than guarded, so a promoted
        # parameter contributes the right thing on every branch.
        if "f_c" in self._promoted_params:
            dfs_dfc = one - t + (f_u - f_c) * (f - f_F) / (span * span)
            c_fc = where(failed, zero, where(coalesced, dfs_dfc, zero))
            actions[self._promoted_params["f_c"].input_name] = lambda V, c=c_fc: c * V
        if "f_F" in self._promoted_params:
            dfs_dfF = -(f_u - f_c) * (f - f_c) / (span * span)
            c_fF = where(failed, zero, where(coalesced, dfs_dfF, zero))
            actions[self._promoted_params["f_F"].input_name] = lambda V, c=c_fF: c * V
        if "f_u" in self._promoted_params:
            c_fu = where(failed, one, where(coalesced, t, zero))
            actions[self._promoted_params["f_u"].input_name] = lambda V, c=c_fu: c * V
        return f_star, self.apply_chain_rule(v, "effective_cavity_fraction", actions, output=f_star)


__all__ = ["EffectiveCavityFraction"]
