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

"""Simplified precipitate coarsening, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SimplePrecipitateGrowth")
class SimplePrecipitateGrowth(Model):
    r"""Simplified, volume-conserving coarsening of a precipitate population,
    defined by
    $\dot{R} = \dfrac{K}{3 R^2}, \qquad \dot{N} = -\dfrac{K}{R^3}$
    where $R$ is the mean precipitate radius, $N$ is the precipitate
    number density and $K$ is the coarsening coefficient.

    $K$ is a promotable parameter, so it may be supplied as a constant or
    driven by another model -- an ``ArrheniusParameter`` for the usual
    thermally activated form.

    Note $\dot{N}$ does not depend on $N$, following the issue
    specification; the two rates are emitted by one leaf because they share
    the single coefficient $K$ and the same $R$.
    """

    hit = HitSchema(
        input("precipitate_radius", Scalar, "Mean precipitate radius"),
        input("precipitate_density", Scalar, "Precipitate number density"),
        derived_output("precipitate_radius", Scalar, attr="_R_rate", suffix="_rate"),
        derived_output("precipitate_density", Scalar, attr="_N_rate", suffix="_rate"),
        parameter("coarsening_coefficient", Scalar, "Coarsening coefficient", allow_promotion=True),
    )

    # ``from_hit`` auto-declares the parameter and the two derived output names.
    coarsening_coefficient: Scalar
    _R_rate: str
    _N_rate: str

    def forward(  # type: ignore[override]
        self,
        R: Scalar,
        N: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        K = self._get_param("coarsening_coefficient", promoted_params, Scalar)

        R2 = R * R
        R3 = R2 * R
        R_dot = K / (3.0 * R2)
        N_dot = -K / R3

        if v is None:
            return R_dot, N_dot

        # Differential pushforward. Closed-form coefficients of typed Scalars;
        # no Jacobian materialised.
        #   d R_dot / d R = -2 K / (3 R^3) = -2 R_dot / R
        #   d R_dot / d K =  1 / (3 R^2)
        #   d N_dot / d R =  3 K / R^4     = -3 N_dot / R
        #   d N_dot / d K = -1 / R^3
        # Both rates are linear in K, so the K coefficients are written as the
        # bare R terms rather than as ``<rate> / K``: zeroing K is a valid way
        # to switch coarsening off, and the quotient form is 0/0 there.
        # N appears in neither rate, so its action is a structural zero and no
        # entry is registered for it.
        actions_R: dict[str, ChainRuleAction] = {
            "precipitate_radius": lambda V, c=-2.0 * R_dot / R: c * V,
        }
        actions_N: dict[str, ChainRuleAction] = {
            "precipitate_radius": lambda V, c=-3.0 * N_dot / R: c * V,
        }
        if "coarsening_coefficient" in self._promoted_params:
            ext = self._promoted_params["coarsening_coefficient"].input_name
            actions_R[ext] = lambda V, c=1.0 / (3.0 * R2): c * V
            actions_N[ext] = lambda V, c=-1.0 / R3: c * V

        # Both outputs are derived names with no ``_var_renames`` entry, so pass
        # the resolved external names directly (same convention as ZenerElement).
        v_R = self.apply_chain_rule(v, self._R_rate, actions_R, output=R_dot)
        v_N = self.apply_chain_rule(v, self._N_rate, actions_N, output=N_dot)
        return R_dot, N_dot, {**v_R, **v_N}


__all__ = ["SimplePrecipitateGrowth"]
