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

"""Python-native mirror of the C++ ``PowerLawKinematicHardeningStaticRecovery`` model."""

from __future__ import annotations

import torch

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, derived_output, input, parameter
from ....types import SR2, Scalar, inner, log, norm, pow


@register_neml2_object("PowerLawKinematicHardeningStaticRecovery")
class PowerLawKinematicHardeningStaticRecovery(Model):
    r"""This object defines kinematic hardening static recovery on a backstress term.
    This particular model uses a power law for recovery
    $\dot{X} = - \left(\frac{\lVert X \rVert}{\tau}\right)^{n-1} \frac{X}{\tau}$
    where $n$ is the power law recovery exponent and $\tau$ is the recovery rate.
    """

    # Rate output is derived from the ``back_stress`` input name with the
    # ``_rate`` suffix — when HIT renames the input (e.g.
    # ``back_stress = 'X1_recv'``) the rate output follows automatically
    # (``X1_recv_rate``). Mirrors C++
    # ``KinematicHardeningStaticRecovery::declare_output_variable<SR2>(rate_name(_X.name()))``.
    hit = HitSchema(
        input("back_stress", SR2, "Back stress"),
        derived_output("back_stress", SR2, attr="_X_rate", suffix="_rate"),
        parameter("tau", Scalar, "Static recovery rate", attr="tau", allow_nonlinear=True),
        parameter("n", Scalar, "Static recovery exponent", attr="n", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the ``tau`` / ``n`` parameters (stored as
    # ``tau`` / ``n``) -- no __init__ needed.
    tau: Scalar
    n: Scalar
    _X_rate: str

    def forward(  # type: ignore[override]
        self,
        back_stress: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Mirrors ``PowerLawKinematicHardeningStaticRecovery::set_value`` in the C++ source.
        X = back_stress
        tau = self._get_param("tau", nl_params, Scalar)
        n = self._get_param("n", nl_params, Scalar)

        # Match the C++ ``machine_precision(_X.scalar_type())`` eps regularizer
        # for ``s = norm(X)`` so the result stays differentiable at X == 0.
        eps = torch.finfo(X.dtype).eps
        s = norm(X, eps)  # Scalar
        # X_dot = -(s/tau)^(n-1) * X / tau
        nm1 = n - 1.0
        s_over_tau = s / tau
        X_dot = -pow(s_over_tau, nm1) * X / tau

        if v is None:
            return X_dot

        # Differential pushforward. The C++ Jacobian forms an SSR4
        #   d X_dot / d X = -s^(n-3) * ((n-1) * outer(X) + s^2 * I) / tau^n
        # but we never materialise it; the directional derivative is the
        # closed-form contraction
        #   action_X(V) = -s^(n-3) * ((n-1) * inner(X, V) * X + s^2 * V) / tau^n
        # using outer(X) : V == X * inner(X, V) (typed inner returns Scalar).
        coef_pre = -pow(s, n - 3.0) / pow(tau, n)

        def x_action(V: SR2) -> SR2:
            return coef_pre * (nm1 * inner(X, V) * X + s * s * V)

        X_name = next(iter(self.input_spec))
        actions: dict[str, ChainRuleAction] = {X_name: x_action}

        if "tau" in self._nl_params:
            # d X_dot / d tau = n * (s/tau)^(n-1) * X / tau^2
            coef_tau = n * pow(s_over_tau, nm1) * X / (tau * tau)
            actions[self._nl_params["tau"].input_name] = lambda V, c=coef_tau: c * V

        if "n" in self._nl_params:
            # d X_dot / d n = -X / s * (s/tau)^n * log(s/tau)
            coef_n = -X / s * pow(s_over_tau, n) * log(s_over_tau)
            actions[self._nl_params["n"].input_name] = lambda V, c=coef_n: c * V

        return X_dot, self.apply_chain_rule(v, self._X_rate, actions, output=X_dot)
