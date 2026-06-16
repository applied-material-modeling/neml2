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

"""Python-native mirror of the C++ ``PowerLawIsotropicHardeningStaticRecovery`` model."""

from __future__ import annotations

import torch

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, abs, log, pow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("PowerLawIsotropicHardeningStaticRecovery")
class PowerLawIsotropicHardeningStaticRecovery(Model):
    r"""This particular model implements a power law recovery of the type
    $\dot{k} = -\left(\frac{\lVert k \rVert}{\tau}\right)^{n-1} \frac{k}{\tau}$
    """

    # Rate output is derived from the ``isotropic_hardening`` input name with
    # the ``_rate`` suffix — when HIT renames the input (e.g.
    # ``isotropic_hardening = 'k_recv'``) the rate output follows
    # automatically (``k_recv_rate``). Mirrors C++
    # ``IsotropicHardeningStaticRecovery::declare_output_variable<Scalar>(rate_name(_h.name()))``.
    hit = HitSchema(
        input("isotropic_hardening", Scalar, "Isotropic hardening variable"),
        derived_output("isotropic_hardening", Scalar, attr="_h_rate", suffix="_rate"),
        parameter("tau", Scalar, "Recovery rate", attr="tau", allow_nonlinear=True),
        parameter("n", Scalar, "Recovery exponent", attr="n", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the ``tau`` / ``n`` parameters (stored as
    # ``tau`` / ``n``) -- no __init__ needed.
    tau: Scalar
    n: Scalar
    _h_rate: str

    def forward(  # type: ignore[override]
        self,
        isotropic_hardening: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``PowerLawIsotropicHardeningStaticRecovery::set_value`` in the C++ source.
        h = isotropic_hardening
        tau = self._get_param("tau", nl_params, Scalar)
        n = self._get_param("n", nl_params, Scalar)

        # h_dot = -(|h| / tau)^(n - 1) * h / tau. ``abs`` / ``pow`` on a Scalar
        # return Scalar at runtime; the explicit casts let pyright see the
        # narrow Scalar type the generic ``TensorWrapper`` signatures discard.
        abs_h = abs(h)
        u = abs_h / tau  # |h| / tau
        nm1 = n - 1.0
        h_dot = -pow(u, nm1) * h / tau

        if v is None:
            return h_dot

        # Differential pushforward. Each action takes a typed Scalar
        # tangent of the input's type and returns the Scalar tangent of h_dot.
        # No Jacobian is materialised; each coefficient is a Scalar that
        # broadcasts against the leading-K tangent axis automatically.
        #
        # Derivations match the dense C++ Jacobian in set_value:
        #   d h_dot / d h   = -n * (|h / tau|)^(n - 1) / |tau|
        #   d h_dot / d tau = n * h * tau^(-1 - n) * |h|^(n - 1)
        #   d h_dot / d n   = -h * tau^(-n) * |h|^(n - 1) * log(|h| / tau + eps)
        # where ``eps`` is the dtype's machine epsilon, matching
        # ``machine_precision(_h.scalar_type())`` in the C++ source.
        coef_h = -n * pow(abs(h / tau), nm1) / abs(tau)

        h_name = next(iter(self.input_spec))
        actions = {h_name: lambda V, c=coef_h: c * V}
        if "tau" in self._nl_params:
            coef_tau = n * h * pow(tau, -1.0 - n) * pow(abs_h, nm1)
            actions[self._nl_params["tau"].input_name] = lambda V, c=coef_tau: c * V
        if "n" in self._nl_params:
            eps = torch.finfo(h.dtype).eps
            coef_n = -h * pow(tau, -n) * pow(abs_h, nm1) * log(abs_h / tau + eps)
            actions[self._nl_params["n"].input_name] = lambda V, c=coef_n: c * V

        return h_dot, self.apply_chain_rule(v, self._h_rate, actions, output=h_dot)
