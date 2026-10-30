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

"""Norton power-law creep flow rate (MOOSE PowerLawCreepStressUpdate law)."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, abs, exp, heaviside, log
from ....types import opaque_pow as wpow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("PowerLawCreepFlowRate")
class PowerLawCreepFlowRate(Model):
    r"""Norton (Bailey) power-law creep flow rate,
    $\dot{\gamma} = A \left< f \right>^n \exp\left( -\frac{Q}{R T} \right)$,
    where $f$ is the driving force (the von Mises effective stress), $A$ is the
    creep coefficient, $n$ is the stress exponent, $Q$ is the activation energy,
    $R$ is the universal gas constant, $T$ is the absolute temperature, and
    $\left< \cdot \right>$ is the Macaulay bracket.

    This is the flow-rate form of the MOOSE ``PowerLawCreepStressUpdate`` creep
    law. The ``temperature`` input is optional: omit it for the
    temperature-independent form $\dot{\gamma} = A \left< f \right>^n$ (then
    $Q$ and $R$ are unused). The MOOSE time-hardening factor $t^m$ is not
    modeled (it corresponds to $m = 0$).
    """

    hit = HitSchema(
        input(
            "yield_function",
            Scalar,
            "Driving force (von Mises effective stress)",
            attr="_f_name",
        ),
        input(
            "temperature",
            Scalar,
            "Absolute temperature (optional; omit for temperature-independent creep)",
            default=None,
            attr="_T_name",
        ),
        output("flow_rate", Scalar, "Creep flow rate"),
        parameter(
            "coefficient",
            Scalar,
            "Power-law creep coefficient",
            attr="A",
            allow_promotion=True,
        ),
        parameter(
            "exponent",
            Scalar,
            "Power-law stress exponent",
            attr="n",
            allow_promotion=True,
        ),
        parameter(
            "activation_energy",
            Scalar,
            "Creep activation energy (used only when temperature is provided)",
            attr="Q",
            default="0.0",
            allow_promotion=True,
        ),
        parameter(
            "gas_constant",
            Scalar,
            "Universal gas constant (used only when temperature is provided)",
            attr="R",
            default="8.3143",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the parameters (A, n, Q, R); no __init__ needed.
    A: Scalar
    n: Scalar
    Q: Scalar
    R: Scalar
    _f_name: str
    _T_name: str | None

    def forward(  # type: ignore[override]
        self,
        *inputs: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Inputs arrive positionally in ``input_spec`` declaration order. The
        # optional ``temperature`` is popped from ``input_spec`` when HIT omits
        # it (default=None), so the present structural inputs are exactly the
        # leading positional args; the remainder are promoted parameters.
        names = list(self.input_spec)
        n_structural = 1 + (1 if self._T_name is not None else 0)
        structural = dict(zip(names[:n_structural], inputs[:n_structural], strict=True))
        promoted_params = inputs[n_structural:]

        f = structural[self._f_name]
        A = self._get_param("A", promoted_params, Scalar)
        n = self._get_param("n", promoted_params, Scalar)

        # gamma_dot = A * <f>^n * exp(-Q/(R T)), with <f> = f * H(f) = max(f, 0).
        Hf = heaviside(f)
        f_abs = abs(f)
        rate = A * wpow(f_abs, n)

        Q = R = T = None
        if self._T_name is not None:
            Q = self._get_param("Q", promoted_params, Scalar)
            R = self._get_param("R", promoted_params, Scalar)
            T = structural[self._T_name]
            rate = rate * exp(-Q / (R * T))

        gamma_dot = rate * Hf

        if v is None:
            return gamma_dot

        # Differential pushforwards (JVP). With gamma_dot = A <f>^n exp(-Q/(R T)):
        #   d/df = (n / f_abs) * gamma_dot
        #   d/dT = (Q / (R T^2)) * gamma_dot
        #   d/dA = gamma_dot / A
        #   d/dn = gamma_dot * log(f_abs)
        #   d/dQ = -gamma_dot / (R T)
        #   d/dR =  gamma_dot * Q / (R^2 T)
        coef_f = n / f_abs * gamma_dot
        actions = {self._f_name: lambda V, c=coef_f: c * V}

        if self._T_name is not None:
            # Q, R, T are non-None whenever temperature is wired (set with the value above).
            assert Q is not None and R is not None and T is not None
            actions[self._T_name] = lambda V, c=Q / (R * T * T) * gamma_dot: c * V
            if "Q" in self._promoted_params:
                nm = self._promoted_params["Q"].input_name
                actions[nm] = lambda V, c=-gamma_dot / (R * T): c * V
            if "R" in self._promoted_params:
                nm = self._promoted_params["R"].input_name
                actions[nm] = lambda V, c=gamma_dot * Q / (R * R * T): c * V

        if "A" in self._promoted_params:
            nm = self._promoted_params["A"].input_name
            actions[nm] = lambda V, c=gamma_dot / A: c * V
        if "n" in self._promoted_params:
            nm = self._promoted_params["n"].input_name
            n_coef = gamma_dot * log(f_abs)
            actions[nm] = lambda V, c=n_coef: c * V

        return gamma_dot, self.apply_chain_rule(v, "flow_rate", actions, output=gamma_dot)
