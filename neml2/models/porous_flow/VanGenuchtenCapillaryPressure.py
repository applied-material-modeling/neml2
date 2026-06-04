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

"""Python-native mirror of the C++ ``VanGenuchtenCapillaryPressure`` model."""

from __future__ import annotations

import math

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar, log10, lt, pow, where


@register_native("VanGenuchtenCapillaryPressure")
class VanGenuchtenCapillaryPressure(Model):
    r"""Define the van Genuchten correlation for capillary pressure
    $P_c = a \left( S_e^{-\frac{1}{m}} - 1 \right)^{1-m}$. Here
    $S_e$ is the effective saturation, $a$ and $m$ are
    shape parameters. Optionally, a logarithmic extension is applied below a
    user-supplied transition saturation $S_p$ to keep the pressure
    finite as $S_e \to 0$.
    """

    hit = HitSchema(
        input("effective_saturation", Scalar, "The effective saturation"),
        output("capillary_pressure", Scalar, "Capillary pressure."),
        parameter(
            "a",
            Scalar,
            "Shape parameter a",
            attr="a",
            allow_nonlinear=True,
        ),
        parameter(
            "m",
            Scalar,
            "Shape parameter m",
            attr="m",
            allow_nonlinear=True,
        ),
        option(
            "log_extension",
            bool,
            "Whether to apply logarithmic extension",
            default=False,
            attr="_log_extension",
        ),
        option(
            "transition_saturation",
            float,
            "The transition value of the effective saturation below which to "
            "apply the logarithmic extension",
            default=0.0,
            attr="_Sp",
        ),
    )

    # ``from_hit`` auto-declares the ``a`` and ``m`` Scalar parameters; the
    # bool / float options land on ``self`` via ``attr=``. Annotate so pyright
    # sees the typed wrappers that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    a: Scalar
    m: Scalar
    _log_extension: bool
    _Sp: float

    def __post_init__(self) -> None:
        # Mirror the C++ constructor's mutual-consistency checks. The HIT
        # parser doesn't distinguish "default" from "user-supplied" on
        # primitive options the way the C++ side does, so we only enforce the
        # range + the log-extension-requires-Sp pairing.
        if self._log_extension and self._Sp <= 0.0:
            raise ValueError(
                "VanGenuchtenCapillaryPressure: log_extension is set to true, but "
                "transition_saturation is not specified (or is zero)."
            )
        if self._Sp < 0.0 or self._Sp > 1.0:
            raise ValueError(
                "VanGenuchtenCapillaryPressure: transition_saturation must be in the range [0, 1]."
            )

    def _calculate_pressure(self, S: Scalar, a: Scalar, m: Scalar) -> tuple[Scalar, Scalar]:
        """Forward + derivative on the (clamped) base branch.

        Mirrors ``VanGenuchtenCapillaryPressure::calculate_pressure``:
        clamp $S$ away from ``1`` by machine epsilon so the
        $S^(-1/m) - 1$ term stays positive when raised to ``1 - m``, then
        return $f = a (S^(-1/m) - 1)^(1-m)$ and its derivative.
        """
        eps = torch.finfo(S.dtype).eps
        one_minus_eps = Scalar.from_value(1.0 - eps, like=S)
        too_close = lt(one_minus_eps, S)  # S > 1 - eps  ⇔  1 - eps < S
        Sc = where(too_close, one_minus_eps, S)

        neg_inv_m = -1.0 / m
        one_minus_m = 1.0 - m
        base = pow(Sc, neg_inv_m) - 1.0  # S^(-1/m) - 1
        f = a * pow(base, one_minus_m)
        # df/dS = -a/m * (1 - m) * (S^(-1/m) - 1)^(-m) * S^(-1/m - 1)
        dfds = -a / m * one_minus_m * pow(base, -m) * pow(Sc, neg_inv_m - 1.0)
        return f, dfds

    def forward(  # type: ignore[override]
        self,
        effective_saturation: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        S = effective_saturation
        a = self._get_param("a", nl_params, Scalar)
        m = self._get_param("m", nl_params, Scalar)

        Pc_base, dPc_dS_base = self._calculate_pressure(S, a, m)

        if self._log_extension:
            # Evaluate the van Genuchten branch at the transition saturation
            # ``Sp`` to anchor the log-linear extension below ``Sp``. Mirrors
            # ``CapillaryPressure::set_value``.
            ln10 = math.log(10.0)
            Sp_scalar = Scalar.from_value(self._Sp, like=S)
            Pcs, dPcs_dS = self._calculate_pressure(Sp_scalar, a, m)
            slope = dPcs_dS / (ln10 * Pcs)
            yintercept = log10(Pcs) - slope * self._Sp
            Pc_ext = pow(Scalar.from_value(10.0, like=S), slope * S + yintercept)
            dPc_ext_dS = (ln10 * slope) * Pc_ext

            mask = lt(S, self._Sp)
            Pc = where(mask, Pc_ext, Pc_base)
            dPc_dS = where(mask, dPc_ext_dS, dPc_dS_base)
        else:
            Pc = Pc_base
            dPc_dS = dPc_dS_base

        if v is None:
            return Pc

        # Differential pushforward: linear coefficient ``dPc/dS``
        # times the incoming Scalar tangent. No Jacobian formed.
        return Pc, self.apply_chain_rule(
            v,
            "capillary_pressure",
            {"effective_saturation": lambda V, c=dPc_dS: c * V},
            output=Pc,
        )


__all__ = ["VanGenuchtenCapillaryPressure"]
