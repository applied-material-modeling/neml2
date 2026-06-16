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

"""Python-native mirror of the C++ ``BrooksCoreyCapillaryPressure`` model."""

from __future__ import annotations

import math

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar, log10, lt, pow, where
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("BrooksCoreyCapillaryPressure")
class BrooksCoreyCapillaryPressure(Model):
    r"""Define the Brooks-Corey correlation for capillary pressure
    $P_c = P_t S_e^{-\frac{1}{p}}$. Here $S_e$ is the effective
    saturation, $P_t$ is the threshold pressure at zero saturation, and
    $p$ is the shape parameter. Optionally, a logarithmic extension is
    applied below a user-supplied transition saturation $S_p$ to keep the
    pressure finite as $S_e \to 0$.
    """

    hit = HitSchema(
        input("effective_saturation", Scalar, "The effective saturation"),
        output("capillary_pressure", Scalar, "Capillary pressure."),
        parameter(
            "threshold_pressure",
            Scalar,
            "The threshold entry pressure",
            attr="Pt",
            allow_nonlinear=True,
        ),
        parameter(
            "exponent",
            Scalar,
            "The shape parameter p",
            attr="p",
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

    # ``from_hit`` auto-declares the ``threshold_pressure`` parameter (stored
    # as ``Pt``) and the ``exponent`` parameter (stored as ``p``); the bool /
    # float options land on ``self`` via ``attr=``. Annotate so pyright sees
    # the typed wrappers that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    Pt: Scalar
    p: Scalar
    _log_extension: bool
    _Sp: float

    def __post_init__(self) -> None:
        # Mirror the C++ constructor's mutual-consistency checks. The HIT
        # parser doesn't distinguish "default" from "user-supplied" on
        # primitive options the way the C++ side does, so we only enforce the
        # range + the log-extension-requires-Sp pairing.
        if self._log_extension and self._Sp <= 0.0:
            raise ValueError(
                "BrooksCoreyCapillaryPressure: log_extension is set to true, but "
                "transition_saturation is not specified (or is zero)."
            )
        if self._Sp < 0.0 or self._Sp > 1.0:
            raise ValueError(
                "BrooksCoreyCapillaryPressure: transition_saturation must be in the range [0, 1]."
            )

    def forward(  # type: ignore[override]
        self,
        effective_saturation: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        S = effective_saturation
        Pt = self._get_param("Pt", nl_params, Scalar)
        p = self._get_param("p", nl_params, Scalar)

        # ``Pc = Pt * S^(-1/p)`` and its derivative ``dPc/dS = -Pt/p *
        # S^(-1/p - 1)`` — typed Scalar algebra end to end, matching
        # ``BrooksCoreyCapillaryPressure::calculate_pressure``. ``pow`` and
        # ``where`` are declared generic over ``TensorWrapper``; narrow the
        # results back to ``Scalar`` so pyright sees the Scalar algebra.
        neg_inv_p = -1.0 / p
        Pc_base = Pt * pow(S, neg_inv_p)
        dPc_dS_base = -Pt / p * pow(S, neg_inv_p - 1.0)

        if self._log_extension:
            # Evaluate the Brooks-Corey branch at the transition saturation
            # ``Sp`` to anchor the log-linear extension below ``Sp``.
            ln10 = math.log(10.0)
            Sp = Scalar.from_value(self._Sp, like=S)
            Pcs = Pt * pow(Sp, neg_inv_p)
            dPcs_dS = -Pt / p * pow(Sp, neg_inv_p - 1.0)
            slope = dPcs_dS / (ln10 * Pcs)
            yintercept = log10(Pcs) - slope * Sp
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


__all__ = ["BrooksCoreyCapillaryPressure"]
