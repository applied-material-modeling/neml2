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

"""Python-native mirror of the C++ ``EffectiveSaturation`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("EffectiveSaturation")
class EffectiveSaturation(Model):
    r"""Define the effective saturation, taking the form of
    $S = \frac{\frac{\phi}{\phi_\mathrm{max}} - S_r}{1-S_r}$ where
    $\phi$ is the volume fraction of the flowing fluid,
    $\phi_\mathrm{max}$ is the maximum allowable volume fraction, and
    $S_r$ is the residual saturation.
    """

    hit = HitSchema(
        input("fluid_fraction", Scalar, "Volume fraction of the fluid"),
        output("effective_saturation", Scalar, "Effective saturation"),
        parameter(
            "residual_saturation",
            Scalar,
            "Liquid's residual volume fraction",
            attr="Sr",
        ),
        parameter(
            "max_fraction",
            Scalar,
            "Maximum allowable volume fraction of the fluid",
            attr="phi_max",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``residual_saturation`` parameter (stored
    # as ``Sr``) and the ``max_fraction`` parameter (stored as ``phi_max``).
    # Annotate so pyright sees the typed wrapper that ``Model.__getattr__``
    # returns rather than ``nn.Module``'s generic ``Module`` hint.
    Sr: Scalar
    phi_max: Scalar

    def forward(  # type: ignore[override]
        self,
        fluid_fraction: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        phi = fluid_fraction
        Sr = self.Sr
        phi_max = self._get_param("phi_max", nl_params, Scalar)

        # ``S = (phi/phi_max - Sr) / (1 - Sr)``
        one_minus_Sr = -Sr + 1.0
        S = (phi / phi_max - Sr) / one_minus_Sr

        if v is None:
            return S

        # Differential pushforward: linear coefficients applied to the
        # incoming tangents. dS/dphi = 1 / (phi_max * (1 - Sr));
        # dS/dphi_max = -phi / (phi_max * phi_max * (1 - Sr)).
        denom = phi_max * one_minus_Sr
        dS_dphi = denom**-1.0
        actions: dict[str, ChainRuleAction] = {
            "fluid_fraction": lambda V, c=dS_dphi: c * V,
        }

        # ``max_fraction`` may have been promoted to a runtime input (mode 3/4)
        # via ``allow_nonlinear=True``. Only add a chain-rule action under its
        # resolved external input name if it was actually promoted; otherwise
        # it's a static parameter and contributes no tangent.
        phi_max_nl = self._nl_params.get("phi_max")
        if phi_max_nl is not None:
            dS_dphi_max = -phi / (phi_max * denom)
            actions[phi_max_nl.input_name] = lambda V, c=dS_dphi_max: c * V

        return S, self.apply_chain_rule(
            v,
            "effective_saturation",
            actions,
            output=S,
        )


__all__ = ["EffectiveSaturation"]
