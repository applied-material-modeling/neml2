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

"""Python-native mirror of C++ ``SwellingAndPhaseChangeDeformationJacobian`` (kinematics)."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar


@register_native("SwellingAndPhaseChangeDeformationJacobian")
class SwellingAndPhaseChangeDeformationJacobian(Model):
    r"""Define the linear isotropic phase change deformation Jacobian for a freezing liquid or a
    melting solid, i.e. $J = \left( 1 + \alpha c \phi^f + (1-c) \phi^f \Delta \Omega \right)$, where
    $\alpha$ is the coefficient of swelling, $\Delta \Omega$ is relative
    difference of the reference volume between the two phases, $\phi^f$ is the fluid
    fraction associated with swelling, and $c$ is the phase fraction.
    """

    # Forward: ``J = 1 + alpha * c * vf + (1 - c) * vf * dOmega``. Linear in
    # the fluid_fraction input (action = (alpha*c + (1-c)*dOmega) * V) and,
    # when the phase_fraction parameter is promoted to a nonlinear input, also
    # linear in c (action = (alpha*vf - vf*dOmega) * V). Pure typed wrapper
    # algebra throughout: Scalar + - * handle sub-batch alignment, so
    # no torch.<op>(.data) calls appear in forward or chain-rule action bodies.

    hit = HitSchema(
        input("fluid_fraction", Scalar, "Volume fraction of the fluid phase."),
        output("jacobian", Scalar, "Phase change deformation Jacobian"),
        parameter(
            "phase_fraction",
            Scalar,
            "Phase fraction during the phase change. 0 means all solid, 1 means all liquid.",
            attr="c",
            allow_nonlinear=True,
        ),
        parameter(
            "swelling_coefficient",
            Scalar,
            "Coefficient of phase expansion",
            attr="alpha",
        ),
        parameter(
            "reference_volume_difference",
            Scalar,
            "Relative difference between the reference volumes of the two phases.",
            attr="dOmega",
        ),
    )

    # ``from_hit`` auto-declares ``phase_fraction`` (stored as ``c``),
    # ``swelling_coefficient`` (stored as ``alpha``), and
    # ``reference_volume_difference`` (stored as ``dOmega``). Annotate so
    # pyright sees the typed wrappers that ``Model.__getattr__`` returns
    # rather than the inherited ``nn.Module`` hint.
    c: Scalar
    alpha: Scalar
    dOmega: Scalar

    def forward(  # type: ignore[override]
        self,
        fluid_fraction: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        vf = fluid_fraction
        c = self._get_param("c", nl_params, Scalar)
        alpha = self.alpha
        dOmega = self.dOmega
        J = 1.0 + alpha * c * vf + (1.0 - c) * vf * dOmega
        if v is None:
            return J

        # Differential pushforward: linear in vf (and, when promoted,
        # linear in c). Each action is the same wrapper map applied to the
        # incoming Scalar tangent V scaled by the held coefficient. No
        # Jacobian materialised. Mirrors the C++
        # ``_J.d(_vf) = (alpha*c + (1-c)*dOmega)`` and
        # ``_J.d(c) = (alpha*vf - vf*dOmega)`` derivatives.
        dJ_dvf = alpha * c + (1.0 - c) * dOmega

        def fluid_fraction_action(V: Scalar, coef=dJ_dvf) -> Scalar:
            return coef * V

        actions: dict = {"fluid_fraction": fluid_fraction_action}

        if "c" in self._nl_params:
            dJ_dc = alpha * vf - vf * dOmega
            actions[self._nl_params["c"].input_name] = lambda V, coef=dJ_dc: coef * V

        return J, self.apply_chain_rule(v, "jacobian", actions, output=J)


__all__ = ["SwellingAndPhaseChangeDeformationJacobian"]
