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

"""Python-native mirror of C++ ``solid_mechanics/kinematics/ThermalDeformationJacobian.h``."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("ThermalDeformationJacobian")
class ThermalDeformationJacobian(Model):
    r"""Define the linear isotropic thermal deformation Jacobian, i.e. $J = 1 + \alpha (T - T_0)$,
    where $\alpha$ is the coefficient of thermal expansion (CTE), $T$ is
    the temperature, and $T_0$ is the reference (stress-free) temperature.
    """

    # Forward: ``J = 1 + alpha * (T - T0)``. The leaf is linear in ``T``
    # (action wrt ``T`` is ``alpha * V``) and linear in ``alpha`` when promoted
    # to a nonlinear input (action is ``(T - T0) * V``). Pure typed wrapper
    # algebra throughout: the Scalar ``+ - *`` operators handle
    # sub-batch alignment, so no ``torch.<op>(.data)`` calls appear in the
    # forward or chain-rule action bodies.

    hit = HitSchema(
        input("temperature", Scalar, "Temperature"),
        output("jacobian", Scalar, "Thermal deformation Jacobian"),
        parameter(
            "reference_temperature",
            Scalar,
            "Reference (stress-free) temperature",
            attr="T0",
        ),
        parameter(
            "CTE",
            Scalar,
            "Coefficient of thermal expansion",
            attr="alpha",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares ``reference_temperature`` (stored as ``T0``)
    # and ``CTE`` (stored as ``alpha``). Annotate so pyright sees the typed
    # wrappers that ``Model.__getattr__`` returns rather than the inherited
    # ``nn.Module`` hint.
    T0: Scalar
    alpha: Scalar

    def forward(  # type: ignore[override]
        self,
        temperature: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        T = temperature
        T0 = self.T0
        alpha = self._get_param("alpha", nl_params, Scalar)
        J = 1.0 + alpha * (T - T0)
        if v is None:
            return J

        # Differential pushforward: linear in T and (when promoted)
        # linear in alpha; each action is the same wrapper map applied to the
        # incoming Scalar tangent V, scaled by the held coefficient. No
        # Jacobian materialised.
        def temperature_action(V: Scalar) -> Scalar:
            return alpha * V

        actions: dict = {"temperature": temperature_action}

        if "alpha" in self._nl_params:
            dT = T - T0
            actions[self._nl_params["alpha"].input_name] = lambda V, c=dT: c * V

        return J, self.apply_chain_rule(v, "jacobian", actions, output=J)


__all__ = ["ThermalDeformationJacobian"]
