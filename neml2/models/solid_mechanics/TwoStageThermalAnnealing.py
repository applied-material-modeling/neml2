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

"""Python-native mirrors of C++ ``solid_mechanics/TwoStageThermalAnnealing.h``.

The C++ ``TwoStageThermalAnnealing<T>`` is instantiated for ``Scalar`` and
``SR2`` (one ``REGISTER(T)`` per primitive type). One shared
``_TwoStageThermalAnnealing`` base holds the forward/action logic; each
registered variant only differs in the ``hit`` schema's wrapper type.
"""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import SR2, Scalar, TensorWrapper, lt, where
from ..chain_rule import ChainRuleDict
from ..model import Model


class _TwoStageThermalAnnealing(Model):
    r"""Thermal-annealing recovery for a hardening variable of type $T$.

    For temperatures below $T_1$ the model keeps the base hardening rate.
    For temperatures above $T_1$ but below $T_2$ the model zeros
    the hardening rate. For temperatures above $T_2$ the model replaces
    the hardening rate with $\dot{h} = -h / \tau$ where $\tau$ is
    the rate of recovery.

    Mirrors the C++ ``TwoStageThermalAnnealing<T>``.
    """

    # Set by each registered subclass: the typed wrapper class T that
    # ``base_rate``, ``base`` and ``modified_rate`` share.
    _value_type: type[TensorWrapper]

    # ``from_hit`` auto-declares the three Scalar parameters (stored under
    # the option names ``T1``, ``T2``, ``tau``). Annotate so pyright sees the
    # typed wrappers that ``Model.__getattr__`` returns.
    T1: Scalar
    T2: Scalar
    tau: Scalar

    def forward(  # type: ignore[override]
        self,
        base_rate: TensorWrapper,
        base: TensorWrapper,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        T1 = self._get_param("T1", nl_params, Scalar)
        T2 = self._get_param("T2", nl_params, Scalar)
        tau = self._get_param("tau", nl_params, Scalar)

        # Region masks as Scalar 0/1 indicators. ``lt`` returns a Scalar
        # boolean tensor; ``where`` selects between the typed Scalar branches.
        # Recovery region uses ``T >= T2`` (i.e. NOT ``T < T2``) to match the
        # C++ ``where(_T >= _T2, one, zero)``; we invert by swapping the
        # ``where`` branches rather than introducing a ``ge`` primitive.
        one = Scalar.from_value(1.0, like=temperature)
        zero = Scalar.from_value(0.0, like=temperature)
        base_region = where(lt(temperature, T1), one, zero)
        recover_region = where(lt(temperature, T2), zero, one)

        # Forward: modified_rate = base_region * base_rate - recover_region * base / tau.
        # ``Scalar * T`` and ``T / Scalar`` broadcast cleanly for both T = Scalar
        # and T = SR2 via the wrapper operator overloads (see SR2.__rmul__ /
        # SR2.__truediv__ for the SR2 specialisation).
        modified_rate = base_region * base_rate - recover_region * base / tau
        if v is None:
            return modified_rate

        # D-062 pushforward: linear in ``base_rate`` and ``base``, no dependence
        # on ``temperature``. Each action is a Scalar coefficient scaling the
        # incoming tangent of the input's type — no Jacobian materialised.
        def base_rate_action(V: TensorWrapper, c=base_region) -> TensorWrapper:
            return c * V

        def base_action(V: TensorWrapper, c=recover_region, t=tau) -> TensorWrapper:
            return -c * V / t

        # ``temperature`` has no declared action — structural zero, matching the
        # C++ ``set_value`` which never assigns ``_modified_rate.d(_T)``.
        return modified_rate, self.apply_chain_rule(
            v,
            "modified_rate",
            {"base_rate": base_rate_action, "base": base_action},
            output=modified_rate,
        )


@register_neml2_object("ScalarTwoStageThermalAnnealing")
class ScalarTwoStageThermalAnnealing(_TwoStageThermalAnnealing):
    """Scalar-valued two-stage thermal annealing. Mirrors ``TwoStageThermalAnnealing<Scalar>``."""

    _value_type = Scalar

    hit = HitSchema(
        input("base_rate", Scalar, "Base hardening rate"),
        input("base", Scalar, "Underlying base hardening variable"),
        input("temperature", Scalar, "Temperature"),
        output("modified_rate", Scalar, "Output for the modified hardening rate."),
        parameter("T1", Scalar, "First stage annealing temperature"),
        parameter("T2", Scalar, "Second stage annealing temperature"),
        parameter("tau", Scalar, "Recovery rate for second stage annealing."),
    )


@register_neml2_object("SR2TwoStageThermalAnnealing")
class SR2TwoStageThermalAnnealing(_TwoStageThermalAnnealing):
    """SR2-valued two-stage thermal annealing. Mirrors ``TwoStageThermalAnnealing<SR2>``."""

    _value_type = SR2

    hit = HitSchema(
        input("base_rate", SR2, "Base hardening rate"),
        input("base", SR2, "Underlying base hardening variable"),
        input("temperature", Scalar, "Temperature"),
        output("modified_rate", SR2, "Output for the modified hardening rate."),
        parameter("T1", Scalar, "First stage annealing temperature"),
        parameter("T2", Scalar, "Second stage annealing temperature"),
        parameter("tau", Scalar, "Recovery rate for second stage annealing."),
    )


__all__ = ["SR2TwoStageThermalAnnealing", "ScalarTwoStageThermalAnnealing"]
