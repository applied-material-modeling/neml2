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

"""Python-native mirrors of C++ ``common/ForwardEulerTimeIntegration.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, derived_input, input, option, output
from ...types import (
    SR2,
    Scalar,
    TensorWrapper,
)


class _ForwardEulerTimeIntegration(Model):
    """$s = s~1 + (t − t~1) · s_rate$ — explicit Euler value-update.

    Mirrors :class:`_BackwardEulerTimeIntegration` but produces $s$ directly
    as a value (not a residual). Used in radial-return-style plasticity where
    the integrated state (e.g. ``plastic_strain``) is computed explicitly from
    a known flow-rate and a fixed trial direction, leaving the implicit solve
    1-D on the consistency parameter only.

    Generic over the wrapped tensor type (Scalar or SR2); subclasses set
    ``_type`` and assemble the variant-specific ``hit`` schema. Variable
    names derive from ``variable`` / ``time`` via the schema's ``suffix=`` /
    ``override=`` machinery — no ``__init__`` needed.
    """

    _type: type[TensorWrapper]

    _var: str
    _var_n: str
    _t: str
    _t_n: str
    _rate: str

    def forward(  # type: ignore[override]
        self,
        s_n: TensorWrapper,
        t: Scalar,
        t_n: Scalar,
        s_dot: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        dt = t - t_n  # Scalar
        s = s_n + dt * s_dot
        if v is None:
            return s
        # Differential pushforwards:
        #   ∂s/∂s~1 = I → V;  ∂s/∂s_rate = dt·I → dt·V
        #   ∂s/∂t = s_rate → s_rate·V (V scalar);  ∂s/∂t~1 = -s_rate → -s_rate·V
        actions = {
            self._var_n: lambda V: V,
            self._rate: lambda V, c=dt: c * V,
            self._t: lambda V, c=s_dot: c * V,
            self._t_n: lambda V, c=s_dot: -(c * V),
        }
        return s, self.apply_chain_rule(v, self._var, actions, output=s)


def _forward_euler_schema(t: type[TensorWrapper]) -> HitSchema:
    """Schema shared by both subclasses, parameterized by the value type.

    Same base options as backward Euler; ``variable`` here is the *output*
    (the explicitly-computed value), with no residual. Field order matches
    forward's positional signature (``s_n, t, t_n, s_dot``).
    """
    return HitSchema(
        output("variable", t, "Integrated variable", attr="_var"),
        derived_input("variable", t, attr="_var_n", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        derived_input("variable", t, attr="_rate", suffix="_rate", override="rate"),
        option("rate", str, "Override name for the variable rate.", default="", attr="_rate_opt"),
    )


@register_native("ScalarForwardEulerTimeIntegration")
class ScalarForwardEulerTimeIntegration(_ForwardEulerTimeIntegration):
    r"""Perform forward Euler time integration defined as
    $s = s_n + (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = Scalar
    hit = _forward_euler_schema(Scalar)


@register_native("SR2ForwardEulerTimeIntegration")
class SR2ForwardEulerTimeIntegration(_ForwardEulerTimeIntegration):
    r"""Perform forward Euler time integration defined as
    $s = s_n + (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = SR2
    hit = _forward_euler_schema(SR2)


__all__ = ["ScalarForwardEulerTimeIntegration", "SR2ForwardEulerTimeIntegration"]
