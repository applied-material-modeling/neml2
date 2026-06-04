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

"""Python-native mirrors of C++ ``common/BackwardEulerTimeIntegration.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, derived_input, derived_output, input, option
from ...types import (
    R2,
    SR2,
    Scalar,
    TensorWrapper,
    Vec,
)


class _BackwardEulerTimeIntegration(Model):
    """``residual = s − s~1 − (t − t~1) · s_rate``.

    Generic over the wrapped tensor type (Scalar or SR2). Subclasses set
    ``_type`` and assemble the variant-specific ``hit`` schema (the input/
    output types differ between the Scalar and SR2 instantiations).

    The schema exposes only the user-facing knobs (``variable`` is required,
    ``time`` defaults to ``"t"``, ``rate`` is an optional explicit override
    for the rate-variable name). All five I/O variables —
    ``<variable>``, ``<variable>~1``, ``<variable>_rate`` (or ``rate``),
    ``<time>``, ``<time>~1``, and the ``<variable>_residual`` output — are
    derived from those base options by the schema; no ``__init__`` needed.
    """

    _type: type[TensorWrapper]

    _var: str
    _var_n: str
    _t: str
    _t_n: str
    _rate: str
    _residual: str

    def forward(  # type: ignore[override]
        self,
        s: TensorWrapper,
        s_n: TensorWrapper,
        t: Scalar,
        t_n: Scalar,
        s_dot: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # residual ``s - s_n - dt * s_dot`` in typed wrapper ops; alignment of
        # global ``dt`` against (possibly) per-sub-batch-site state is automatic.
        dt = t - t_n  # Scalar
        r = s - s_n - dt * s_dot
        if v is None:
            return r
        # Differential pushforwards — the push-forward of each input
        # tangent, no ``dt``-padding tricks, no Jacobian:
        #   ∂r/∂s = I → V;  ∂r/∂s~1 = -I → -V;  ∂r/∂s_rate = -dt·I → -dt·V
        #   ∂r/∂t = -s_rate → -s_rate·V (V scalar);  ∂r/∂t~1 = s_rate → s_rate·V
        actions = {
            self._var: lambda V: V,
            self._var_n: lambda V: -V,
            self._rate: lambda V, c=dt: -(c * V),
            self._t: lambda V, c=s_dot: -(c * V),
            self._t_n: lambda V, c=s_dot: c * V,
        }
        return r, self.apply_chain_rule(v, self._residual, actions, output=r)


def _backward_euler_schema(t: type[TensorWrapper]) -> HitSchema:
    """Schema shared by both subclasses, parameterized by the value type.

    User-facing options: ``variable`` (required), ``time`` (defaults to "t"),
    ``rate`` (optional explicit override). All five I/O variables are derived
    from those. The ``derived_*`` field order matches the leaf's ``forward``
    positional signature (``s, s_n, t, t_n, s_dot``).
    """
    return HitSchema(
        input("variable", t, "Variable being integrated", attr="_var"),
        derived_input("variable", t, attr="_var_n", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        derived_input("variable", t, attr="_rate", suffix="_rate", override="rate"),
        option("rate", str, "Override name for the variable rate.", default="", attr="_rate_opt"),
        derived_output("variable", t, attr="_residual", suffix="_residual"),
    )


@register_neml2_object("ScalarBackwardEulerTimeIntegration")
class ScalarBackwardEulerTimeIntegration(_BackwardEulerTimeIntegration):
    r"""Define the backward Euler time integration residual
    $r = s - s_n - (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = Scalar
    hit = _backward_euler_schema(Scalar)


@register_neml2_object("SR2BackwardEulerTimeIntegration")
class SR2BackwardEulerTimeIntegration(_BackwardEulerTimeIntegration):
    r"""Define the backward Euler time integration residual
    $r = s - s_n - (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = SR2
    hit = _backward_euler_schema(SR2)


@register_neml2_object("R2BackwardEulerTimeIntegration")
class R2BackwardEulerTimeIntegration(_BackwardEulerTimeIntegration):
    r"""Define the backward Euler time integration residual
    $r = s - s_n - (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = R2
    hit = _backward_euler_schema(R2)


@register_neml2_object("VecBackwardEulerTimeIntegration")
class VecBackwardEulerTimeIntegration(_BackwardEulerTimeIntegration):
    r"""Define the backward Euler time integration residual
    $r = s - s_n - (t - t_n) \dot{s}$, where $s$ is the variable being
    integrated, $\dot{s}$ is the variable rate, and $t$ is time.
    Subscripts $n$ denote quantities from the previous time step.
    """

    _type = Vec
    hit = _backward_euler_schema(Vec)


__all__ = [
    "ScalarBackwardEulerTimeIntegration",
    "SR2BackwardEulerTimeIntegration",
    "R2BackwardEulerTimeIntegration",
    "VecBackwardEulerTimeIntegration",
]
