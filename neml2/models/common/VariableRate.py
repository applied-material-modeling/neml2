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

"""Python-native mirrors of C++ ``common/VariableRate.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, derived_input, derived_output, input
from ...types import (
    R2,
    SR2,
    Scalar,
    TensorWrapper,
    Vec,
)


class _VariableRate(Model):
    """``rate = (v - v~1) / (t - t~1)`` -- first-order discrete time derivative.

    Generic over the wrapped tensor type (Scalar, Vec, SR2, R2). Subclasses
    set ``_type`` and assemble the variant-specific ``hit`` schema (the
    input/output types differ across the four instantiations).

    The schema exposes only the user-facing knobs (``variable`` is the
    differentiated variable, ``time`` defaults to ``"t"``). The four
    derived names -- ``<variable>~1``, ``<time>~1``, and the
    ``<variable>_rate`` output -- fall out of those base options via the
    schema's ``suffix=`` machinery; no ``__init__`` needed.
    """

    _type: type[TensorWrapper]

    _v: str
    _vn: str
    _t: str
    _tn: str
    _rate: str

    def forward(  # type: ignore[override]
        self,
        x: TensorWrapper,
        x_n: TensorWrapper,
        t: Scalar,
        t_n: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Forward: ``rate = (x - x_n) / (t - t_n)`` in typed wrapper algebra.
        # Sub-batch alignment of (possibly scalar) ``dt`` against per-site
        # ``dx`` is handled inside the wrapper operators.
        dt = t - t_n  # Scalar
        dx = x - x_n  # TensorWrapper of the variant's type
        rate = dx / dt
        if v is None:
            return rate
        # Differential pushforwards:
        #   d rate / d x   = I / dt        -> V / dt
        #   d rate / d x_n = -I / dt       -> -V / dt
        #   d rate / d t   = -rate / dt    -> -(rate / dt) * V (V scalar)
        #   d rate / d t_n =  rate / dt    ->  (rate / dt) * V (V scalar)
        rate_over_dt = rate / dt  # cached for the two time tangents
        actions = {
            self._v: lambda V, c=dt: V / c,
            self._vn: lambda V, c=dt: -(V / c),
            self._t: lambda V, c=rate_over_dt: -(c * V),
            self._tn: lambda V, c=rate_over_dt: c * V,
        }
        return rate, self.apply_chain_rule(v, self._rate, actions, output=rate)


def _variable_rate_schema(t: type[TensorWrapper]) -> HitSchema:
    """Schema shared by all four subclasses, parameterized by the value type.

    User-facing options mirror the C++ ``VariableRate<T>::expected_options``:
    ``variable`` is the variable being differentiated, ``time`` defaults to
    ``"t"``. The three derived names (``variable~1``, ``time~1``,
    ``variable_rate``) follow via ``suffix=``. Field order matches the leaf's
    ``forward`` positional signature (``v, v_n, t, t_n``).
    """
    return HitSchema(
        input("variable", t, "The variable being differentiated", attr="_v"),
        derived_input("variable", t, attr="_vn", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_tn", suffix="~1"),
        derived_output("variable", t, attr="_rate", suffix="_rate"),
    )


@register_native("ScalarVariableRate")
class ScalarVariableRate(_VariableRate):
    r"""Calculate the first order discrete time derivative of a variable as
    $\dot{f} = \frac{f-f_n}{t-t_n}$, where $f$ is the variable, $f_n$ is the variable at the
    previous time step, and $t$ is time.
    """

    _type = Scalar
    hit = _variable_rate_schema(Scalar)


@register_native("VecVariableRate")
class VecVariableRate(_VariableRate):
    r"""Calculate the first order discrete time derivative of a variable as
    $\dot{f} = \frac{f-f_n}{t-t_n}$, where $f$ is the variable, $f_n$ is the variable at the
    previous time step, and $t$ is time.
    """

    _type = Vec
    hit = _variable_rate_schema(Vec)


@register_native("SR2VariableRate")
class SR2VariableRate(_VariableRate):
    r"""Calculate the first order discrete time derivative of a variable as
    $\dot{f} = \frac{f-f_n}{t-t_n}$, where $f$ is the variable, $f_n$ is the variable at the
    previous time step, and $t$ is time.
    """

    _type = SR2
    hit = _variable_rate_schema(SR2)


@register_native("R2VariableRate")
class R2VariableRate(_VariableRate):
    r"""Calculate the first order discrete time derivative of a variable as
    $\dot{f} = \frac{f-f_n}{t-t_n}$, where $f$ is the variable, $f_n$ is the variable at the
    previous time step, and $t$ is time.
    """

    _type = R2
    hit = _variable_rate_schema(R2)


__all__ = [
    "R2VariableRate",
    "SR2VariableRate",
    "ScalarVariableRate",
    "VecVariableRate",
]
