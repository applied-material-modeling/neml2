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

"""Python-native mirrors of C++ ``common/InputParameter.h``.

The C++ ``InputParameter<T>`` is instantiated for every primitive tensor type
via the ``FOR_ALL_PRIMITIVETENSOR`` macro. The native surface re-implements the
subset of those types that already have a typed wrapper under
``neml2.types`` (``Scalar``, ``Vec``, ``Rot``, ``WR2``, ``R2``, ``SR2``,
``SSR4``, ``MillerIndex``) -- the same set as ``ConstantParameter`` /
``CopyVariable``. One shared ``_InputParameter`` base holds the
identity forward/action; each registered variant only differs in the ``hit``
schema's wrapper type.

The C++ side assigns ``_p = _input_var()`` (forward) and
``_p.d(_input_var) = imap_v<T>(...)`` (the identity Jacobian). The native
equivalent is wrapper-algebra identity ``parameter = variable`` with a
closed-form identity pushforward ``action(V) = V`` -- the canonical D-062
linear leaf, structurally identical to ``CopyVariable`` but exposing the
input as a "parameter" output rather than a generic "to" copy.
"""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import (
    R2,
    SR2,
    SSR4,
    WR2,
    MillerIndex,
    Rot,
    Scalar,
    TensorWrapper,
    Vec,
)


class _InputParameter(Model):
    """``parameter = variable`` -- promote an input variable into a parameter output.

    Mirrors the C++ ``InputParameter<T>``. The forward is the identity on a
    typed wrapper, and the D-062 pushforward is the identity action -- so the
    body is purely typed wrapper algebra with no ``.data`` access. The
    canonical I/O names match the C++ option names (``variable`` input /
    ``parameter`` output); HIT overrides on those options resolve through
    ``_store_schema_values``.
    """

    def forward(  # type: ignore[override]
        self,
        variable: TensorWrapper,
        *nl_params: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # Forward: typed identity. No .data, no torch.* -- the typed wrapper
        # is its own pass-through value (mirrors the C++ ``_p = _input_var()``).
        out = variable
        if v is None:
            return out
        # Linear leaf: d(parameter)/d(variable) = I, so the action on any
        # incoming tangent V of the input's type is V itself (the typed mirror
        # of the C++ ``imap_v<T>(_input_var.options())`` identity Jacobian).
        return out, self.apply_chain_rule(
            v,
            "parameter",
            {"variable": lambda V: V},
            output=out,
        )


@register_native("ScalarInputParameter")
class ScalarInputParameter(_InputParameter):
    """Scalar-valued input parameter. Mirrors ``InputParameter<Scalar>``."""

    hit = HitSchema(
        input("variable", Scalar, "The input variable that defines this parameter"),
        output("parameter", Scalar, "Output parameter"),
    )


@register_native("VecInputParameter")
class VecInputParameter(_InputParameter):
    """Vec-valued input parameter. Mirrors ``InputParameter<Vec>``."""

    hit = HitSchema(
        input("variable", Vec, "The input variable that defines this parameter"),
        output("parameter", Vec, "Output parameter"),
    )


@register_native("RotInputParameter")
class RotInputParameter(_InputParameter):
    """Rot-valued input parameter. Mirrors ``InputParameter<Rot>``."""

    hit = HitSchema(
        input("variable", Rot, "The input variable that defines this parameter"),
        output("parameter", Rot, "Output parameter"),
    )


@register_native("WR2InputParameter")
class WR2InputParameter(_InputParameter):
    """WR2-valued input parameter. Mirrors ``InputParameter<WR2>``."""

    hit = HitSchema(
        input("variable", WR2, "The input variable that defines this parameter"),
        output("parameter", WR2, "Output parameter"),
    )


@register_native("R2InputParameter")
class R2InputParameter(_InputParameter):
    """R2-valued input parameter. Mirrors ``InputParameter<R2>``."""

    hit = HitSchema(
        input("variable", R2, "The input variable that defines this parameter"),
        output("parameter", R2, "Output parameter"),
    )


@register_native("SR2InputParameter")
class SR2InputParameter(_InputParameter):
    """SR2-valued input parameter. Mirrors ``InputParameter<SR2>``."""

    hit = HitSchema(
        input("variable", SR2, "The input variable that defines this parameter"),
        output("parameter", SR2, "Output parameter"),
    )


@register_native("SSR4InputParameter")
class SSR4InputParameter(_InputParameter):
    """SSR4-valued input parameter. Mirrors ``InputParameter<SSR4>``."""

    hit = HitSchema(
        input("variable", SSR4, "The input variable that defines this parameter"),
        output("parameter", SSR4, "Output parameter"),
    )


@register_native("MillerIndexInputParameter")
class MillerIndexInputParameter(_InputParameter):
    """MillerIndex-valued input parameter. Mirrors ``InputParameter<MillerIndex>``."""

    hit = HitSchema(
        input("variable", MillerIndex, "The input variable that defines this parameter"),
        output("parameter", MillerIndex, "Output parameter"),
    )


__all__ = [
    "MillerIndexInputParameter",
    "R2InputParameter",
    "RotInputParameter",
    "SR2InputParameter",
    "SSR4InputParameter",
    "ScalarInputParameter",
    "VecInputParameter",
    "WR2InputParameter",
]
