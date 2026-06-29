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

"""Python-native mirrors of C++ ``common/CopyVariable.h``.

The C++ ``CopyVariable<T>`` is instantiated for every primitive tensor type via
the ``FOR_ALL_PRIMITIVETENSOR`` macro. The native surface re-implements the
subset that already has a typed wrapper under ``neml2.types``
(``Scalar``, ``Vec``, ``MRP``, ``WR2``, ``R2``, ``SR2``, ``SSR4``,
``MillerIndex``) -- the same set as ``ConstantParameter``. One shared
``_CopyVariable`` base holds the identity forward/action; each registered
variant only differs in the ``hit`` schema's wrapper type.

The C++ side assigns ``_to = _from()`` (forward) and
``_to.d(_from) = imap_v<T>(...)`` (the identity Jacobian). The native
equivalent is wrapper-algebra identity ``out = inp`` with a closed-form
identity pushforward ``action(V) = V`` -- the canonical D-062 linear leaf
(see ``CrackGeometricFunctionAT1`` for the same pattern on a single Scalar).
"""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import (
    MRP,
    R2,
    SR2,
    SSR4,
    WR2,
    MillerIndex,
    Scalar,
    TensorWrapper,
    Vec,
)
from ..chain_rule import ChainRuleDict
from ..model import Model


class _CopyVariable(Model):
    """``to = from`` -- copy one typed variable into another.

    Mirrors the C++ ``CopyVariable<T>``. The forward is the identity on a
    typed wrapper, and the D-062 pushforward is the identity action -- so the
    body is purely typed wrapper algebra with no ``.data`` access. The
    canonical I/O names match the C++ option names (``from`` / ``to``); HIT
    overrides on those options resolve through ``_store_schema_values``.
    """

    def forward(  # type: ignore[override]
        self,
        inp: TensorWrapper,
        *promoted_params: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # Forward: typed identity. No .data, no torch.* -- the typed wrapper
        # is its own pass-through value.
        out = inp
        if v is None:
            return out
        # Linear leaf: d(to)/d(from) = I, so the action on any incoming
        # tangent V of the input's type is V itself (the typed mirror of the
        # C++ ``imap_v<T>(_from.options())`` identity Jacobian).
        return out, self.apply_chain_rule(
            v,
            "to",
            {"from": lambda V: V},
            output=out,
        )


@register_neml2_object("CopyScalar")
class CopyScalar(_CopyVariable):
    """Scalar-valued variable copy. Mirrors ``CopyVariable<Scalar>``."""

    hit = HitSchema(
        input("from", Scalar, "Variable to copy value from"),
        output("to", Scalar, "Variable to copy value to"),
    )


@register_neml2_object("CopyVec")
class CopyVec(_CopyVariable):
    """Vec-valued variable copy. Mirrors ``CopyVariable<Vec>``."""

    hit = HitSchema(
        input("from", Vec, "Variable to copy value from"),
        output("to", Vec, "Variable to copy value to"),
    )


@register_neml2_object("CopyMRP")
class CopyMRP(_CopyVariable):
    """MRP-valued variable copy. Mirrors ``CopyVariable<MRP>``."""

    hit = HitSchema(
        input("from", MRP, "Variable to copy value from"),
        output("to", MRP, "Variable to copy value to"),
    )


@register_neml2_object("CopyWR2")
class CopyWR2(_CopyVariable):
    """WR2-valued variable copy. Mirrors ``CopyVariable<WR2>``."""

    hit = HitSchema(
        input("from", WR2, "Variable to copy value from"),
        output("to", WR2, "Variable to copy value to"),
    )


@register_neml2_object("CopyR2")
class CopyR2(_CopyVariable):
    """R2-valued variable copy. Mirrors ``CopyVariable<R2>``."""

    hit = HitSchema(
        input("from", R2, "Variable to copy value from"),
        output("to", R2, "Variable to copy value to"),
    )


@register_neml2_object("CopySR2")
class CopySR2(_CopyVariable):
    """SR2-valued variable copy. Mirrors ``CopyVariable<SR2>``."""

    hit = HitSchema(
        input("from", SR2, "Variable to copy value from"),
        output("to", SR2, "Variable to copy value to"),
    )


@register_neml2_object("CopySSR4")
class CopySSR4(_CopyVariable):
    """SSR4-valued variable copy. Mirrors ``CopyVariable<SSR4>``."""

    hit = HitSchema(
        input("from", SSR4, "Variable to copy value from"),
        output("to", SSR4, "Variable to copy value to"),
    )


@register_neml2_object("CopyMillerIndex")
class CopyMillerIndex(_CopyVariable):
    """MillerIndex-valued variable copy. Mirrors ``CopyVariable<MillerIndex>``."""

    hit = HitSchema(
        input("from", MillerIndex, "Variable to copy value from"),
        output("to", MillerIndex, "Variable to copy value to"),
    )


__all__ = [
    "CopyMillerIndex",
    "CopyR2",
    "CopyMRP",
    "CopySR2",
    "CopySSR4",
    "CopyScalar",
    "CopyVec",
    "CopyWR2",
]
