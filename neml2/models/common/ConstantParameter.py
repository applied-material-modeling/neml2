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

"""Python-native mirrors of C++ ``common/ConstantParameter.h``.

The C++ ``ConstantParameter<T>`` is instantiated for every primitive tensor
type via the ``FOR_ALL_PRIMITIVETENSOR`` macro. The native surface only
re-implements the subset of those types that already have a typed wrapper
under ``neml2.types`` (``Scalar``, ``Vec``, ``Rot``, ``WR2``, ``R2``,
``SR2``, ``SSR4``, ``MillerIndex``). One shared ``_ConstantParameter`` base
holds the forward/action logic; each registered variant only differs in the
``hit`` schema's wrapper type.
"""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import BLOCK_NAME, HitSchema, output, parameter
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
from ..chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ..model import Model


class _ConstantParameter(Model):
    """``parameter = value`` — a single typed constant exposed as an output variable.

    Mirrors the C++ ``ConstantParameter<T>``. The ``value`` parameter is declared
    with ``allow_promotion=True`` so it independently resolves through the four
    ``declare_typed_parameter`` modes (literal HIT value / ``[Tensors]`` cross-ref
    / ``[Models]`` output wiring → promoted input / bare input promotion). When
    ``value`` is statically bound (mode 1 or 2) the output simply returns it;
    when ``value`` is promoted to a runtime input (mode 3 or 4) the forward
    forwards the input through unchanged, and the chain-rule action is the
    identity $∂parameter/∂value = I$ — the typed mirror of the C++
    ``imap_v<T>(value->options())``.
    """

    # The leaf is linear in ``value`` (out = value), so all second-order
    # partials vanish (∂²out/∂value² = 0). Setting SUPPORTS_SECOND_ORDER lets
    # it be wrapped by Normality, which demands second-order support on every
    # inner leaf. The forward routes through ``propagate_tangents`` with no
    # ``actions_2``; ``apply_chain_rule_2`` collapses v2 / vh to applying
    # ``actions_1`` on the input tangents, which is the correct zero-Hessian
    # contribution for a linear leaf.
    SUPPORTS_SECOND_ORDER = True

    # Set by each registered subclass: the typed wrapper class T that ``value``
    # and the ``parameter`` output share.
    _value_type: type[TensorWrapper]

    # Resolved output variable name (default = HIT block name) — written by
    # ``_store_schema_values`` from the ``parameter`` output's ``attr=``.
    _p: str
    # Static-mode storage for the ``value`` parameter, written by
    # ``declare_typed_parameter`` (mode 1/2). Annotated so pyright sees the
    # typed wrapper that ``Model.__getattr__`` re-wraps it as.
    value: TensorWrapper

    def forward(  # type: ignore[override]
        self,
        *promoted_params: TensorWrapper,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # The model has no structural inputs: the ``*args`` pack only ever
        # carries the optional promoted ``value`` (mode 3/4). Read the
        # parameter through ``_get_param`` so both the static and promoted
        # paths return the same typed wrapper.
        value = self._get_param("value", promoted_params, self._value_type)
        # forward: pass the value through as the output.
        out = value
        if v is None:
            return out
        # Linear leaf: ∂out/∂value = I ⇒ the action is the identity. When
        # ``value`` is static no input is promoted, so ``v`` carries no entry
        # for it and ``apply_chain_rule`` returns ``{"parameter": {}}`` — the
        # structural-zero case from the C++ side where ``_p.d(*value)`` is
        # never assigned. Second-order: no ``actions_2`` since the leaf is
        # linear; ``propagate_tangents`` does ``actions_1(v2[a,b])`` which is
        # the correct zero-Hessian contribution.
        return out, *self.propagate_tangents(
            v,
            "parameter",
            {"value": lambda V: V},
            output=out,
            v2=v2,
            vh=vh,
        )


@register_neml2_object("ScalarConstantParameter")
class ScalarConstantParameter(_ConstantParameter):
    """Scalar-valued constant parameter. Mirrors ``ConstantParameter<Scalar>``."""

    _value_type = Scalar
    value: Scalar  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            Scalar,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            Scalar,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("VecConstantParameter")
class VecConstantParameter(_ConstantParameter):
    """Vec-valued constant parameter. Mirrors ``ConstantParameter<Vec>``."""

    _value_type = Vec
    value: Vec  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            Vec,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            Vec,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("RotConstantParameter")
class RotConstantParameter(_ConstantParameter):
    """Rot-valued constant parameter. Mirrors ``ConstantParameter<Rot>``."""

    _value_type = Rot
    value: Rot  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            Rot,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            Rot,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("WR2ConstantParameter")
class WR2ConstantParameter(_ConstantParameter):
    """WR2-valued constant parameter. Mirrors ``ConstantParameter<WR2>``."""

    _value_type = WR2
    value: WR2  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            WR2,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            WR2,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("R2ConstantParameter")
class R2ConstantParameter(_ConstantParameter):
    """R2-valued constant parameter. Mirrors ``ConstantParameter<R2>``."""

    _value_type = R2
    value: R2  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            R2,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            R2,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("SR2ConstantParameter")
class SR2ConstantParameter(_ConstantParameter):
    """SR2-valued constant parameter. Mirrors ``ConstantParameter<SR2>``."""

    _value_type = SR2
    value: SR2  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            SR2,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            SR2,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("SSR4ConstantParameter")
class SSR4ConstantParameter(_ConstantParameter):
    """SSR4-valued constant parameter. Mirrors ``ConstantParameter<SSR4>``."""

    _value_type = SSR4
    value: SSR4  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            SSR4,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            SSR4,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


@register_neml2_object("MillerIndexConstantParameter")
class MillerIndexConstantParameter(_ConstantParameter):
    """MillerIndex-valued constant parameter. Mirrors ``ConstantParameter<MillerIndex>``."""

    _value_type = MillerIndex
    value: MillerIndex  # type: ignore[assignment]

    hit = HitSchema(
        output(
            "parameter",
            MillerIndex,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
            attr="_p",
        ),
        parameter(
            "value",
            MillerIndex,
            "The constant value of the parameter",
            attr="value",
            allow_promotion=True,
        ),
    )


__all__ = [
    "MillerIndexConstantParameter",
    "R2ConstantParameter",
    "RotConstantParameter",
    "SR2ConstantParameter",
    "SSR4ConstantParameter",
    "ScalarConstantParameter",
    "VecConstantParameter",
    "WR2ConstantParameter",
]
