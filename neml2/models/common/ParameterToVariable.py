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

"""Python-native mirrors of C++ ``common/ParameterToVariable.h``.

The C++ ``ParameterToVariable<T>`` is instantiated for every primitive tensor
type via the ``FOR_ALL_PRIMITIVETENSOR`` macro. The native surface only
re-implements the subset of those types that already have a typed wrapper
under ``neml2.types`` (``Scalar``, ``Vec``, ``MRP``, ``WR2``, ``R2``,
``SR2``, ``SSR4``, ``MillerIndex``) -- the same set as ``ConstantParameter`` /
``InputParameter``. One shared ``_ParameterToVariable`` base holds the
identity forward/action; each registered variant only differs in the ``hit``
schema's wrapper type.

The C++ side assigns ``_var = _input_param`` (forward) -- no derivative is
materialised because the ``from`` slot is declared as a *parameter*, not an
input variable. The native equivalent is wrapper-algebra identity
``to = from`` with a closed-form identity pushforward ``action(V) = V`` for
the promoted case (modes 3/4 of ``declare_typed_parameter``). When the
parameter stays statically bound (modes 1/2) the chain-rule dict simply has
no entry for it and ``apply_chain_rule`` returns the structural-zero output
that mirrors the C++ "no Jacobian" behavior.
"""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameter
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


class _ParameterToVariable(Model):
    """``to = from`` -- expose a parameter as an output variable.

    Mirrors the C++ ``ParameterToVariable<T>``. The ``from`` parameter is
    declared with ``allow_promotion=True`` so it independently resolves
    through the four ``declare_typed_parameter`` modes (literal HIT value /
    ``[Tensors]`` cross-ref / ``[Models]`` output wiring -> promoted input /
    bare input promotion). When ``from`` is statically bound the output
    simply returns it; when ``from`` is promoted to a runtime input the
    forward forwards the input through unchanged, and the chain-rule action
    is the identity. The forward is purely typed wrapper algebra with no
    ``.data`` access.
    """

    # Set by each registered subclass: the typed wrapper class T that
    # ``from`` and the ``to`` output share.
    _value_type: type[TensorWrapper]

    # Resolved output variable name (default = literal ``"to"``) -- written
    # by ``_store_schema_values`` from the ``to`` output's ``attr=``.
    _to: str
    # Static-mode storage for the ``from`` parameter, written by
    # ``declare_typed_parameter`` (mode 1/2). Annotated so pyright sees the
    # typed wrapper that ``Model.__getattr__`` re-wraps it as. The ctor_name
    # of the parameter is ``param`` to match the C++ ``declare_parameter<T>(
    # "param", "from")`` naming.
    param: TensorWrapper

    def forward(  # type: ignore[override]
        self,
        *promoted_params: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # The model has no structural inputs: the ``*promoted_params`` pack only
        # ever carries the optional promoted ``param`` (mode 3/4). Read
        # the parameter through ``_get_param`` so both the static and
        # promoted paths return the same typed wrapper.
        from_val = self._get_param("param", promoted_params, self._value_type)
        # forward: identity pass-through -- the typed mirror of the C++
        # ``_var = _input_param`` assignment.
        out = from_val
        if v is None:
            return out
        # Linear leaf: d(to)/d(param) = I ⇒ the action is the identity. When
        # ``param`` is static no input is promoted, so ``v`` carries no
        # entry for it and ``apply_chain_rule`` returns the structural-zero
        # output -- matching the C++ side which never writes a Jacobian
        # because ``from`` is a parameter slot, not a variable.
        return out, self.apply_chain_rule(
            v,
            "to",
            {"param": lambda V: V},
            output=out,
        )


@register_neml2_object("ScalarParameterToVariable")
class ScalarParameterToVariable(_ParameterToVariable):
    """Scalar parameter-to-variable. Mirrors ``ParameterToVariable<Scalar>``."""

    _value_type = Scalar
    param: Scalar  # type: ignore[assignment]

    hit = HitSchema(
        output("to", Scalar, "The name of the variable", attr="_to"),
        parameter(
            "from",
            Scalar,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("VecParameterToVariable")
class VecParameterToVariable(_ParameterToVariable):
    """Vec parameter-to-variable. Mirrors ``ParameterToVariable<Vec>``."""

    _value_type = Vec
    param: Vec  # type: ignore[assignment]

    hit = HitSchema(
        output("to", Vec, "The name of the variable", attr="_to"),
        parameter(
            "from",
            Vec,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("RotParameterToVariable")
class RotParameterToVariable(_ParameterToVariable):
    """MRP parameter-to-variable. Mirrors ``ParameterToVariable<MRP>``."""

    _value_type = MRP
    param: MRP  # type: ignore[assignment]

    hit = HitSchema(
        output("to", MRP, "The name of the variable", attr="_to"),
        parameter(
            "from",
            MRP,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("WR2ParameterToVariable")
class WR2ParameterToVariable(_ParameterToVariable):
    """WR2 parameter-to-variable. Mirrors ``ParameterToVariable<WR2>``."""

    _value_type = WR2
    param: WR2  # type: ignore[assignment]

    hit = HitSchema(
        output("to", WR2, "The name of the variable", attr="_to"),
        parameter(
            "from",
            WR2,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("R2ParameterToVariable")
class R2ParameterToVariable(_ParameterToVariable):
    """R2 parameter-to-variable. Mirrors ``ParameterToVariable<R2>``."""

    _value_type = R2
    param: R2  # type: ignore[assignment]

    hit = HitSchema(
        output("to", R2, "The name of the variable", attr="_to"),
        parameter(
            "from",
            R2,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("SR2ParameterToVariable")
class SR2ParameterToVariable(_ParameterToVariable):
    """SR2 parameter-to-variable. Mirrors ``ParameterToVariable<SR2>``."""

    _value_type = SR2
    param: SR2  # type: ignore[assignment]

    hit = HitSchema(
        output("to", SR2, "The name of the variable", attr="_to"),
        parameter(
            "from",
            SR2,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("SSR4ParameterToVariable")
class SSR4ParameterToVariable(_ParameterToVariable):
    """SSR4 parameter-to-variable. Mirrors ``ParameterToVariable<SSR4>``."""

    _value_type = SSR4
    param: SSR4  # type: ignore[assignment]

    hit = HitSchema(
        output("to", SSR4, "The name of the variable", attr="_to"),
        parameter(
            "from",
            SSR4,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


@register_neml2_object("MillerIndexParameterToVariable")
class MillerIndexParameterToVariable(_ParameterToVariable):
    """MillerIndex parameter-to-variable. Mirrors ``ParameterToVariable<MillerIndex>``."""

    _value_type = MillerIndex
    param: MillerIndex  # type: ignore[assignment]

    hit = HitSchema(
        output("to", MillerIndex, "The name of the variable", attr="_to"),
        parameter(
            "from",
            MillerIndex,
            "The input parameter",
            attr="param",
            allow_promotion=True,
        ),
    )


__all__ = [
    "MillerIndexParameterToVariable",
    "R2ParameterToVariable",
    "RotParameterToVariable",
    "SR2ParameterToVariable",
    "SSR4ParameterToVariable",
    "ScalarParameterToVariable",
    "VecParameterToVariable",
    "WR2ParameterToVariable",
]
