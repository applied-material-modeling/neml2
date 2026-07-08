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

"""Python-native mirrors of C++ ``common/LinearCombination.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameter, parameters, var_inputs
from ...types import (
    SR2,
    Scalar,
    TensorWrapper,
)
from ..chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ..model import Model


class _LinearCombination(Model):
    """$to = Σ wᵢ · fromᵢ + b$ (Einstein summation over the from-list).

    Mirrors the C++ ``LinearCombination<T>``. ``offset`` is the constant $b$
    added to the final summation (default ``0``, i.e. no offset). Both weights
    and offset are tracked as ``Scalar`` parameters with ``allow_promotion=True``,
    so each can independently resolve as a literal HIT value, a ``[Tensors]``
    cross-reference, or a runtime-promoted input (modes 1 / 2 / 3 / 4 —
    the same scheme as any other ``declare_typed_parameter`` call). This is
    intentionally less configurable than the C++ ``offset_as_parameter`` /
    ``weight_as_parameter`` knobs: every coefficient is a parameter by default
    on the native side.
    """

    # Linear in every input ⇒ all second derivatives are zero. Opting into
    # the second-order chain rule lets this leaf sit inside a Normality
    # wrap (e.g. the radial-return demo's ``trial_overstress`` feeding
    # ``trial_normality``); the v2 path just propagates incoming Hessian-applied
    # tangents through ``actions_1`` with no g'' contribution.
    SUPPORTS_SECOND_ORDER = True

    _value_type: type[TensorWrapper]

    _from_vars: list[str]
    _to: str
    weight: list[str]
    offset: Scalar

    # Schema fields shared by every ``T`` instantiation. Splat into each
    # subclass's ``HitSchema`` (HitField is a frozen dataclass, so sharing
    # instances across schemas is safe).
    _COMMON_PARAMETERS = (
        parameters(
            "weights",
            Scalar,
            "Per-input weight parameters. When the list is length 1, the single "
            "weight is broadcast across every from-variable.",
            default=["1"],
            attr="weight",
            allow_promotion=True,
        ),
        parameter(
            "offset",
            Scalar,
            "The constant coefficient added to the final summation",
            default="0",
            attr="offset",
            allow_promotion=True,
        ),
    )

    def __post_init__(self) -> None:
        # ``weights`` of length 1 broadcasts across all from-vars (mirrors the
        # C++ default); otherwise it must match the from-var count one-for-one.
        # __post_init__ runs after ``parameters`` populates ``self.weight``
        # (pass 1) but before the deferred parameter declarations (pass 2), so
        # we can validate the count here without touching parameter state.
        if len(self.weight) != 1 and len(self.weight) != len(self._from_vars):
            raise ValueError(
                f"{type(self).__name__}: weights must have length 1 or "
                f"{len(self._from_vars)}, got {len(self.weight)}: "
                f"{self.weight!r}"
            )

    def forward(  # type: ignore[override]
        self,
        *args: TensorWrapper,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Split positional inputs: the leading structural inputs (one per
        # from-var) followed by the *promoted_params pack that holds any
        # mode-3/4-promoted parameters.
        n_from = len(self._from_vars)
        inputs, promoted_params = args[:n_from], args[n_from:]
        if len(inputs) != n_from:
            raise ValueError(f"{type(self).__name__} expected {n_from} inputs, got {len(inputs)}")
        weights = self._get_param_list("weight", promoted_params, Scalar)
        if len(weights) == 1:
            weights = weights * n_from
        offset = self._get_param("offset", promoted_params, Scalar)

        result = weights[0] * inputs[0]
        for w, x in zip(weights[1:], inputs[1:], strict=True):
            result = result + w * x
        # Add the constant Scalar offset. ``WrapperT + Scalar`` is defined
        # on every value wrapper (Scalar/Vec/SR2/WR2/MRP/R2/SSR4) and routes
        # through ``types._base.align_scalar_base`` to broadcast the Scalar
        # against the trailing base dims. Sub-batch alignment is automatic.
        out = result + offset
        if v is None:
            return out
        actions_1 = {
            fv: (lambda V, w=w: w * V) for fv, w in zip(self._from_vars, weights, strict=True)
        }
        # Linear ⇒ no actions_2 needed; propagate_tangents handles the v/v2/vh
        # dispatch and the right-length return tuple.
        return out, *self.propagate_tangents(v, self._to, actions_1, output=out, v2=v2, vh=vh)


@register_neml2_object("ScalarLinearCombination")
class ScalarLinearCombination(_LinearCombination):
    r"""Calculate linear combination of multiple Scalar tensors as
    $u = w_i v_i + b$ (Einstein summation assumed), where $w_i$ are
    the weights, and $v_i$ are the variables to be summed. $b$ is a
    constant offset.
    """

    _value_type = Scalar

    hit = HitSchema(
        var_inputs("from", Scalar, "Scalar tensors to be summed", attr="_from_vars"),
        output("to", Scalar, "The sum", attr="_to"),
        *_LinearCombination._COMMON_PARAMETERS,
    )


@register_neml2_object("SR2LinearCombination")
class SR2LinearCombination(_LinearCombination):
    r"""Calculate linear combination of multiple SR2 tensors as
    $u = w_i v_i + b$ (Einstein summation assumed), where $w_i$ are
    the weights, and $v_i$ are the variables to be summed. $b$ is a
    constant offset.
    """

    _value_type = SR2

    hit = HitSchema(
        var_inputs("from", SR2, "SR2 tensors to be summed", attr="_from_vars"),
        output("to", SR2, "The sum", attr="_to"),
        *_LinearCombination._COMMON_PARAMETERS,
    )


__all__ = ["ScalarLinearCombination", "SR2LinearCombination"]
