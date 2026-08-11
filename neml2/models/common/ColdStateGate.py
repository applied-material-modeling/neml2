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

r"""Apply a prediction only from a cold state.

A predictor that extrapolates from nothing is valuable exactly once -- on the
step where there is no previous solution to start from. Applied at every step it
is usually *worse* than simply reusing the converged previous state, because the
previous state is already close and the extrapolation walks away from it.

Measured on ``cp_coupled``: disabling the gate on the existing
``CrystalPlasticityStrainPredictor`` takes the Newton counts from
``[16, 4, 3, 3]`` to ``[16, 13, 9, 8]`` -- the cold step is unchanged and every
warm step roughly triples.

Three crystal-plasticity leaves already carry their own ``threshold`` option and
their own ``where``. This is the same rule written once, so a new predictor gets
gating by composition instead of by a fourth copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import SR2, Scalar, TensorWrapper, abs, lt, norm, where  # noqa: A004
from ..model import Model

if TYPE_CHECKING:
    from ..chain_rule import ChainRuleDict


class _ColdStateGate(Model):
    r"""$y = \lVert r\rVert < \varepsilon\ ?\ p : r$.

    *p* is the prediction, *r* the reference (normally the previous converged
    value of the same quantity) and $\varepsilon$ the threshold. The reference
    doubles as the fallback: when the state is not cold, the prediction is
    discarded and the previous value passes through unchanged.

    Note that ``where`` evaluates **both** branches, so the gate protects the
    *quality* of the guess, not the cost of computing it. Skipping the work
    needs control flow the compiled routes do not have; if predictor cost ever
    dominates, that is a scheduling change, not a change here.
    """

    _type: type[TensorWrapper]

    @staticmethod
    def _magnitude(x: TensorWrapper) -> Scalar:
        """Scalar magnitude of the reference. Overridden per wrapper type."""
        raise NotImplementedError

    _pred: str
    _ref: str
    _out: str
    threshold: Scalar

    def forward(  # type: ignore[override]
        self,
        prediction: TensorWrapper,
        reference: TensorWrapper,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        threshold = self._get_param("threshold", promoted_params, Scalar)
        cold = lt(self._magnitude(reference), threshold)
        out = where(cold, prediction, reference)
        if v is None:
            return out

        # A predictor is a one-shot warm start and is never differentiated, so
        # the tangent is structurally zero -- the same trivial pass-through the
        # other predictor leaves provide so chain-rule callers do not error.
        def trivial(V):
            return V * 0.0

        return out, self.apply_chain_rule(
            v,
            self._out,
            {self._pred: trivial, self._ref: trivial},
            output=out,
        )


def _gate_schema(t: type[TensorWrapper]) -> HitSchema:
    return HitSchema(
        input(
            "prediction", t, "The predicted value, used only when the state is cold", attr="_pred"
        ),
        input(
            "reference",
            t,
            "The value whose norm decides coldness -- normally the previous converged "
            "value of the gated quantity. Doubles as the fallback when the gate is shut.",
            attr="_ref",
        ),
        output("gated", t, "The prediction when cold, the reference otherwise", attr="_out"),
        parameter(
            "threshold",
            Scalar,
            "Apply the prediction only while the reference's norm is below this.",
            default="1e-3",
            allow_promotion=True,
        ),
    )


@register_neml2_object("ScalarColdStateGate")
class ScalarColdStateGate(_ColdStateGate):
    r"""Apply a `Scalar` prediction only from a cold state. See :class:`_ColdStateGate`."""

    _type = Scalar
    hit = _gate_schema(Scalar)

    @staticmethod
    def _magnitude(x: TensorWrapper) -> Scalar:
        # `norm` has no Scalar overload -- for a scalar the norm IS |x|.
        assert isinstance(x, Scalar)
        return abs(x)


@register_neml2_object("SR2ColdStateGate")
class SR2ColdStateGate(_ColdStateGate):
    r"""Apply an `SR2` prediction only from a cold state. See :class:`_ColdStateGate`."""

    _type = SR2
    hit = _gate_schema(SR2)

    @staticmethod
    def _magnitude(x: TensorWrapper) -> Scalar:
        assert isinstance(x, SR2)
        return norm(x)


__all__ = ["ScalarColdStateGate", "SR2ColdStateGate"]
