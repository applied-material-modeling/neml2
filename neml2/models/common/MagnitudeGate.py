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

r"""Substitute one value for another while a reference is negligible.

Elementwise, ``y = norm(reference) < threshold ? prediction : reference``. The
reference both decides the switch and supplies the fallback, so above the
threshold this is a pass-through.

The motivating use is warm starts, which is where the measured numbers below
come from -- but nothing here knows what a predictor is, so the leaf is named
for what it does rather than for that one caller.

**Why a warm start wants it.** A predictor that extrapolates from nothing is
valuable exactly once: on the step with no previous solution to start from.
Applied at every step it is usually *worse* than reusing the converged previous
state, because that state is already close and the extrapolation walks away from
it. Measured on ``cp_coupled``, disabling the gate on the existing
``CrystalPlasticityStrainPredictor`` takes the Newton counts from
``[16, 4, 3, 3]`` to ``[16, 13, 9, 8]`` -- the first step unchanged, every later
step roughly tripled.

Three crystal-plasticity leaves already carry their own ``threshold`` option and
their own ``where``. This is that rule written once, so a new predictor gets
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


class _MagnitudeGate(Model):
    r"""$y = \lVert r\rVert < \varepsilon\ ?\ p : r$.

    *p* is the prediction, *r* the reference and $\varepsilon$ the threshold,
    all elementwise over the batch. The reference doubles as the fallback: once
    it is no longer negligible the prediction is discarded and *r* passes
    through untouched.

    For a warm start *r* is the previous converged value of the quantity *p*
    predicts, which makes "not negligible" mean "there is already a better guess
    than any extrapolation".

    Note that ``where`` evaluates **both** branches, so this protects the
    *quality* of the substituted value, not the cost of computing it. Skipping
    the work needs control flow the compiled routes do not have; if predictor
    cost ever dominates, that is a scheduling change, not a change here.
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
        below = lt(self._magnitude(reference), threshold)
        out = where(below, prediction, reference)
        if v is None:
            return out

        # Exactly one branch is live per element, so the pushforward routes the
        # incoming tangent to whichever input was selected and zeros the other.
        # The switch's own dependence on ||reference|| is a jump at the
        # threshold -- measure zero, and not differentiable there in any case --
        # so it contributes nothing, the same convention MacaulaySplit and
        # HermiteSmoothStep use for their element-wise switches.
        #
        # Predictor callers never differentiate this, but the leaf is a generic
        # selector and is named as one, so it carries the real derivative rather
        # than a stub. That is also what lets it have a ModelUnitTest, which
        # finite-differences every input.
        def prediction_action(V):
            return where(below, V, V * 0.0)

        def reference_action(V):
            return where(below, V * 0.0, V)

        return out, self.apply_chain_rule(
            v,
            self._out,
            {self._pred: prediction_action, self._ref: reference_action},
            output=out,
        )


def _gate_schema(t: type[TensorWrapper]) -> HitSchema:
    return HitSchema(
        input(
            "prediction",
            t,
            "The value substituted in while the reference is negligible.",
            attr="_pred",
        ),
        input(
            "reference",
            t,
            "The value whose norm decides the switch, and the fallback above the "
            "threshold. For a warm start this is the previous converged value of the "
            "quantity being predicted.",
            attr="_ref",
        ),
        output(
            "gated",
            t,
            "The prediction while the reference is negligible, the reference otherwise.",
            attr="_out",
        ),
        parameter(
            "threshold",
            Scalar,
            "Substitute the prediction only while the reference's norm is below this.",
            default="1e-3",
            allow_promotion=True,
        ),
    )


@register_neml2_object("ScalarMagnitudeGate")
class ScalarMagnitudeGate(_MagnitudeGate):
    r"""Substitute a `Scalar` while a reference is negligible.

    See :class:`_MagnitudeGate` for the switch and its derivative.
    """

    _type = Scalar
    hit = _gate_schema(Scalar)

    @staticmethod
    def _magnitude(x: TensorWrapper) -> Scalar:
        # `norm` has no Scalar overload -- for a scalar the norm IS |x|.
        assert isinstance(x, Scalar)
        return abs(x)


@register_neml2_object("SR2MagnitudeGate")
class SR2MagnitudeGate(_MagnitudeGate):
    r"""Substitute an `SR2` while a reference is negligible.

    See :class:`_MagnitudeGate` for the switch and its derivative.
    """

    _type = SR2
    hit = _gate_schema(SR2)

    @staticmethod
    def _magnitude(x: TensorWrapper) -> Scalar:
        assert isinstance(x, SR2)
        return norm(x)


__all__ = ["ScalarMagnitudeGate", "SR2MagnitudeGate"]
