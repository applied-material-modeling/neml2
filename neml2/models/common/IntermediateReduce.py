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

"""Python-native mirrors of C++ ``common/IntermediateMean.h`` / ``IntermediateSum.h``."""

from __future__ import annotations

from typing import ClassVar

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output
from ...types import (
    SR2,
    Scalar,
    TensorWrapper,
    mean,
    sum,
)
from ..chain_rule import ChainRuleDict
from ..model import Model

# Linear sub-batch reductions only: the pushforward of a linear reduction is the
# same reduction applied to the tangent. A non-linear op (max/min/prod/...) would
# need a value-dependent action and a matching sub-batch primitive in
# ``neml2.types`` first, so it does not belong in this table.
_OPS = {"mean": mean, "sum": sum}


class _IntermediateReduce(Model):
    """Reduce one sub-batch axis of a typed variable, dropping ``sub_batch_ndim``
    by one.

    Concrete leaves set ``_op`` (``"mean"`` or ``"sum"``) and declare their own
    ``from``/``to`` wrapper type. The reduced axis is selected by ``dim``
    (relative to the sub-batch region); it defaults to the trailing axis.
    """

    _op: ClassVar[str]
    _dim: int

    def forward(  # type: ignore[override]
        self,
        x: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        if x.sub_batch_ndim < 1:
            raise ValueError(
                f"{type(self).__name__} requires at least one sub-batch dim; "
                f"got sub_batch_ndim={x.sub_batch_ndim}"
            )
        op = _OPS[self._op]
        out = op(x.sub_batch, self._dim)
        if v is None:
            return out

        def action(V: TensorWrapper) -> TensorWrapper:
            return op(V.sub_batch, self._dim)

        return out, self.apply_chain_rule(v, "to", {"from": action}, output=out)


@register_neml2_object("SR2IntermediateMean")
class SR2IntermediateMean(_IntermediateReduce):
    """``to = mean(from, dim=sub_batch_axis)``; reduces sub_batch_ndim by 1.

    Mirrors C++ ``IntermediateMean<SR2>``. Used in CP to average per-crystal
    cauchy stress into a mean over crystals (taylor-style homogenisation).
    """

    _op = "mean"

    hit = HitSchema(
        input("from", SR2, "Input tensor to average"),
        output("to", SR2, "Averaged tensor"),
        option("dim", int, "Sub-batch dimension to average", default=-1, attr="_dim"),
    )


@register_neml2_object("ScalarIntermediateMean")
class ScalarIntermediateMean(_IntermediateReduce):
    """Scalar variant of :class:`SR2IntermediateMean`."""

    _op = "mean"

    hit = HitSchema(
        input("from", Scalar, "Input scalar to average"),
        output("to", Scalar, "Averaged scalar"),
        option("dim", int, "Sub-batch dimension to average", default=-1, attr="_dim"),
    )


@register_neml2_object("SR2IntermediateSum")
class SR2IntermediateSum(_IntermediateReduce):
    """``to = sum(from, dim=sub_batch_axis)``; reduces sub_batch_ndim by 1.

    Mirrors C++ ``IntermediateSum<SR2>``.
    """

    _op = "sum"

    hit = HitSchema(
        input("from", SR2, "Variable to reduce"),
        output("to", SR2, "The reduced variable"),
        option("dim", int, "Sub-batch dimension to sum", default=-1, attr="_dim"),
    )


@register_neml2_object("ScalarIntermediateSum")
class ScalarIntermediateSum(_IntermediateReduce):
    """Scalar variant of :class:`SR2IntermediateSum`."""

    _op = "sum"

    hit = HitSchema(
        input("from", Scalar, "Variable to reduce"),
        output("to", Scalar, "The reduced variable"),
        option("dim", int, "Sub-batch dimension to sum", default=-1, attr="_dim"),
    )


__all__ = [
    "SR2IntermediateMean",
    "ScalarIntermediateMean",
    "SR2IntermediateSum",
    "ScalarIntermediateSum",
]
