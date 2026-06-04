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

"""Python-native mirrors of C++ ``common/IntermediateDiff.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import (
    SR2,
    diff,
)


@register_neml2_object("SR2IntermediateDiff")
class SR2IntermediateDiff(Model):
    """Finite difference along an intermediate dimension.

    Mirrors C++ ``IntermediateDiff<SR2>``: apply the ``n``-th order finite
    difference along the selected sub-batch axis. The selected axis length
    shrinks by ``n``; ``sub_batch_ndim`` is preserved (this is a per-axis
    differencing operator, not a reduction).
    """

    # ``intmd_diff`` couples every output site along the chosen sub-batch axis
    # to the input sites it differences -- declare dense per D-049.
    hit = HitSchema(
        input("from", SR2, "Variable to reduce"),
        output("to", SR2, "The reduced variable"),
        option(
            "dim",
            int,
            "Intermediate dimension to take the finite difference",
            attr="_dim",
        ),
        option("n", int, "Order of the differentiation", default=1, attr="_n"),
    )
    _dim: int
    _n: int
    list_deriv = {("to", "from"): "dense"}

    def forward(  # type: ignore[override]
        self,
        x: SR2,
        v: ChainRuleDict | None = None,
    ):
        d = self._dim
        n = self._n
        out = diff(x.sub_batch, n, d)
        if v is None:
            return out

        # ``intmd_diff`` is linear in its argument, so the pushforward is the
        # same operator applied to the tangent. The leading-K convention
        # prepends K to the dynamic batch, leaving sub_batch_ndim unchanged --
        # the sub-batch ``dim`` index is therefore identical for input and
        # tangent.
        def action(V: SR2) -> SR2:
            return diff(V.sub_batch, n, d)

        return out, self.apply_chain_rule(v, "to", {"from": action}, output=out)


__all__ = ["SR2IntermediateDiff"]
