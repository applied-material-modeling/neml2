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

"""Python-native mirrors of C++ ``common/IntermediateSum.h``."""

from __future__ import annotations

from typing import cast

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import (
    SR2,
    sub_batch_sum,
)


@register_native("SR2IntermediateSum")
class SR2IntermediateSum(Model):
    """Sum an intermediate dimension.

    Mirrors C++ ``IntermediateSum<SR2>``: reduces the trailing sub-batch axis
    by summing along it, dropping ``sub_batch_ndim`` by 1.
    """

    # The output couples sites along the sub-batch axis (every output depends
    # on every input site) -- declare dense per D-049.
    hit = HitSchema(
        input("from", SR2, "Variable to reduce"),
        output("to", SR2, "The reduced variable"),
    )
    list_deriv = {("to", "from"): "dense"}

    def forward(  # type: ignore[override]
        self,
        x: SR2,
        v: ChainRuleDict | None = None,
    ):
        out = cast(SR2, sub_batch_sum(x.sub_batch.retag(1), -1))
        if v is None:
            return out

        def action(V: SR2) -> SR2:
            return cast(SR2, sub_batch_sum(V, -1))

        return out, self.apply_chain_rule(v, "to", {"from": action}, output=out)


__all__ = ["SR2IntermediateSum"]
