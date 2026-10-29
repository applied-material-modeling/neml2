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

r"""Test-only model reducing a per-(common, extra) field over its extra axis.

``y_g = sum_i e_{g,i}`` -- the trailing (extra) sub-batch axis of ``e`` (sub-batch
``(common, extra)``) is summed out, leaving a per-common field. Its purpose is to make a
DENSE, per-common residual depend on a BLOCK ``(common, extra)`` unknown, so the
equation-system assembler exercises the dense-row-against-block-column-WITH-extra tangent
fold in ``_convert_tangent_to_block`` (the ``col_structure == "block"`` branch that must
fold the column extra axis into column storage). ``dy_g/de_{g,i} = 1`` for every ``i`` --
a dense coupling across the extra axis.
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleAction, ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import Scalar, sum


@register_neml2_object("ReduceExtraAxis")
class ReduceExtraAxis(Model):
    r"""Sum a per-(common, extra) field over its extra axis (test fixture)."""

    hit = HitSchema(
        input("e", Scalar, "Per-(common, extra) field (two trailing sub-batch axes)"),
        output("y", Scalar, "``y_g = sum_i e_{g,i}`` (per-common)"),
    )

    def forward(  # type: ignore[override]
        self,
        e: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        y = sum(e.sub_batch, -1)

        if v is None:
            return y

        actions: dict[str, ChainRuleAction] = {
            "e": lambda V: sum(V.sub_batch, -1),
        }
        return y, self.apply_chain_rule(v, "y", actions, output=y)
