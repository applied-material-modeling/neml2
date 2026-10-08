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

r"""Test-only model adding a global scalar to a per-(common, extra) field.

``y_{g,i} = e_{g,i} + s`` -- the single scalar ``s`` (no sub-batch) is broadcast over
BOTH sub-batch axes of ``e`` (sub-batch ``(common, extra)``). Its purpose is to make a
block-extra residual depend on a sub-batch-trivial unknown that lands in a DENSE column
group, so the equation-system assembler exercises the block-row-with-extra-against-a-
dense-column tangent fold in ``_convert_tangent_to_block`` (the ``row_structure ==
"block"`` extra-into-row-storage branch). ``dy/ds`` is constant over both sub-batch axes.
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleAction, ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import Scalar


@register_neml2_object("AddScalarToExtra")
class AddScalarToExtra(Model):
    r"""Broadcast-add a global scalar onto a per-(common, extra) field (test fixture)."""

    hit = HitSchema(
        input("e", Scalar, "Per-(common, extra) field (two trailing sub-batch axes)"),
        input("s", Scalar, "Global scalar offset (no sub-batch), broadcast over both axes"),
        output("y", Scalar, "``y_{g,i} = e_{g,i} + s``"),
    )

    def forward(  # type: ignore[override]
        self,
        e: Scalar,
        s: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        s_broadcast = s.sub_batch.unsqueeze(-1).sub_batch.unsqueeze(-1)
        y = e + s_broadcast

        if v is None:
            return y

        actions: dict[str, ChainRuleAction] = {
            "e": lambda V: V,
            "s": lambda V: V.sub_batch.unsqueeze(-1).sub_batch.unsqueeze(-1),
        }
        return y, self.apply_chain_rule(v, "y", actions, output=y)
