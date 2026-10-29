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

r"""SR2-valued counterpart of :mod:`ReduceExtraAxis` for the non-scalar column fold.

``y_g = sum_i e_{g,i}`` with ``e`` an SR2 (base 6) field on sub-batch ``(common, extra)``.
It makes a DENSE, per-common residual reduce a BLOCK ``(common, extra)`` unknown whose
per-site state is non-scalar (``col_var_base = 6``), so the ``col_structure == "block"``
col-extra fold must flatten ``(extra, col_base)`` in that order -- a Scalar column
(``col_var_base = 1``) cannot distinguish the two orderings. ``dy_g/de_{g,i}`` is the
6x6 identity for every ``i`` -- a dense coupling across the extra axis.
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleAction, ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import SR2, sum


@register_neml2_object("ReduceExtraAxisSR2")
class ReduceExtraAxisSR2(Model):
    r"""Sum an SR2 per-(common, extra) field over its extra axis (test fixture)."""

    hit = HitSchema(
        input("e", SR2, "Per-(common, extra) SR2 field (two trailing sub-batch axes)"),
        output("y", SR2, "``y_g = sum_i e_{g,i}`` (per-common SR2)"),
    )

    def forward(  # type: ignore[override]
        self,
        e: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        y = sum(e.sub_batch, -1)

        if v is None:
            return y

        actions: dict[str, ChainRuleAction] = {
            "e": lambda V: sum(V.sub_batch, -1),
        }
        return y, self.apply_chain_rule(v, "y", actions, output=y)
