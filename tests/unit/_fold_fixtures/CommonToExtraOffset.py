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

r"""Test-only model adding a per-common offset to a per-(common, extra) field.

``y_{g,i} = e_{g,i} + c_g`` -- the per-common input ``c`` (sub-batch ``(common,)``) is
broadcast over the trailing extra axis of ``e`` (sub-batch ``(common, extra)``). Its only
purpose is to make a block-extra residual depend on a block common-only unknown, so the
equation-system assembler exercises the ROW-extra tangent fold
(``_convert_tangent_to_paired_block_rowextra``): row carries ``(common, extra)`` while the
coupled column is common-only. ``dy/dc`` is constant over the extra axis.
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleAction, ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import Scalar


@register_neml2_object("CommonToExtraOffset")
class CommonToExtraOffset(Model):
    r"""Broadcast-add a per-common offset onto a per-(common, extra) field (test fixture)."""

    hit = HitSchema(
        input("e", Scalar, "Per-(common, extra) field (trailing two sub-batch axes)"),
        input("c", Scalar, "Per-common offset (single sub-batch axis), broadcast over extra"),
        output("y", Scalar, "``y_{g,i} = e_{g,i} + c_g``"),
    )

    def forward(  # type: ignore[override]
        self,
        e: Scalar,
        c: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        y = e + c.sub_batch.unsqueeze(-1)

        if v is None:
            return y

        actions: dict[str, ChainRuleAction] = {
            "e": lambda V: V,
            "c": lambda V: V.sub_batch.unsqueeze(-1),
        }
        return y, self.apply_chain_rule(v, "y", actions, output=y)
