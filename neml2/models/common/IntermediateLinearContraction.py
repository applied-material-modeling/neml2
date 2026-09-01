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

"""Contract an operator against a field over the last intermediate dimension."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar, sum
from ...types.functions import fullify
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("IntermediateLinearContraction")
class IntermediateLinearContraction(Model):
    """Contract an operator with a field over the last intermediate dimension.

    $$
    \\text{out}_i = \\sum_j \\text{operator}_{ij}\\, \\text{field}_j
    $$

    ``operator`` is a two-dimensional intermediate quantity (sub_batch_ndim=2,
    rows x cols) and ``field`` is a one-dimensional field (sub_batch_ndim=1)
    whose length matches the column axis of ``operator``. The output ``out`` is
    the field on the operator's row axis. Linear in both operands, so the
    pushforward is the same contraction applied to each tangent. Reusable
    beyond any one physics.
    """

    hit = HitSchema(
        input(
            "operator",
            Scalar,
            "Linear operator whose trailing intermediate axis is contracted (rows x cols).",
            attr="_operator_name",
        ),
        input(
            "field",
            Scalar,
            "Field contracted against the operator's column axis.",
            attr="_field_name",
        ),
        output(
            "out",
            Scalar,
            "Contraction result over the operator's row axis.",
            attr="_out_name",
        ),
    )

    _operator_name: str
    _field_name: str
    _out_name: str

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        operator, field = inputs
        # Contract the shared column axis; field broadcasts over the operator rows.
        out = sum((operator * field.sub_batch.unsqueeze(0)).sub_batch, -1)
        if v is None:
            return out

        # fullify before contracting: the summed column axis is K-paired, so the
        # broadcast eye must be materialised first (as in the other FV leaves).
        def operator_action(V_in: Scalar) -> Scalar:
            V_full = fullify(V_in)
            return sum((V_full * field.sub_batch.unsqueeze(0)).sub_batch, -1)

        def field_action(V_in: Scalar) -> Scalar:
            V_full = fullify(V_in)
            return sum((operator * V_full.sub_batch.unsqueeze(0)).sub_batch, -1)

        actions = {self._operator_name: operator_action, self._field_name: field_action}
        return out, self.apply_chain_rule(v, self._out_name, actions, output=out)


__all__ = ["IntermediateLinearContraction"]
