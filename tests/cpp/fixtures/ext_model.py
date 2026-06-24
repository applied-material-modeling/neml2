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

"""External (out-of-package) NEML2 model for the cpp-eager ``--load`` test.

This file lives OUTSIDE the installed ``neml2`` package, so its
``@register_neml2_object`` type is invisible to the factory unless explicitly
imported. ``neml2::eager::load_model(input, "model", load={this file})`` must run
the ``--load`` hook to register ``ExtScaleStress`` before building the model;
without it, loading the companion ``ext_model.i`` fails with an unknown-type
error. Uses absolute ``neml2.*`` imports (it is not part of the package tree).
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import SR2


@register_neml2_object("ExtScaleStress")
class ExtScaleStress(Model):
    """``out_stress = 2 * in_stress`` -- a trivial external model.

    The factor-of-two scaling makes the output observably distinct from the
    input, so the C++ test can confirm the external model actually ran (not just
    loaded).
    """

    hit = HitSchema(
        input("in_stress", SR2, "Input stress"),
        output("out_stress", SR2, "Output stress (= 2 * input stress)"),
    )

    def forward(  # type: ignore[override]
        self,
        in_stress: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        out = in_stress * 2.0
        if v is None:
            return out
        return out, self.apply_chain_rule(
            v, "out_stress", {"in_stress": lambda V: V * 2.0}, output=out
        )
