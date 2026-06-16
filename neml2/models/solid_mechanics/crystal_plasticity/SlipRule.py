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

"""Python-native mirror of the C++ abstract ``SlipRule`` base.

The C++ ``SlipRule`` (see
``include/neml2/models/solid_mechanics/crystal_plasticity/SlipRule.h``) is an
abstract intermediate between ``Model`` and concrete crystal-plasticity slip
rules (``PowerLawSlipRule`` and future variants). It declares the canonical
``resolved_shears`` / ``slip_strengths`` inputs and the ``slip_rates`` output,
leaving the actual flow law to subclasses.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. The class scaffolds the canonical input/output names so any future
shared logic has a home; ``forward`` raises ``NotImplementedError`` so any
accidental direct use surfaces immediately. Native models are flat (no schema
inheritance), so existing concrete leaves like ``PowerLawSlipRule`` declare
their own ``HitSchema`` instead of inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....schema import HitSchema, input, output
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    pass


class SlipRule(Model):
    """Parent class of slip rules, mapping from resolved shear and internal state to slip rates."""

    hit = HitSchema(
        input("resolved_shears", Scalar, "Name of the resolved shear tensor"),
        input("slip_strengths", Scalar, "Name of the tensor containing the slip system strengths"),
        output("slip_rates", Scalar, "Name of the slip rate tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        rss: Scalar,
        tau: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native slip rules implement the flow law
        # (resolved shear, slip strength) -> slip rate and its pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
