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

"""Python-native mirror of the C++ abstract ``SlipStrengthMap`` base.

The C++ ``SlipStrengthMap`` (see
``include/neml2/models/solid_mechanics/crystal_plasticity/SlipStrengthMap.h``)
is an abstract intermediate between ``Model`` and concrete maps from internal
variables to per-slip-system strengths (``SingleSlipStrengthMap``,
``DislocationObstacleStrengthMap``, future variants). The C++ base only
declares the canonical ``slip_strengths`` output; the input set varies by
subclass.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. The class scaffolds the canonical output name; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``SingleSlipStrengthMap`` and ``DislocationObstacleStrengthMap`` declare
their own ``HitSchema`` (with their own input set) instead of inheriting from
this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....schema import HitSchema, output
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    pass


class SlipStrengthMap(Model):
    """Parent class of maps between internal variables and the slip system strengths."""

    hit = HitSchema(
        output("slip_strengths", Scalar, "Name of the slip system strengths"),
    )

    def forward(  # type: ignore[override]
        self,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native slip-strength maps implement the internal
        # variable -> per-slip strength mapping and its pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
