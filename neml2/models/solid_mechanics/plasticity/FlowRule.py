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

"""Python-native mirror of the C++ abstract ``FlowRule`` base.

The C++ ``FlowRule`` (see
``include/neml2/models/solid_mechanics/plasticity/FlowRule.h``) is an abstract
intermediate that declares a ``flow_rate : Scalar`` input -- the consistency
parameter from the KKT conditions -- which concrete flow rules consume to
produce internal-variable rates (associative plastic flow, associative
isotropic/kinematic plastic hardening, etc.).

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds only the canonical ``flow_rate`` input; concrete
native subclasses declare their own ``HitSchema`` (native models are flat, no
schema inheritance) and implement the forward mapping. ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, input
from ....types import Scalar

if TYPE_CHECKING:
    pass


class FlowRule(Model):
    """Map the flow rate (i.e., the consistency parameter in the KKT
    conditions) to the rate of internal variables.
    """

    hit = HitSchema(
        input("flow_rate", Scalar, "Flow rate"),
    )

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native flow rules consume ``flow_rate`` and emit
        # the rates of one or more internal variables, with their own typed
        # outputs declared in their HitSchema.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
