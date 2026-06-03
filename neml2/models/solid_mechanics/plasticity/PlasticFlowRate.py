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

"""Python-native mirror of the C++ abstract ``PlasticFlowRate`` base.

The C++ ``PlasticFlowRate`` (see
``include/neml2/models/solid_mechanics/plasticity/PlasticFlowRate.h``) is the
abstract intermediate that fixes the canonical I/O surface of any yield-
function -> flow-rate map (e.g. ``PerzynaPlasticFlowRate``,
``KocksMeckingFlowSwitch``, ...).

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical input/output names; native models
are flat (no schema inheritance), so concrete leaves like
``PerzynaPlasticFlowRate`` declare their own ``HitSchema`` instead of
inheriting from this base. ``forward`` raises ``NotImplementedError`` so any
accidental direct use surfaces immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import Scalar

if TYPE_CHECKING:
    pass


class PlasticFlowRate(Model):
    """Map yield function to plastic flow rate (the consistency parameter in
    the KKT conditions).
    """

    hit = HitSchema(
        input("yield_function", Scalar, "Yield function"),
        output("flow_rate", Scalar, "Flow rate"),
    )

    def forward(  # type: ignore[override]
        self,
        yield_function: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native plastic-flow-rate leaves implement the
        # forward operator (yield function -> flow rate) and its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
