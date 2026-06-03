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

"""Python-native mirror of the C++ abstract ``Eigenstrain`` base.

The C++ ``Eigenstrain`` (see
``include/neml2/models/solid_mechanics/kinematics/Eigenstrain.h``) is an
abstract intermediate between ``Model`` and concrete eigenstrain leaves
(``ThermalEigenstrain``, ``VolumeChangeEigenstrain``,
``PhaseTransformationEigenstrain``). Its only contribution to the canonical
surface is the ``eigenstrain : SR2`` output declared in
``expected_options()``; concrete subclasses add their own input variables
and parameters.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical ``eigenstrain`` output name so
the documented surface is visible in one place; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``ThermalEigenstrain`` declare their own ``HitSchema`` instead of
inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, output
from ....types import SR2

if TYPE_CHECKING:
    pass


class Eigenstrain(Model):
    """Abstract base for models that compute an eigenstrain SR2 output."""

    hit = HitSchema(
        output("eigenstrain", SR2, "Eigenstrain"),
    )

    def forward(  # type: ignore[override]
        self,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native eigenstrain leaves declare their own input
        # variables (temperature, volume, phase fraction, ...) and implement
        # the forward operator (-> eigenstrain) together with its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
