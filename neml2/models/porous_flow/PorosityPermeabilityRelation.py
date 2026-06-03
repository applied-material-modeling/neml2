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

"""Python-native mirror of the C++ abstract ``PorosityPermeabilityRelation`` base.

The C++ ``PorosityPermeabilityRelation`` (see
``include/neml2/models/porous_flow/PorosityPermeabilityRelation.h``) is an
abstract intermediate between ``Model`` and concrete porosity-permeability
correlations (``PowerLawPermeability``, ``ExponentialLawPermeability``,
``KozenyCarmanPermeability``). It declares the ``porosity : Scalar`` input
and ``permeability : Scalar`` output; concrete subclasses implement the
forward operator and its differential pushforward.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical I/O names; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``PowerLawPermeability`` declare their own ``HitSchema`` rather than
inheriting from this base.
"""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import Scalar


class PorosityPermeabilityRelation(Model):
    """Define the relationship between non-dimensionalized porosity and permeability."""

    hit = HitSchema(
        input("porosity", Scalar, "The porosity"),
        output("permeability", Scalar, "The porosity-dependent permeability"),
    )

    def forward(  # type: ignore[override]
        self,
        porosity: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native porosity-permeability leaves implement the
        # forward operator (porosity -> permeability) and its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
