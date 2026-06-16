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

"""Python-native mirror of the C++ abstract ``StrainEnergyDensity`` base.

The C++ ``StrainEnergyDensity`` (see
``include/neml2/models/phase_field_fracture/StrainEnergyDensity.h``) is an
abstract intermediate between ``Model`` and concrete strain-energy-density
models (``LinearIsotropicStrainEnergyDensity``, ...). It fixes the canonical
I/O contract for any model that splits the elastic strain energy density into
an active part (driving fracture) and an inactive part: ``strain : SR2`` in,
``active_strain_energy_density : Scalar`` and
``inactive_strain_energy_density : Scalar`` out. Concrete forms supply the
actual decomposition and its differential pushforward.

Because the C++ class is not registered with ``register_NEML2_object``, this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. Native models are flat (no schema inheritance), so concrete leaves
declare their own ``HitSchema`` instead of inheriting from this base.
``forward`` raises ``NotImplementedError`` so any accidental direct use
surfaces immediately.
"""

from __future__ import annotations

from ...schema import HitSchema, input, output
from ...types import SR2, Scalar
from ..chain_rule import ChainRuleDict
from ..model import Model


class StrainEnergyDensity(Model):
    """Parent class of strain energy density models used in phase-field fracture."""

    hit = HitSchema(
        input("strain", SR2, "Elastic strain"),
        output("active_strain_energy_density", Scalar, "Active part of the strain energy density"),
        output(
            "inactive_strain_energy_density",
            Scalar,
            "Inactive part of the strain energy density",
        ),
    )

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native strain-energy-density leaves implement the
        # forward operator (strain -> (active, inactive) energy split) and its
        # differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
