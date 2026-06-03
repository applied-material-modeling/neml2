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

"""Python-native mirror of the C++ abstract ``KinematicHardening`` base.

The C++ ``KinematicHardening`` (see
``include/neml2/models/solid_mechanics/plasticity/KinematicHardening.h``) is
the abstract intermediate that fixes the canonical I/O surface of any
kinematic-hardening rule: kinematic plastic strain in, back stress out.
Concrete laws (Linear, FredrickArmstrong, Chaboche, ...) supply the
constitutive form.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical input/output names; native models
are flat (no schema inheritance), so concrete leaves like
``LinearKinematicHardening`` declare their own ``HitSchema`` instead of
inheriting from this base. ``forward`` raises ``NotImplementedError`` so any
accidental direct use surfaces immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2

if TYPE_CHECKING:
    pass


class KinematicHardening(Model):
    """Map kinematic plastic strain to back stress."""

    hit = HitSchema(
        input("kinematic_plastic_strain", SR2, "Kinematic plastic strain"),
        output("back_stress", SR2, "Back stress"),
    )

    def forward(  # type: ignore[override]
        self,
        kinematic_plastic_strain: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native kinematic-hardening leaves implement the
        # forward operator (kinematic plastic strain -> back stress) and its
        # differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
