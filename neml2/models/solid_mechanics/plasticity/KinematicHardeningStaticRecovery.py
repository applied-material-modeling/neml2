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

"""Python-native mirror of the C++ abstract ``KinematicHardeningStaticRecovery`` base.

The C++ ``KinematicHardeningStaticRecovery`` (see
``include/neml2/models/solid_mechanics/plasticity/KinematicHardeningStaticRecovery.h``)
is an abstract intermediate that declares the ``back_stress : SR2`` input and
the matching ``back_stress_rate : SR2`` output (produced by
``rate_name(_X.name())`` with the default empty rate prefix and ``"_rate"``
suffix). Concrete children -- e.g. ``PowerLawKinematicHardeningStaticRecovery``
-- supply the actual recovery law in ``set_value``.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. The class scaffolds the canonical input/output names that concrete
native kinematic static recovery leaves reuse; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``PowerLawKinematicHardeningStaticRecovery`` declare their own
``HitSchema`` instead of inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....schema import HitSchema, input, output
from ....types import SR2
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    pass


class KinematicHardeningStaticRecovery(Model):
    """This object defines kinematic hardening static recovery on a backstress term."""

    hit = HitSchema(
        input("back_stress", SR2, "Back stress"),
        output("back_stress_rate", SR2, "Rate of back stress"),
    )

    def forward(  # type: ignore[override]
        self,
        back_stress: SR2,
        *promoted_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native kinematic-hardening static-recovery leaves
        # implement the forward operator (X -> X_dot) and its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
