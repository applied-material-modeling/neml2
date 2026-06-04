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

"""Python-native mirror of the C++ abstract ``IsotropicHardeningStaticRecovery`` base.

The C++ ``IsotropicHardeningStaticRecovery`` (see
``include/neml2/models/solid_mechanics/plasticity/IsotropicHardeningStaticRecovery.h``)
is an abstract intermediate that declares the ``isotropic_hardening : Scalar``
input and the matching ``isotropic_hardening_rate : Scalar`` output (produced
by ``rate_name(_h.name())`` with the default empty rate prefix and ``"_rate"``
suffix). Concrete children -- e.g. ``PowerLawIsotropicHardeningStaticRecovery``
-- supply the actual recovery law in ``set_value``.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. The class scaffolds the canonical input/output names that concrete
native isotropic static recovery leaves reuse; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``PowerLawIsotropicHardeningStaticRecovery`` declare their own
``HitSchema`` instead of inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import Scalar

if TYPE_CHECKING:
    pass


class IsotropicHardeningStaticRecovery(Model):
    """Static recovery for isotropic hardening."""

    hit = HitSchema(
        input("isotropic_hardening", Scalar, "Isotropic hardening variable"),
        output("isotropic_hardening_rate", Scalar, "Rate of isotropic hardening"),
    )

    def forward(  # type: ignore[override]
        self,
        isotropic_hardening: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native isotropic-hardening static-recovery leaves
        # implement the forward operator (h -> h_dot) and its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
