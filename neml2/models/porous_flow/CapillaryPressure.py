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

"""Python-native mirror of the C++ abstract ``CapillaryPressure`` base.

The C++ ``CapillaryPressure`` (see
``include/neml2/models/porous_flow/CapillaryPressure.h``) is an abstract
intermediate between ``Model`` and concrete capillary-pressure correlations
(``BrooksCoreyCapillaryPressure``, ``VanGenuchtenCapillaryPressure``). It
declares the ``effective_saturation : Scalar`` input and
``capillary_pressure : Scalar`` output, along with the optional logarithmic
extension knobs ``log_extension`` / ``transition_saturation``; concrete
subclasses implement ``calculate_pressure`` on the base branch.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. The class scaffolds the canonical I/O names and option surface;
``forward`` raises ``NotImplementedError`` so any accidental direct use
surfaces immediately. Native models are flat (no schema inheritance), so
existing concrete leaves like ``BrooksCoreyCapillaryPressure`` and
``VanGenuchtenCapillaryPressure`` declare their own ``HitSchema`` rather than
inheriting from this base.
"""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import Scalar


class CapillaryPressure(Model):
    """Relate the porous flow capillary pressure to the effective saturation."""

    hit = HitSchema(
        input("effective_saturation", Scalar, "The effective saturation"),
        output("capillary_pressure", Scalar, "Capillary pressure."),
        option(
            "log_extension",
            bool,
            "Whether to apply logarithmic extension",
            default=False,
            attr="_log_extension",
        ),
        option(
            "transition_saturation",
            float,
            "The transition value of the effective saturation below which to "
            "apply the logarithmic extension",
            default=0.0,
            attr="_Sp",
        ),
    )

    _log_extension: bool
    _Sp: float

    def forward(  # type: ignore[override]
        self,
        effective_saturation: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native capillary-pressure leaves implement the
        # forward operator (effective_saturation -> capillary_pressure) and
        # its differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
