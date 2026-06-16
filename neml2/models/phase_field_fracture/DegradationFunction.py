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

"""Python-native mirror of the C++ abstract ``DegradationFunction`` base.

The C++ ``DegradationFunction`` (see
``include/neml2/models/phase_field_fracture/DegradationFunction.h``) is an
abstract intermediate that fixes the canonical I/O contract for any model
that maps the phase-field variable to a scalar degradation factor:
``phase : Scalar`` in, ``degradation : Scalar`` out. Concrete forms
(``PowerDegradationFunction``, ``RationalDegradationFunction``, ...) supply
the actual ``g(d)`` expression and its differential pushforward.

Because the C++ class is not registered with ``register_NEML2_object``, this
native port is also unregistered: ``@register_neml2_object`` is intentionally
omitted. Native models are flat (no schema inheritance), so existing
concrete leaves declare their own ``HitSchema`` instead of inheriting from
this base. ``forward`` raises ``NotImplementedError`` so any accidental
direct use surfaces immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...schema import HitSchema, input, output
from ...types import Scalar
from ..chain_rule import ChainRuleDict
from ..model import Model

if TYPE_CHECKING:
    pass


class DegradationFunction(Model):
    """Parent class of degradation functions used in phase-field fracture models."""

    hit = HitSchema(
        input("phase", Scalar, "Phase-field variable"),
        output("degradation", Scalar, "Value of the degradation function"),
    )

    def forward(  # type: ignore[override]
        self,
        phase: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native degradation function leaves implement the
        # forward operator (phase -> degradation) and its differential
        # pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
