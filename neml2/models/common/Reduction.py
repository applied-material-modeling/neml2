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

"""Python-native mirror of the C++ abstract ``Reduction<T>`` base.

The C++ ``Reduction`` (see ``include/neml2/models/common/Reduction.h``) is an
abstract intermediate templated on the reduced variable type $T$. It declares
a single ``from`` input and a ``to`` output of the same type; concrete
instantiations (sub-batch sums, means, dynamic reductions, etc.) provide the
actual reduction operator and its differential pushforward.

Because the C++ template itself is not registered (no
``register_NEML2_object`` on the abstract base), this native port is also
unregistered: ``@register_neml2_object`` is intentionally omitted. Native models are
flat (no schema inheritance), so concrete leaves declare their own
``HitSchema`` independently rather than inheriting from this base; ``forward``
raises ``NotImplementedError`` so any accidental direct use surfaces
immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...schema import HitSchema, input, output
from ...types import Scalar
from ..chain_rule import ChainRuleDict
from ..model import Model

if TYPE_CHECKING:
    pass


class Reduction(Model):
    """The base class for variable reductions.

    Concrete native reduction leaves declare their own ordinate-typed
    ``from`` input and ``to`` output; the shared surface here uses
    ``Scalar`` only to satisfy the schema declaration. Subclasses replace
    the canonical types with the appropriate primitive tensor wrapper.
    """

    hit = HitSchema(
        input("from", Scalar, "Variable to reduce"),
        output("to", Scalar, "The reduced variable"),
    )

    def forward(  # type: ignore[override]
        self,
        *inputs: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native reduction leaves implement the forward
        # operator (from -> reduced value) and its differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
