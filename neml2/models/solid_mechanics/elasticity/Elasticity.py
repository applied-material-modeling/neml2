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

"""Python-native mirror of the C++ abstract ``Elasticity`` base.

The C++ ``Elasticity`` (see
``include/neml2/models/solid_mechanics/elasticity/Elasticity.h``) is the
abstract base for all elasticity models. It wires the canonical ``strain`` /
``stress`` variable surface and exposes two switches, both inherited by every
concrete elastic relation:

* ``compliance``: when ``True`` the model maps stress to strain; otherwise
  strain to stress (the default).
* ``rate_form``: when ``True`` the input/output names get the ``_rate`` suffix,
  so the model defines the relation between rates instead of totals.

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical input/output names so concrete
native elasticity leaves can reuse the contract; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``LinearIsotropicElasticity`` declare their own ``HitSchema`` instead of
inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, input, option, output
from ....types import SR2

if TYPE_CHECKING:
    pass


class Elasticity(Model):
    """Relate elastic strain to stress."""

    hit = HitSchema(
        input("strain", SR2, "Elastic strain"),
        output("stress", SR2, "Stress"),
        option(
            "compliance",
            bool,
            "Whether the model defines the compliance relationship, i.e., mapping "
            "from stress to elastic strain. When set to false (default), the model "
            "maps elastic strain to stress.",
            default=False,
            attr="_compliance",
        ),
        option(
            "rate_form",
            bool,
            "Whether the model defines the stress-strain relationship in rate form. "
            "When set to true, the model maps elastic strain *rate* to stress *rate*.",
            default=False,
            attr="_rate_form",
        ),
    )

    _compliance: bool
    _rate_form: bool

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native elasticity leaves implement the forward
        # operator (strain -> stress, or stress -> strain when compliance) and
        # its differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
