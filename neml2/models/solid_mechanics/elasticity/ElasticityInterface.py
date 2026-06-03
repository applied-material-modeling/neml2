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

"""Python-native mirror of the C++ abstract ``ElasticityInterface``.

The C++ ``ElasticityInterface`` (see
``include/neml2/models/solid_mechanics/elasticity/ElasticityInterface.h``) is
a templated CRTP mixin: ``ElasticityInterface<Derived, N>`` adds a uniform
"choose any N independent elastic constants from {LAME_LAMBDA, BULK_MODULUS,
SHEAR_MODULUS, YOUNGS_MODULUS, POISSONS_RATIO, P_WAVE_MODULUS}" surface to a
derived elasticity model. It declares three HIT options:

* ``coefficients``: the N parameter values (literal floats or cross-refs).
* ``coefficient_types``: the matching list of ``ElasticConstant`` names that
  identifies what each coefficient represents.
* ``coefficient_as_parameter``: per-coefficient flag selecting (trainable)
  parameter vs static buffer storage; a single value broadcasts to all N.

Concrete C++ derivations (e.g. ``IsotropicElasticityTensor``,
``CubicElasticityTensor``) then look up the matching converter in
``IsotropicElasticityConverter::table`` / ``CubicElasticityConverter::table``
to map the user-chosen parameterization into the model's canonical pair (K, G)
or triple (C1, C2, C3).

Because the C++ class is not registered (no ``register_NEML2_object``), this
native port is also unregistered: ``@register_native`` is intentionally
omitted. The class scaffolds the canonical option surface so concrete native
elasticity leaves can be authored consistently; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
Native models are flat (no schema inheritance), so existing concrete leaves
like ``IsotropicElasticityTensor`` declare their own ``HitSchema`` and parse
``coefficients`` / ``coefficient_types`` directly in ``from_hit`` instead of
inheriting from this base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....model import Model
from ....schema import HitSchema, option

if TYPE_CHECKING:
    pass


class ElasticityInterface(Model):
    """Interface for objects defining elasticity tensors in terms of other parameters."""

    hit = HitSchema(
        option(
            "coefficients",
            list,
            "Coefficients used to define the elasticity tensor",
            reader=lambda node, name: node.param_list_str(name),
        ),
        option(
            "coefficient_types",
            list,
            "Types for each parameter, options are: INVALID, P_WAVE_MODULUS, "
            "POISSONS_RATIO, YOUNGS_MODULUS, SHEAR_MODULUS, BULK_MODULUS, "
            "LAME_LAMBDA",
            reader=lambda node, name: node.param_list_str(name),
        ),
        option(
            "coefficient_as_parameter",
            list,
            "Whether to treat the coefficients as (trainable) parameters. Default "
            "is true. Setting this option to false will treat the coefficients as "
            "buffers.",
            default=[True],
            reader=lambda node, name: node.param_list_str(name),
        ),
    )

    def forward(  # type: ignore[override]
        self,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native elasticity-interface leaves implement the
        # forward operator and its differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
