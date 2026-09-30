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

"""Solid-mechanics damage models.

One file per registered type; this package re-imports each so the
``@register_neml2_object`` side effects fire on package import.

Registered:

* Shared: ``DamagedStress`` -- applies sigma = (1 - D) sigma_tilde, the final
  stage of any scalar-damage composition.
* Mazars CDM family: ``MazarsEquivalentStrain``, ``MazarsDamageStressAlpha``.
  ``MazarsDamage`` with the simplified strain-magnitude α weighting is
  intentionally not ported -- the stress-based α of Mazars (1986) §3.5.1 is
  the variant in use here.
* Simo-Ju CDM family: ``WeibullDamage`` (three-parameter Brandyberry form)
  and ``ViscousDamageRelaxation`` (rate regularization of any damage target).
"""

from .DamagedStress import DamagedStress
from .MazarsDamageStressAlpha import MazarsDamageStressAlpha
from .MazarsEquivalentStrain import MazarsEquivalentStrain
from .ViscousDamageRelaxation import ViscousDamageRelaxation
from .WeibullDamage import WeibullDamage

__all__ = [
    "DamagedStress",
    "MazarsEquivalentStrain",
    "MazarsDamageStressAlpha",
    "WeibullDamage",
    "ViscousDamageRelaxation",
]
