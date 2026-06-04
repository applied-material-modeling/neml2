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


"""Native phase-field-fracture models.

One file per C++ header under ``include/neml2/models/phase_field_fracture/``;
this package re-imports each file so ``@register_neml2_object`` side effects fire on
package import.
"""

from .CrackGeometricFunction import CrackGeometricFunction
from .CrackGeometricFunctionAT1 import CrackGeometricFunctionAT1
from .CrackGeometricFunctionAT2 import CrackGeometricFunctionAT2
from .DegradationFunction import DegradationFunction
from .LinearIsotropicStrainEnergyDensity import LinearIsotropicStrainEnergyDensity
from .PowerDegradationFunction import PowerDegradationFunction
from .RationalDegradationFunction import RationalDegradationFunction
from .StrainEnergyDensity import StrainEnergyDensity

__all__ = [
    "CrackGeometricFunction",
    "CrackGeometricFunctionAT1",
    "CrackGeometricFunctionAT2",
    "DegradationFunction",
    "LinearIsotropicStrainEnergyDensity",
    "PowerDegradationFunction",
    "RationalDegradationFunction",
    "StrainEnergyDensity",
]
