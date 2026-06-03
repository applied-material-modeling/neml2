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


"""Native KWN (Kampmann--Wagner Numerical) models.

One file per C++ header under ``include/neml2/models/kwn/``; this package
re-imports each file so ``@register_native`` side effects fire on package
import.
"""

from .ChemicalGibbsFreeEnergyDifference import ChemicalGibbsFreeEnergyDifference
from .CurrentConcentration import CurrentConcentration
from .IdealSolutionVolumetricDrivingForce import IdealSolutionVolumetricDrivingForce
from .KineticFactor import KineticFactor
from .NucleationBarrierAndCriticalRadius import NucleationBarrierAndCriticalRadius
from .NucleationFluxMagnitude import NucleationFluxMagnitude
from .PrecipitateVolumeFraction import PrecipitateVolumeFraction
from .ProjectedDiffusivitySum import ProjectedDiffusivitySum
from .RateLimitedPrecipitateGrowthRate import RateLimitedPrecipitateGrowthRate
from .SFFKPrecipitationGrowthRate import SFFKPrecipitationGrowthRate
from .ZeldovichFactor import ZeldovichFactor

__all__ = [
    "ChemicalGibbsFreeEnergyDifference",
    "CurrentConcentration",
    "IdealSolutionVolumetricDrivingForce",
    "KineticFactor",
    "NucleationBarrierAndCriticalRadius",
    "NucleationFluxMagnitude",
    "PrecipitateVolumeFraction",
    "ProjectedDiffusivitySum",
    "RateLimitedPrecipitateGrowthRate",
    "SFFKPrecipitationGrowthRate",
    "ZeldovichFactor",
]
