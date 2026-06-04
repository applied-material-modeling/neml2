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


"""Native solid-mechanics crystal-plasticity models.

One file per C++ header under
``include/neml2/models/solid_mechanics/crystal_plasticity/``; this package
re-imports each so the ``@register_neml2_object`` side effects fire on package import.
"""

from .CrystalPlasticityStrainPredictor import CrystalPlasticityStrainPredictor
from .DislocationObstacleStrengthMap import DislocationObstacleStrengthMap
from .ElasticStrainRate import ElasticStrainRate
from .FixOrientation import FixOrientation
from .LinearSingleSlipHardeningRule import LinearSingleSlipHardeningRule
from .OrientationRate import OrientationRate
from .PerSlipForestDislocationEvolution import PerSlipForestDislocationEvolution
from .PlasticDeformationRate import PlasticDeformationRate
from .PlasticSpatialVelocityGradient import PlasticSpatialVelocityGradient
from .PlasticVorticity import PlasticVorticity
from .PowerLawSlipRule import PowerLawSlipRule
from .ResolvedShear import ResolvedShear
from .SingleSlipHardeningRule import SingleSlipHardeningRule
from .SingleSlipStrengthMap import SingleSlipStrengthMap
from .SlipRule import SlipRule
from .SlipStrengthMap import SlipStrengthMap
from .SumSlipRates import SumSlipRates
from .VoceSingleSlipHardeningRule import VoceSingleSlipHardeningRule

__all__ = [
    "CrystalPlasticityStrainPredictor",
    "DislocationObstacleStrengthMap",
    "ElasticStrainRate",
    "FixOrientation",
    "LinearSingleSlipHardeningRule",
    "OrientationRate",
    "PerSlipForestDislocationEvolution",
    "PlasticDeformationRate",
    "PlasticSpatialVelocityGradient",
    "PlasticVorticity",
    "PowerLawSlipRule",
    "ResolvedShear",
    "SingleSlipHardeningRule",
    "SingleSlipStrengthMap",
    "SlipRule",
    "SlipStrengthMap",
    "SumSlipRates",
    "VoceSingleSlipHardeningRule",
]
