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

"""Native solid-mechanics plasticity models.

One file per C++ header under
``include/neml2/models/solid_mechanics/plasticity/``; this package re-imports
each so the ``@register_neml2_object`` side effects fire on package import.
"""

from .AssociativeIsotropicPlasticHardening import AssociativeIsotropicPlasticHardening
from .AssociativeJ2FlowDirection import AssociativeJ2FlowDirection
from .AssociativeKinematicPlasticHardening import AssociativeKinematicPlasticHardening
from .AssociativePlasticFlow import AssociativePlasticFlow
from .ChabochePlasticHardening import ChabochePlasticHardening
from .FlowRule import FlowRule
from .FredrickArmstrongPlasticHardening import FredrickArmstrongPlasticHardening
from .GTNYieldFunction import GTNYieldFunction
from .GursonCavitation import GursonCavitation
from .IsotropicHardening import IsotropicHardening
from .IsotropicHardeningStaticRecovery import IsotropicHardeningStaticRecovery
from .IsotropicMandelStress import IsotropicMandelStress
from .KinematicHardening import KinematicHardening
from .KinematicHardeningStaticRecovery import KinematicHardeningStaticRecovery
from .KocksMeckingActivationEnergy import KocksMeckingActivationEnergy
from .KocksMeckingFlowSwitch import KocksMeckingFlowSwitch
from .KocksMeckingFlowViscosity import KocksMeckingFlowViscosity
from .KocksMeckingIntercept import KocksMeckingIntercept
from .KocksMeckingRateSensitivity import KocksMeckingRateSensitivity
from .KocksMeckingYieldStress import KocksMeckingYieldStress
from .LinearIsotropicElasticJ2TrialStressUpdate import LinearIsotropicElasticJ2TrialStressUpdate
from .LinearIsotropicHardening import LinearIsotropicHardening
from .LinearKinematicHardening import LinearKinematicHardening
from .MandelStress import MandelStress
from .Normality import Normality
from .OlevskySinteringStress import OlevskySinteringStress
from .PerzynaPlasticFlowRate import PerzynaPlasticFlowRate
from .PlasticFlowRate import PlasticFlowRate
from .PowerLawIsotropicHardeningStaticRecovery import PowerLawIsotropicHardeningStaticRecovery
from .PowerLawKinematicHardeningStaticRecovery import PowerLawKinematicHardeningStaticRecovery
from .SlopeSaturationVoceIsotropicHardening import SlopeSaturationVoceIsotropicHardening
from .VoceIsotropicHardening import VoceIsotropicHardening
from .YieldFunction import YieldFunction

__all__ = [
    "AssociativeIsotropicPlasticHardening",
    "AssociativeJ2FlowDirection",
    "AssociativeKinematicPlasticHardening",
    "AssociativePlasticFlow",
    "ChabochePlasticHardening",
    "FlowRule",
    "FredrickArmstrongPlasticHardening",
    "GTNYieldFunction",
    "GursonCavitation",
    "IsotropicHardening",
    "IsotropicMandelStress",
    "KinematicHardening",
    "KocksMeckingActivationEnergy",
    "KocksMeckingFlowSwitch",
    "KocksMeckingFlowViscosity",
    "KocksMeckingIntercept",
    "KocksMeckingRateSensitivity",
    "KocksMeckingYieldStress",
    "LinearIsotropicElasticJ2TrialStressUpdate",
    "LinearIsotropicHardening",
    "LinearKinematicHardening",
    "MandelStress",
    "Normality",
    "OlevskySinteringStress",
    "PerzynaPlasticFlowRate",
    "PlasticFlowRate",
    "PowerLawIsotropicHardeningStaticRecovery",
    "PowerLawKinematicHardeningStaticRecovery",
    "SlopeSaturationVoceIsotropicHardening",
    "VoceIsotropicHardening",
    "YieldFunction",
]
