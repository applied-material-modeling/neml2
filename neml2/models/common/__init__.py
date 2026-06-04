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


"""Native common models.

One file per C++ header under ``include/neml2/models/common/``; this package
re-imports each file so ``@register_native`` side effects fire on package import.
"""

from .ArrheniusParameter import ArrheniusParameter
from .BackwardEulerTimeIntegration import (
    R2BackwardEulerTimeIntegration,
    ScalarBackwardEulerTimeIntegration,
    SR2BackwardEulerTimeIntegration,
    VecBackwardEulerTimeIntegration,
)
from .BilinearInterpolation import (
    ScalarBilinearInterpolation,
    SR2BilinearInterpolation,
    VecBilinearInterpolation,
)
from .ComposedModel import ComposedModel
from .ConstantExtrapolationPredictor import ConstantExtrapolationPredictor
from .ConstantParameter import (
    MillerIndexConstantParameter,
    R2ConstantParameter,
    RotConstantParameter,
    ScalarConstantParameter,
    SR2ConstantParameter,
    SSR4ConstantParameter,
    VecConstantParameter,
    WR2ConstantParameter,
)
from .CopyVariable import (
    CopyMillerIndex,
    CopyR2,
    CopyRot,
    CopyScalar,
    CopySR2,
    CopySSR4,
    CopyVec,
    CopyWR2,
)
from .Determinant import R2Determinant, SR2Determinant
from .DynamicMean import SR2DynamicMean
from .DynamicSum import SR2DynamicSum
from .FBComplementarity import FBComplementarity
from .ForwardEulerTimeIntegration import (
    ScalarForwardEulerTimeIntegration,
    SR2ForwardEulerTimeIntegration,
)
from .HermiteSmoothStep import HermiteSmoothStep
from .ImplicitUpdate import ImplicitUpdate
from .InputParameter import (
    MillerIndexInputParameter,
    R2InputParameter,
    RotInputParameter,
    ScalarInputParameter,
    SR2InputParameter,
    SSR4InputParameter,
    VecInputParameter,
    WR2InputParameter,
)
from .IntermediateDiff import SR2IntermediateDiff
from .IntermediateMean import SR2IntermediateMean
from .IntermediateSum import SR2IntermediateSum
from .Interpolation import Interpolation
from .IrreversibleScalar import IrreversibleScalar
from .LinearCombination import ScalarLinearCombination, SR2LinearCombination
from .LinearExtrapolationPredictor import LinearExtrapolationPredictor
from .LinearInterpolation import ScalarLinearInterpolation
from .MacaulaySplit import MacaulaySplit
from .MixedControlSetup import MixedControlSetup
from .ParameterToVariable import (
    MillerIndexParameterToVariable,
    R2ParameterToVariable,
    RotParameterToVariable,
    ScalarParameterToVariable,
    SR2ParameterToVariable,
    SSR4ParameterToVariable,
    VecParameterToVariable,
    WR2ParameterToVariable,
)
from .R2Multiplication import R2Multiplication
from .R2ToSR2 import R2ToSR2
from .R2ToWR2 import R2ToWR2
from .Reduction import Reduction
from .RotationMatrix import RotationMatrix
from .ScalarMultiplication import ScalarMultiplication
from .ScalarPNorm import ScalarPNorm
from .ScalarToDiagonalSR2 import ScalarToDiagonalSR2
from .SR2Invariant import SR2Invariant
from .SR2ToR2 import SR2ToR2
from .SymmetricHermiteInterpolation import SymmetricHermiteInterpolation
from .VariableRate import (
    R2VariableRate,
    ScalarVariableRate,
    SR2VariableRate,
    VecVariableRate,
)
from .VecComponents import VecComponents
from .WR2ExplicitExponentialTimeIntegration import WR2ExplicitExponentialTimeIntegration
from .WR2ImplicitExponentialTimeIntegration import WR2ImplicitExponentialTimeIntegration

__all__ = [
    "ArrheniusParameter",
    "ComposedModel",
    "ImplicitUpdate",
    "SR2LinearCombination",
    "ScalarLinearCombination",
    "SR2Invariant",
    "R2BackwardEulerTimeIntegration",
    "ScalarBackwardEulerTimeIntegration",
    "SR2BackwardEulerTimeIntegration",
    "VecBackwardEulerTimeIntegration",
    "ScalarForwardEulerTimeIntegration",
    "SR2ForwardEulerTimeIntegration",
    "FBComplementarity",
    "HermiteSmoothStep",
    "IrreversibleScalar",
    "MacaulaySplit",
    "ConstantExtrapolationPredictor",
    "LinearExtrapolationPredictor",
    "MillerIndexConstantParameter",
    "R2ConstantParameter",
    "RotConstantParameter",
    "SR2ConstantParameter",
    "SSR4ConstantParameter",
    "ScalarConstantParameter",
    "VecConstantParameter",
    "WR2ConstantParameter",
    "CopyMillerIndex",
    "CopyR2",
    "CopyRot",
    "CopyScalar",
    "CopySR2",
    "CopySSR4",
    "CopyVec",
    "CopyWR2",
    "MillerIndexInputParameter",
    "R2InputParameter",
    "RotInputParameter",
    "SR2InputParameter",
    "SSR4InputParameter",
    "ScalarInputParameter",
    "VecInputParameter",
    "WR2InputParameter",
    "MillerIndexParameterToVariable",
    "R2ParameterToVariable",
    "RotParameterToVariable",
    "SR2ParameterToVariable",
    "SSR4ParameterToVariable",
    "ScalarParameterToVariable",
    "VecParameterToVariable",
    "WR2ParameterToVariable",
    "ScalarLinearInterpolation",
    "ScalarBilinearInterpolation",
    "SR2BilinearInterpolation",
    "VecBilinearInterpolation",
    "RotationMatrix",
    "ScalarMultiplication",
    "ScalarPNorm",
    "ScalarToDiagonalSR2",
    "SymmetricHermiteInterpolation",
    "WR2ExplicitExponentialTimeIntegration",
    "WR2ImplicitExponentialTimeIntegration",
    "MixedControlSetup",
    "R2Determinant",
    "R2Multiplication",
    "R2ToSR2",
    "R2ToWR2",
    "SR2Determinant",
    "SR2DynamicMean",
    "SR2DynamicSum",
    "SR2IntermediateDiff",
    "Interpolation",
    "Reduction",
    "SR2IntermediateMean",
    "SR2IntermediateSum",
    "SR2ToR2",
    "SR2VariableRate",
    "R2VariableRate",
    "ScalarVariableRate",
    "VecComponents",
    "VecVariableRate",
]
