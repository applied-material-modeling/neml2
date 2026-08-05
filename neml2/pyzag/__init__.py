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

"""Adapter exposing a NEML2 nonlinear system to the pyzag time-integration library.

:class:`NEML2PyzagModel` wraps a NEML2 ``ModelNonlinearSystem`` as a
``pyzag.nonlinear.NonlinearFunctionOperatorFactory`` -- assembling the per-chunk
residual and bidiagonal Jacobian from NEML2's equation-systems layer and mirroring
the model's HIT parameters as ``torch.nn.Parameter`` s for gradient-based
calibration. The backing block-operator implementations live in
:mod:`neml2.pyzag.operators`.
"""

from .interface import NEML2PyzagModel, change_lag_order, lag_order
from .operators import (
    NEML2BlockJacobian,
    NEML2BlockVector,
    NEML2SolvableBlockOperator,
    NEML2Wrapper,
)

__all__ = [
    "NEML2PyzagModel",
    "change_lag_order",
    "lag_order",
    "NEML2BlockVector",
    "NEML2SolvableBlockOperator",
    "NEML2BlockJacobian",
    "NEML2Wrapper",
]
