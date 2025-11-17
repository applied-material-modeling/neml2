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

import torch
import typing

Number = typing.Union[int, float, bool]

# Bring core functionality and tensors into the main namespace
from .core import *
from .tensors import *
from .math import *

# pybind11-stubgen generates incorrect type annotations for Unions
# so unfortunately we need to maintain this list
# see issue https://github.com/sizmailov/pybind11-stubgen/issues/276
TensorLike = typing.Union[
    Vec,
    Rot,
    WR2,
    R2,
    Scalar,
    SR2,
    R3,
    SFR3,
    R4,
    SFFR4,
    WFFR4,
    SSR4,
    SWR4,
    WSR4,
    WWR4,
    Quaternion,
    MillerIndex,
    Tensor,
]

# Other submodules
from . import pyzag
from . import crystallography
from . import reserved
