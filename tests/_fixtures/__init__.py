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

"""Test-only Python-native NEML2 models.

These are the native counterparts of the C++ test helpers under
``tests/include/`` + ``tests/src/`` (``TabulatedPolynomialModel``,
``TorchScriptFlowRate``). They live here rather than in
``neml2.models`` because they only exist to back regression-test
inputs — they have no place in the production native package.

Importing this package side-effect-registers every fixture with the native
factory registry via ``@register_native``. The native ``regression/conftest.py``
imports it so the registrations fire before pytest collects the input files.
"""

from . import TabulatedPolynomialModel as _tabulated_polynomial  # noqa: F401
from . import TorchScriptFlowRate as _torch_script_flow_rate  # noqa: F401
