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

"""Pytest conftest for the regression suite.

Side-effect-imports the test-only ``_fixtures`` package so its
``@register_neml2_object``-decorated models (``TabulatedPolynomialModel``,
``TorchScriptFlowRate``) are registered with the native factory before
any scenario ``.i`` file is collected. The package lives next to this
conftest at ``tests/regression/_fixtures/`` because it is regression-test
infrastructure with no consumer outside the regression suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put ``tests/regression/`` on sys.path so ``import _fixtures`` resolves
# regardless of pytest's rootdir / cwd, then trigger registration.
_REGRESSION_DIR = Path(__file__).resolve().parent
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

import _fixtures  # noqa: E402, F401  (side-effect: @register_neml2_object fires)
