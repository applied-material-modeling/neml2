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

"""Register the shared test-only native models for the AOTI suite.

The ``request_ad_forward`` scenario uses ``SurrogateFlowRate`` (a request_AD
machine-learning surrogate), which lives in ``tests/regression/_fixtures`` rather
than the production package. Side-effect-import it so ``@register_neml2_object``
fires before the AOTI scenarios are collected -- the same mechanism
``tests/regression/conftest.py`` uses for the eager regression suite.
"""

import importlib
import sys
from pathlib import Path

_REGRESSION = Path(__file__).resolve().parent.parent / "regression"
if str(_REGRESSION) not in sys.path:
    sys.path.insert(0, str(_REGRESSION))

# Side-effect import: ``@register_neml2_object`` fires for the test fixtures.
# ``import_module`` (rather than ``import _fixtures``) honors the dynamic sys.path
# entry above without pyright flagging an unresolved import (``_fixtures`` lives
# under tests/regression/, not this directory).
importlib.import_module("_fixtures")
