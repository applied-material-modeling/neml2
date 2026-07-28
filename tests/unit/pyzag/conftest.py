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

from pathlib import Path

import pytest

# Default-valued SR2LinearCombination weights/offset that the native model
# registers as parameters but are framework plumbing, not material parameters.
LINCOMB_INTERNAL_PARAMS = [
    "Eerate_offset",
    "Eerate_weight_0",
    "Eerate_weight_1",
]

# Per-scenario tolerances for comparing native trajectories against the gold
# baked by the retired C++ pipeline (float op-order drift).
COMPARE_TOLERANCES = {
    "km_mixed_model": {"rtol": 1e-4, "atol": 1e-8},
}

MODELS_DIR = Path(__file__).parent / "models"
GOLD_DIR = Path(__file__).parent / "gold"


@pytest.fixture
def models_dir():
    return MODELS_DIR


@pytest.fixture
def gold_dir():
    return GOLD_DIR


@pytest.fixture
def lincomb_internal_params():
    return list(LINCOMB_INTERNAL_PARAMS)


@pytest.fixture
def compare_tolerances():
    return dict(COMPARE_TOLERANCES)
