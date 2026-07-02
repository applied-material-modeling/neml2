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

from __future__ import annotations

import pytest
import torch
from neml2.types import (
    Scalar,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
)

# ---------------------------------------------------------------------------
# Trigonometric Functions
# ---------------------------------------------------------------------------


def test_sin() -> None:
    s = sin(Scalar(torch.tensor([-2.0, 0.0, 1.0])))
    assert torch.allclose(s.data, torch.sin(torch.tensor([-2.0, 0.0, 1.0])))


def test_cos() -> None:
    s = cos(Scalar(torch.tensor([-2.0, 0.0, 1.0])))
    assert torch.allclose(s.data, torch.cos(torch.tensor([-2.0, 0.0, 1.0])))


def test_tan() -> None:
    s = tan(Scalar(torch.tensor([-2.0, 0.0, 1.0])))
    assert torch.allclose(s.data, torch.tan(torch.tensor([-2.0, 0.0, 1.0])))


def test_asin() -> None:
    s = asin(Scalar(torch.tensor([0.5, 0.0, -0.25])))
    assert torch.allclose(s.data, torch.asin(torch.tensor([0.5, 0.0, -0.25])))


def test_acos() -> None:
    s = acos(Scalar(torch.tensor([0.5, 0.0, -0.25])))
    assert torch.allclose(s.data, torch.acos(torch.tensor([0.5, 0.0, -0.25])))


def test_atan() -> None:
    s = atan(Scalar(torch.tensor([-2.0, 0.0, 1.0])))
    assert torch.allclose(s.data, torch.atan(torch.tensor([-2.0, 0.0, 1.0])))