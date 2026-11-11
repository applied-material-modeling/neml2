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

"""Typed ``pow`` / ``opaque_pow`` with a ``Scalar`` exponent must align sub-batch axes.

A plain-batch exponent (e.g. a temperature-dependent rate sensitivity of
batch shape ``(B,)``) raised against a sub-batched base (per-slip ratios of
shape ``(B, n_slip)``) has to broadcast over the base's sub-batch axis, the
same way typed ``*`` / ``/`` do. Right-aligning the raw tensors instead pairs
``B`` with ``n_slip`` and either fails or silently mis-broadcasts.
"""

from __future__ import annotations

import pytest
import torch

from neml2.types import Scalar
from neml2.types.functions import opaque_pow, pow


@pytest.mark.parametrize("fn", [pow, opaque_pow])
def test_batch_exponent_broadcasts_over_base_sub_batch(fn):
    base = Scalar(torch.rand(5, 3, dtype=torch.float64) + 0.5, sub_batch_ndim=1)
    expo = Scalar(torch.linspace(2.0, 4.0, 5, dtype=torch.float64))
    out = fn(base, expo)
    assert out.sub_batch_ndim == 1
    assert torch.allclose(out.data, base.data ** expo.data[:, None])


@pytest.mark.parametrize("fn", [pow, opaque_pow])
def test_sub_batched_exponent_on_plain_base(fn):
    base = Scalar(torch.rand(5, dtype=torch.float64) + 0.5)
    expo = Scalar(torch.linspace(2.0, 4.0, 15, dtype=torch.float64).reshape(5, 3), sub_batch_ndim=1)
    out = fn(base, expo)
    assert out.sub_batch_ndim == 1
    assert torch.allclose(out.data, base.data[:, None] ** expo.data)


@pytest.mark.parametrize("fn", [pow, opaque_pow])
def test_float_exponent_keeps_sub_batch(fn):
    base = Scalar(torch.rand(5, 3, dtype=torch.float64) + 0.5, sub_batch_ndim=1)
    out = fn(base, 2.5)
    assert out.sub_batch_ndim == 1
    assert torch.allclose(out.data, base.data**2.5)
