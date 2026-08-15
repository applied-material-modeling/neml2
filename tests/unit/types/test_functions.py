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
    acos,
    asin,
    atan,
    cos,
    cumsum,
    sin,
    tan,
    triu,
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


# ---------------------------------------------------------------------------
# Intermediate-dim helpers: cumsum, triu
# ---------------------------------------------------------------------------


def test_cumsum_sub_batch() -> None:
    x = Scalar(torch.tensor([1.0, 2.0, 3.0, 4.0]), sub_batch_ndim=1)
    c = cumsum(x.sub_batch, dim=0)
    assert c.sub_batch_ndim == 1
    assert torch.allclose(c.data, torch.tensor([1.0, 3.0, 6.0, 10.0]))


def test_cumsum_selects_axis() -> None:
    # (2, 3) sub-batch grid; cumulative sum down each axis independently.
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = Scalar(m, sub_batch_ndim=2)
    assert torch.allclose(cumsum(x.sub_batch, dim=0).data, torch.cumsum(m, dim=0))
    assert torch.allclose(cumsum(x.sub_batch, dim=1).data, torch.cumsum(m, dim=1))


def test_triu_default_and_diagonal() -> None:
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x = Scalar(m, sub_batch_ndim=2)
    assert torch.allclose(triu(x.sub_batch).data, torch.triu(m))
    assert torch.allclose(triu(x.sub_batch, diagonal=1).data, torch.triu(m, diagonal=1))


def test_triu_batched() -> None:
    # A leading batch axis is preserved; triu acts on the trailing (row, col) grid.
    m = torch.arange(2.0 * 3.0 * 3.0).reshape(2, 3, 3)
    x = Scalar(m, sub_batch_ndim=2)
    out = triu(x.sub_batch, diagonal=1)
    assert out.sub_batch_ndim == 2
    assert torch.allclose(out.data, torch.triu(m, diagonal=1))


def test_triu_requires_two_sub_batch_axes() -> None:
    x = Scalar(torch.tensor([1.0, 2.0, 3.0]), sub_batch_ndim=1)
    with pytest.raises(ValueError, match="sub_batch_ndim >= 2"):
        triu(x.sub_batch)


def test_triu_rejects_non_sub_batch_view() -> None:
    x = Scalar(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sub_batch_ndim=2)
    with pytest.raises(TypeError, match="t.sub_batch view"):
        triu(x.dynamic_batch)  # type: ignore[arg-type]
