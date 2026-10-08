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

"""Unit tests for the ``SquareMatrix`` user-tensor.

Pins each structured ``fill`` pattern -- the two ways a user supplies an interaction
matrix: build the whole thing (``dense``) or map a handful of physical coefficients
onto the grid by a slip-family partition (``block`` / ``diagonal_blocks``) -- plus the
emitted sub-batch tagging and the dimension-error messages.
"""

from __future__ import annotations

import pytest
import torch

import neml2.user_tensors  # noqa: F401 (register SquareMatrix)
from neml2 import load_string
from neml2.types import Scalar


def _matrix(body: str):
    """Build a ``SquareMatrix`` [Tensors] entry and return (data, sub_batch_ndim)."""
    f = load_string(f"[Tensors]\n[h]\ntype=SquareMatrix\n{body}\n[]\n[]")
    t = f.get_tensor("h")
    return t.data, t.sub_batch_ndim


def test_emits_two_axis_sub_batch_scalar():
    """The result is a Scalar carried on two sub-batch axes (n, n)."""
    data, sbn = _matrix("m=4\nfill=identity")
    assert sbn == 2
    assert tuple(data.shape) == (4, 4)
    torch.testing.assert_close(data, torch.eye(4, dtype=torch.float64), rtol=0, atol=0)


def test_zero_and_identity():
    data, _ = _matrix("m=3\nfill=zero")
    torch.testing.assert_close(data, torch.zeros(3, 3, dtype=torch.float64), rtol=0, atol=0)
    data, _ = _matrix("m=3\nfill=identity")
    torch.testing.assert_close(data, torch.eye(3, dtype=torch.float64), rtol=0, atol=0)


def test_diagonal():
    data, _ = _matrix("m=3\nfill=diagonal\ndata='1 2 3'")
    torch.testing.assert_close(
        data, torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)), rtol=0, atol=0
    )


def test_diagonal_blocks():
    """Each block fills that many consecutive diagonal entries with one scalar."""
    data, _ = _matrix("m=4\nfill=diagonal_blocks\ndata='10 20'\nblocks='2 2'")
    expected = torch.diag(torch.tensor([10.0, 10.0, 20.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(data, expected, rtol=0, atol=0)


def test_block_maps_family_coefficients():
    """``block`` sets one coefficient per slip-family pair (the Franciosi picture):
    data[bi*nblocks + bj] fills the whole family bi x family bj submatrix."""
    # self = 1.0 within a family, latent = 0.4 across families; two families of 2.
    data, _ = _matrix("fill=block\ndata='1.0 0.4 0.4 1.0'\nblocks='2 2'")
    expected = torch.tensor(
        [[1.0, 1.0, 0.4, 0.4], [1.0, 1.0, 0.4, 0.4], [0.4, 0.4, 1.0, 1.0], [0.4, 0.4, 1.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(data, expected, rtol=0, atol=0)


def test_block_asymmetric_is_row_major():
    """Off-diagonal family blocks are placed row-major: data[bi*nb+bj] -> (bi rows, bj cols)."""
    data, _ = _matrix("fill=block\ndata='1 2 3 4'\nblocks='1 1'")
    torch.testing.assert_close(
        data, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), rtol=0, atol=0
    )


def test_dense_is_row_major():
    data, _ = _matrix("m=2\nfill=dense\ndata='1 2 3 4'")
    torch.testing.assert_close(
        data, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), rtol=0, atol=0
    )


def test_m_inferred_from_blocks():
    """``m`` is optional when ``blocks`` is given -- the size is their sum."""
    data, _ = _matrix("fill=diagonal_blocks\ndata='1 2 3'\nblocks='1 2 1'")
    assert tuple(data.shape) == (4, 4)


@pytest.mark.parametrize(
    "body,msg",
    [
        ("m=3\nfill=diagonal\ndata='1 2'", "length m"),
        ("m=4\nfill=block\ndata='1 2 3 4'\nblocks='2 3'", "sum to m"),
        ("m=4\nfill=block\ndata='1 2 3'\nblocks='2 2'", "type 'block'"),
        ("m=2\nfill=dense\ndata='1 2 3'", "type 'dense'"),
        ("m=3\nfill=bogus", "invalid type"),
        ("fill=diagonal\ndata='1 2 3'", "required unless"),
    ],
)
def test_dimension_errors(body, msg):
    """Mis-specified sizes raise a clear ValueError, not a raw tensor error."""
    with pytest.raises(ValueError, match=msg):
        _matrix(body)


def test_feeds_interaction_strength_map():
    """The built matrix drives DislocationInteractionStrengthMap end to end."""
    txt = """
    [Tensors]
      [hmat]
        type = SquareMatrix
        fill = block
        data = '1.0 0.4 0.4 1.0'
        blocks = '2 2'
      []
    []
    [Models]
      [model]
        type = DislocationInteractionStrengthMap
        dislocation_density = 'dislocation_density'
        interaction_matrix = 'hmat'
        constant_strength = 5.0
        alpha = 0.3
        mu = 1000.0
        b = 0.1
      []
    []
    """
    model = load_string(txt).get_model("model")
    rho = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    tau = model.call_by_name({"dislocation_density": Scalar(rho, 1).with_sub_batch_ndim(1)})[
        "slip_strengths"
    ].data.reshape(4)
    h = torch.tensor(
        [[1.0, 1.0, 0.4, 0.4], [1.0, 1.0, 0.4, 0.4], [0.4, 0.4, 1.0, 1.0], [0.4, 0.4, 1.0, 1.0]],
        dtype=torch.float64,
    )
    expected = 5.0 + 0.3 * 1000.0 * 0.1 * torch.sqrt(h @ rho)
    torch.testing.assert_close(tau, expected, rtol=0, atol=1e-10)
