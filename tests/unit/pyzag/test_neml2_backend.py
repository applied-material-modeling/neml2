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

"""Unit tests for the neml2-backed pyzag block-operator backend."""

import pytest
import torch

from neml2.es import AssembledMatrix, AssembledVector
from neml2.es.axis_layout import AxisLayout
from neml2.pyzag.operators import (
    CachingLU,
    NEML2BlockVector,
    NEML2SolvableBlockOperator,
    _require_le_one_intmd,
    _select_dynamic_am,
    _select_dynamic_av,
    _transpose_am,
)
from neml2.solvers import DenseLU
from neml2.types import Scalar, Tensor

torch.manual_seed(42)


def _dense_layout(names):
    return AxisLayout(
        [[n] for n in names],
        {n: Scalar for n in names},
        {},
        tuple("dense" for _ in names),
    )


def _block(nb, batch, r, c):
    return Tensor(torch.randn(nb, batch, r, c, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=0)


def test_transpose_am_roundtrip():
    rl = _dense_layout(["a", "b"])
    cl = _dense_layout(["c", "d"])
    am = AssembledMatrix(
        rl, cl, [[_block(4, 2, 2, 3), _block(4, 2, 2, 5)], [_block(4, 2, 7, 3), _block(4, 2, 7, 5)]]
    )
    amt = _transpose_am(am)
    assert tuple(amt.tensors[0][0].data.shape[-2:]) == (3, 2)
    assert torch.equal(amt.tensors[0][1].data, am.tensors[1][0].data.transpose(-1, -2))
    amtt = _transpose_am(amt)
    for i in range(2):
        for j in range(2):
            assert torch.equal(amtt.tensors[i][j].data, am.tensors[i][j].data)


def test_select_dynamic():
    rl = _dense_layout(["a"])
    am = AssembledMatrix(rl, rl, [[_block(6, 2, 3, 3)]])
    sl = _select_dynamic_am(am, slice(1, 4))
    assert sl.tensors[0][0].data.shape[0] == 3
    assert torch.equal(sl.tensors[0][0].data, am.tensors[0][0].data[1:4])
    v = AssembledVector(
        rl, [Tensor(torch.randn(6, 2, 3, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=0)]
    )
    vs = _select_dynamic_av(v, slice(2, None))
    assert vs.tensors[0].data.shape[0] == 4


def test_two_intmd_block_rejected():
    rl = _dense_layout(["a"])
    blk = Tensor(torch.zeros(4, 2, 3, 3, 2, 2, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=2)
    am = AssembledMatrix(rl, rl, [[blk]])
    with pytest.raises(NotImplementedError):
        _require_le_one_intmd(am)
    with pytest.raises(NotImplementedError):
        _transpose_am(am)
    op = NEML2SolvableBlockOperator(am)
    v = NEML2BlockVector([torch.zeros(4, 2, 3, 3, 2, dtype=torch.float64)], rl, [2])
    with pytest.raises(NotImplementedError):
        op.matvec(v)


def test_caching_lu_matches_dense_lu():
    rl = _dense_layout(["a"])
    n = 5
    diag = torch.randn(3, 2, n, n, dtype=torch.float64) + 4.0 * torch.eye(n, dtype=torch.float64)
    am = AssembledMatrix(rl, rl, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    b = AssembledVector(
        rl, [Tensor(torch.randn(3, 2, n, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=0)]
    )
    x_dense = DenseLU().solve(am, b)
    x_cache = CachingLU().solve(am, b)
    assert torch.allclose(x_dense.tensors[0].data, x_cache.tensors[0].data)


def test_caching_lu_reuses_factorization():
    rl = _dense_layout(["a"])
    n = 4
    diag = torch.randn(2, 1, n, n, dtype=torch.float64) + 3.0 * torch.eye(n, dtype=torch.float64)
    am = AssembledMatrix(rl, rl, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    b = AssembledVector(
        rl, [Tensor(torch.randn(2, 1, n, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=0)]
    )
    solver = CachingLU()
    solver.solve(am, b)
    key_after_first = solver._key
    solver.solve(am, b)
    assert solver._key == key_after_first


def test_pcr_single_group_dense_matches_thomas():
    from pyzag import chunktime

    rl = _dense_layout(["a"])
    nblk, batch, n = 8, 2, 4
    A = torch.randn(nblk, batch, n, n, dtype=torch.float64) + 6.0 * torch.eye(
        n, dtype=torch.float64
    )
    Bsub = 0.2 * torch.randn(nblk, batch, n, n, dtype=torch.float64)
    rhs = torch.randn(nblk, batch, n, dtype=torch.float64)
    Aop = NEML2SolvableBlockOperator(
        AssembledMatrix(rl, rl, [[Tensor(A, batch_ndim=2, sub_batch_ndim=0)]])
    )
    Bop = NEML2SolvableBlockOperator(
        AssembledMatrix(rl, rl, [[Tensor(Bsub[1:], batch_ndim=2, sub_batch_ndim=0)]])
    )
    v = NEML2BlockVector([rhs.clone()], rl, [0])
    xt = chunktime.BidiagonalThomasFactorization(Aop, Bop).matvec(v.clone())
    xp = chunktime.BidiagonalPCRFactorization(Aop, Bop).matvec(v.clone())
    assert torch.allclose(xt.raw_tensors[0], xp.raw_tensors[0], atol=1e-10)


def test_pcr_block_group_raises_not_implemented():
    lay = AxisLayout([["a"]], {"a": Scalar}, {"a": torch.Size([3])}, ("block",))
    blk = torch.zeros(4, 2, 3, 1, 1, dtype=torch.float64)
    op = NEML2SolvableBlockOperator(
        AssembledMatrix(lay, lay, [[Tensor(blk, batch_ndim=2, sub_batch_ndim=1)]])
    )
    v = NEML2BlockVector([torch.zeros(4, 2, 3, 1, dtype=torch.float64)], lay, [1])
    with pytest.raises(NotImplementedError):
        op.pcr_init(op, v)


def test_factored_caches_lu_and_slices():
    rl = _dense_layout(["a"])
    n = 4
    diag = torch.randn(6, 2, n, n, dtype=torch.float64) + 5.0 * torch.eye(n, dtype=torch.float64)
    am = AssembledMatrix(rl, rl, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    op = NEML2SolvableBlockOperator.factored(am)
    assert op._lu is not None and op._piv is not None
    sliced = op[1:3]
    assert sliced._lu is not None and sliced._lu.shape[0] == 2
    b = NEML2BlockVector([torch.randn(6, 2, n, dtype=torch.float64)], rl, [0])
    x = op.solve(b)
    ref = DenseLU().solve(am, b.to_av())
    assert torch.allclose(x.raw_tensors[0], ref.tensors[0].data, atol=1e-10)


def test_single_group_solve_matvec_roundtrip():
    rl = _dense_layout(["a"])
    n = 4
    diag = torch.randn(2, 3, n, n, dtype=torch.float64) + 5.0 * torch.eye(n, dtype=torch.float64)
    am = AssembledMatrix(rl, rl, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    op = NEML2SolvableBlockOperator(am)
    x = NEML2BlockVector([torch.randn(2, 3, n, dtype=torch.float64)], rl, [0])
    b = op.matvec(x)
    x_back = op.solve(b)
    assert torch.allclose(x.raw_tensors[0], x_back.raw_tensors[0], atol=1e-9)
