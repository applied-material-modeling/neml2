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

"""Unit tests for the pyzag block types' full method surface.

The end-to-end tests exercise the happy solve/adjoint path; these target the
per-method behaviour (``NEML2BlockVector`` algebra, ``NEML2SolvableBlockOperator``
matvec/solve/slicing, ``NEML2BlockJacobian`` walks, wrapper, cached LU) that
pyzag invokes in line-search / predictor / indexing branches a converging model
never reaches.
"""

from typing import cast

import pytest
import torch
from pyzag import chunktime

from neml2.es import AssembledMatrix
from neml2.es.axis_layout import AxisLayout, SubBatchStructure
from neml2.pyzag.operators import (
    CachingLU,
    NEML2BlockJacobian,
    NEML2BlockVector,
    NEML2SolvableBlockOperator,
    NEML2Wrapper,
)
from neml2.types import SR2, Scalar, Tensor

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _dense_layout(names):
    """Single-DOF-per-variable dense layout, one group per name."""
    return AxisLayout(
        [[n] for n in names],
        {n: Scalar for n in names},
        {},
        cast(tuple[SubBatchStructure, ...], tuple("dense" for _ in names)),
    )


def _dense_op(names, nblk, batch, seed=0):
    """SPD single-group dense operator plus its layout (one group of scalars)."""
    torch.manual_seed(seed)
    lay = AxisLayout([names], {n: Scalar for n in names}, {}, ("dense",))
    m = len(names)
    M = torch.randn(nblk, batch, m, m)
    diag = M @ M.transpose(-1, -2) + (m + 4.0) * torch.eye(m)
    am = AssembledMatrix(lay, lay, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    return NEML2SolvableBlockOperator(am), lay


def _two_group(N, batch, nblk, seed=0):
    """Arrowhead BLOCK(per-site)+DENSE(global) operator, layout, and a matching rhs."""
    torch.manual_seed(seed)
    npp = nss = 6
    lay = AxisLayout(
        [["p"], ["s"]],
        {"p": SR2, "s": SR2},
        {"p": torch.Size([N]), "s": torch.Size([])},
        ("block", "dense"),
    )
    Ms = torch.randn(nblk, batch, N, npp, npp)
    App = Ms @ Ms.transpose(-1, -2) + (npp + 4.0) * torch.eye(npp)
    Md = torch.randn(nblk, batch, nss, nss)
    Ass = Md @ Md.transpose(-1, -2) + (nss + 4.0) * torch.eye(nss)
    Aps = 0.05 * torch.randn(nblk, batch, N, npp, nss)
    Asp = 0.05 * torch.randn(nblk, batch, N, nss, npp)
    am = AssembledMatrix(
        lay,
        lay,
        [
            [
                Tensor(App, batch_ndim=2, sub_batch_ndim=1),
                Tensor(Aps, batch_ndim=2, sub_batch_ndim=1),
            ],
            [
                Tensor(Asp, batch_ndim=2, sub_batch_ndim=1),
                Tensor(Ass, batch_ndim=2, sub_batch_ndim=0),
            ],
        ],
    )
    vp = torch.randn(nblk, batch, N, npp)
    vs = torch.randn(nblk, batch, nss)
    rhs = NEML2BlockVector([vp, vs], lay, [1, 0])
    return NEML2SolvableBlockOperator(am), lay, rhs


def _bv(nblk=3, batch=2, N=4):
    """A two-group (block + dense) NEML2BlockVector with distinct entries."""
    lay = AxisLayout(
        [["p"], ["s"]],
        {"p": SR2, "s": SR2},
        {"p": torch.Size([N]), "s": torch.Size([])},
        ("block", "dense"),
    )
    vp = torch.randn(nblk, batch, N, 6)
    vs = torch.randn(nblk, batch, 6)
    return NEML2BlockVector([vp, vs], lay, [1, 0]), lay


# --------------------------------------------------------------------------- #
# NEML2BlockVector                                                             #
# --------------------------------------------------------------------------- #
def test_block_vector_init_validation():
    lay = _dense_layout(["a", "b"])
    with pytest.raises(ValueError):
        NEML2BlockVector([torch.zeros(2, 1, 1)], lay, [0, 0])  # too few tensors
    with pytest.raises(ValueError):
        NEML2BlockVector([torch.zeros(2, 1, 1), torch.zeros(2, 1, 1)], lay, [0])  # bad intmd len


def test_block_vector_properties():
    v, _ = _bv(nblk=3, batch=2, N=4)
    assert v.device == v.raw_tensors[0].device
    assert v.dtype == torch.float64
    assert v.nblk == 3
    assert v.batch_size == 2
    assert v.block_size == 4 * 6 + 6  # per-site group (N*6) + dense group (6)


def test_block_vector_clone_is_deep():
    v, _ = _bv()
    c = v.clone()
    v.raw_tensors[0].add_(1.0)
    assert not torch.allclose(c.raw_tensors[0], v.raw_tensors[0])


def test_block_vector_norm_and_flatten():
    v, _ = _bv(nblk=3, batch=2, N=4)
    assert v.norm().shape == (3, 2)
    flat = v.flatten()
    assert flat.shape == (2, 3 * (4 * 6 + 6))


def test_block_vector_arithmetic():
    v, _ = _bv()
    assert torch.allclose((v + v).raw_tensors[0], 2.0 * v.raw_tensors[0])
    assert torch.allclose((v - v).raw_tensors[1], torch.zeros_like(v.raw_tensors[1]))
    assert torch.allclose((-v).raw_tensors[0], -v.raw_tensors[0])
    assert torch.allclose((v * 3.0).raw_tensors[1], 3.0 * v.raw_tensors[1])
    with pytest.raises(TypeError):
        v + object()  # type: ignore[operator]
    with pytest.raises(TypeError):
        v - object()  # type: ignore[operator]


def test_block_vector_where_scale_flip():
    v, _ = _bv(nblk=3, batch=2, N=4)
    other = v.clone()
    other.raw_tensors[0].add_(5.0)
    mask = torch.tensor([True, False])
    picked = v.where(mask, other)
    assert torch.allclose(picked.raw_tensors[0][:, 0], v.raw_tensors[0][:, 0])
    assert torch.allclose(picked.raw_tensors[0][:, 1], other.raw_tensors[0][:, 1])
    scaled = v.scale_batches(torch.tensor([2.0, 0.5]))
    assert torch.allclose(scaled.raw_tensors[0][:, 0], 2.0 * v.raw_tensors[0][:, 0])
    assert torch.allclose(v.flip(0).raw_tensors[0], v.raw_tensors[0].flip(0))
    with pytest.raises(TypeError):
        v.where(mask, object())  # type: ignore[arg-type]


def test_block_vector_getitem_setitem():
    v, lay = _bv(nblk=4, batch=2, N=3)
    single = v[0]  # int index keeps a length-1 dynamic axis
    assert single.raw_tensors[0].shape[0] == 1
    assert single.raw_tensors[0].shape[1:] == v.raw_tensors[0].shape[1:]
    chunk = v[1:3]
    assert chunk.raw_tensors[0].shape[0] == 2
    target = v.clone()
    repl = NEML2BlockVector(
        [torch.zeros_like(v.raw_tensors[0][0:1]), torch.zeros_like(v.raw_tensors[1][0:1])],
        lay,
        [1, 0],
    )
    target[0:1] = repl
    assert torch.count_nonzero(target.raw_tensors[0][0]) == 0
    with pytest.raises(TypeError):
        target[0:1] = object()  # type: ignore[assignment]


def test_block_vector_cat_and_zeros_like():
    v, _ = _bv(nblk=3, batch=2, N=4)
    catted = NEML2BlockVector.cat([v, v], dim=0)
    assert catted.raw_tensors[0].shape[0] == 6
    z = NEML2BlockVector.zeros_like(v)
    assert torch.count_nonzero(z.raw_tensors[0]) == 0
    assert z.raw_tensors[0].shape == v.raw_tensors[0].shape
    with pytest.raises(ValueError):
        NEML2BlockVector.cat([], dim=0)
    with pytest.raises(TypeError):
        NEML2BlockVector.cat([v, object()], dim=0)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        NEML2BlockVector.zeros_like(object())  # type: ignore[arg-type]


def test_block_vector_to_from_av_roundtrip():
    v, _ = _bv()
    av = v.to_av()
    back = NEML2BlockVector.from_av(av)
    for a, b in zip(v.raw_tensors, back.raw_tensors, strict=True):
        assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# NEML2SolvableBlockOperator                                                   #
# --------------------------------------------------------------------------- #
def test_operator_properties_and_guards():
    op, lay = _dense_op(["a", "b", "c"], nblk=5, batch=2)
    assert op.dtype == torch.float64
    assert op.device == op.am.tensors[0][0].data.device
    assert op.nblk == 5
    assert op.batch_size == 2
    assert op._is_single_dense()
    with pytest.raises(TypeError):
        op.matvec(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        op.t_matvec(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        op.solve(object())  # type: ignore[arg-type]


def test_operator_matvec_solve_roundtrip_two_group():
    op, lay, _ = _two_group(N=4, batch=2, nblk=3, seed=1)
    x = NEML2BlockVector([torch.randn(3, 2, 4, 6), torch.randn(3, 2, 6)], lay, [1, 0])
    b = op.matvec(x)  # exercises the DENSE-consumes-BLOCK aggregation branch
    x_back = op.solve(b)  # 2-group Schur solve
    assert torch.allclose(x.raw_tensors[0], x_back.raw_tensors[0], atol=1e-8)
    assert torch.allclose(x.raw_tensors[1], x_back.raw_tensors[1], atol=1e-8)


def test_operator_t_matvec_runs():
    op, lay, rhs = _two_group(N=3, batch=2, nblk=2, seed=2)
    out = op.t_matvec(rhs)
    assert out.raw_tensors[0].shape == rhs.raw_tensors[0].shape


def test_operator_solve_three_group_not_implemented():
    lay = _dense_layout(["a", "b", "c"])

    def blk():
        return Tensor(torch.zeros(2, 1, 1, 1), batch_ndim=2, sub_batch_ndim=0)

    am = AssembledMatrix(lay, lay, [[blk() for _ in range(3)] for _ in range(3)])
    op = NEML2SolvableBlockOperator(am)
    rhs = NEML2BlockVector([torch.zeros(2, 1, 1) for _ in range(3)], lay, [0, 0, 0])
    with pytest.raises(NotImplementedError):
        op.solve(rhs)


def test_operator_clone_is_deep():
    op, _ = _dense_op(["a", "b"], nblk=3, batch=2)
    c = op.clone()
    op.am.tensors[0][0].data.add_(1.0)
    assert not torch.allclose(c.am.tensors[0][0].data, op.am.tensors[0][0].data)


def test_operator_getitem_setitem():
    op, _ = _dense_op(["a", "b"], nblk=5, batch=2)
    op = NEML2SolvableBlockOperator.factored(op.am)
    assert op[0].nblk == 1  # int -> length-1 slice, cache carried
    assert op[0]._lu is not None
    assert op[-1].nblk == 1
    assert op[1:4].nblk == 3
    other = op[0:1]
    op[0:1] = other  # cache invalidated on assignment
    assert op._lu is None
    with pytest.raises(TypeError):
        op[0:1] = object()  # type: ignore[assignment]


def test_operator_pad_front():
    op, _ = _dense_op(["a", "b"], nblk=3, batch=2)
    assert op.pad_front(0).nblk == 3
    assert op.pad_front(2).nblk == 5
    padded = op.pad_front(1)
    assert torch.count_nonzero(padded.am.tensors[0][0].data[0]) == 0
    with pytest.raises(ValueError):
        op.pad_front(-1)


def test_operator_pcr_init_guards():
    op, _ = _dense_op(["a"], nblk=4, batch=2)
    v = NEML2BlockVector([torch.randn(4, 2, 1)], op.am.row_layout, [0])
    with pytest.raises(TypeError):
        op.pcr_init(object(), v)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        op.pcr_init(op, object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# NEML2BlockJacobian                                                           #
# --------------------------------------------------------------------------- #
def _bidiag(nblk, batch, m, seed=0):
    torch.manual_seed(seed)
    names = [f"v{i}" for i in range(m)]
    lay = AxisLayout([names], {n: Scalar for n in names}, {}, ("dense",))
    Md = torch.randn(nblk, batch, m, m)
    diag = Md @ Md.transpose(-1, -2) + (m + 4.0) * torch.eye(m)
    sub = 0.2 * torch.randn(nblk, batch, m, m)
    diag_am = AssembledMatrix(lay, lay, [[Tensor(diag, batch_ndim=2, sub_batch_ndim=0)]])
    sub_am = AssembledMatrix(lay, lay, [[Tensor(sub, batch_ndim=2, sub_batch_ndim=0)]])
    return NEML2BlockJacobian(diag_am, sub_am, lay), lay


def test_jacobian_properties():
    jac, _ = _bidiag(nblk=5, batch=2, m=3)
    assert jac.dtype == torch.float64
    assert jac.device == jac.diag_am.tensors[0][0].data.device
    assert jac.nblk_steps == 5
    assert jac.batch_size == 2
    assert jac.block_size == 3


def test_jacobian_walk_guards():
    jac, _ = _bidiag(nblk=4, batch=2, m=3)
    # forward on a forward-walk jac is fine; on an adjoint walk it must raise.
    jac.forward_system(chunktime.BidiagonalThomasFactorization)
    adj = jac.as_adjoint_walk()
    with pytest.raises(RuntimeError):
        adj.forward_system(chunktime.BidiagonalThomasFactorization)
    with pytest.raises(RuntimeError):
        jac.adjoint_system(chunktime.BidiagonalThomasFactorization)
    adj.adjoint_system(chunktime.BidiagonalThomasFactorization)


def test_jacobian_terminal_and_couple():
    jac, lay = _bidiag(nblk=4, batch=2, m=3)
    g = torch.randn(2, 3)
    term = jac.solve_terminal_adjoint(g)
    assert term.raw_tensors[0].shape[-1] == 3
    a_first = NEML2BlockVector([torch.randn(1, 2, 3)], lay, [0])
    coupled = jac.as_adjoint_walk().couple_prev_chunk(a_first)
    assert coupled.raw_tensors[0].shape[-1] == 3
    with pytest.raises(TypeError):
        jac.couple_prev_chunk(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# NEML2Wrapper                                                                 #
# --------------------------------------------------------------------------- #
def test_wrapper_roundtrip_and_guard():
    lay = _dense_layout(["a", "b"])
    wrap = NEML2Wrapper(lay)
    flat = torch.randn(3, 2, 2)
    bv = wrap.wrap_vector(flat)
    assert isinstance(bv, NEML2BlockVector)
    assert torch.allclose(wrap.unwrap_vector(bv), flat)
    diag = AssembledMatrix(
        lay, lay, [[Tensor(torch.zeros(3, 2, 1, 1), batch_ndim=2, sub_batch_ndim=0)]]
    )
    jac = wrap.wrap_jacobian(diag, diag)
    assert isinstance(jac, NEML2BlockJacobian)
    with pytest.raises(TypeError):
        wrap.unwrap_vector(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CachingLU (matrix RHS branch)                                               #
# --------------------------------------------------------------------------- #
def test_caching_lu_matrix_rhs():
    lay = AxisLayout([["a"]], {"a": Scalar}, {}, ("dense",))
    n = 4
    A_raw = torch.randn(2, 1, n, n) + 4.0 * torch.eye(n)
    A = AssembledMatrix(lay, lay, [[Tensor(A_raw, batch_ndim=2, sub_batch_ndim=0)]])
    B = AssembledMatrix(
        lay, lay, [[Tensor(torch.randn(2, 1, n, 3), batch_ndim=2, sub_batch_ndim=0)]]
    )
    out = CachingLU().solve(A, B)
    assert isinstance(out, AssembledMatrix)
    assert out.tensors[0][0].data.shape[-1] == 3
