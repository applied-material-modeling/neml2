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

import torch

from neml2.es import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    ModelNonlinearSystem,
    SparseVector,
)
from neml2.models.model import Model
from neml2.solvers import DenseLU, Newton, NewtonWithLineSearch, RetCode, SchurComplement
from neml2.types import SR2, Scalar, Tensor


class ScalarResidual(Model):
    input_spec = {"x": Scalar, "c": Scalar}
    output_spec = {"x_residual": Scalar}

    def forward(self, x: Scalar, c: Scalar, v=None):
        r = x * x - c
        if v is None:
            return r
        return r, self.apply_chain_rule(
            v,
            "x_residual",
            {
                "x": lambda V: 2.0 * x * V,
                "c": lambda V: -V,
            },
            output=r,
        )


def test_dense_lu_solves_batched_vector_rhs():
    layout = AxisLayout([["x"]], {"x": Scalar})
    A = AssembledMatrix(
        layout,
        layout,
        [[Tensor(torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float64), batch_ndim=1)]],
    )
    b = AssembledVector(
        layout,
        [Tensor(torch.tensor([[6.0], [20.0]], dtype=torch.float64), batch_ndim=1)],
    )

    x = DenseLU().solve(A, b)

    assert isinstance(x, AssembledVector)
    assert torch.equal(x.tensors[0].data, torch.tensor([[3.0], [5.0]], dtype=torch.float64))


def test_dense_lu_solves_batched_matrix_rhs():
    layout = AxisLayout([["x"]], {"x": Scalar})
    rhs_layout = AxisLayout([["c"]], {"c": Scalar})
    A = AssembledMatrix(
        layout,
        layout,
        [[Tensor(torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float64), batch_ndim=1)]],
    )
    B = AssembledMatrix(
        layout,
        rhs_layout,
        [[Tensor(torch.tensor([[[6.0]], [[20.0]]], dtype=torch.float64), batch_ndim=1)]],
    )

    X = DenseLU().solve(A, B)

    assert isinstance(X, AssembledMatrix)
    assert torch.equal(X.tensors[0][0].data, torch.tensor([[[3.0]], [[5.0]]], dtype=torch.float64))


def test_newton_converges_batched_scalar_system():
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = Newton(atol=1e-12, rtol=1e-12, miters=20).solve(sys)
    solved = sys.u().disassemble()["x"]

    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_converges_same_as_newton():
    """Linesearch should converge the well-behaved x^2 - c = 0 case identically."""
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = NewtonWithLineSearch(atol=1e-12, rtol=1e-12, miters=20).solve(sys)
    solved = sys.u().disassemble()["x"]

    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_strong_wolfe_variant():
    """The STRONG_WOLFE step rule should also converge the same problem."""
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = NewtonWithLineSearch(
        atol=1e-12, rtol=1e-12, miters=20, linesearch_type="STRONG_WOLFE"
    ).solve(sys)
    solved = sys.u().disassemble()["x"]
    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_rejects_bad_linesearch_type():
    import pytest

    with pytest.raises(ValueError, match="linesearch_type"):
        NewtonWithLineSearch(linesearch_type="HERMES")


def _scalar_system():
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )
    return sys


def test_newton_quiet_by_default(capsys):
    """By default (``NEML2_LOGS`` unset, the ``newton`` channel at the built-in
    ``warning`` level) the solver emits nothing and returns no log data."""
    from neml2 import log

    log.reset_defaults()
    result = Newton(atol=1e-12, rtol=1e-12, miters=20).solve(_scalar_system())
    assert result.ret is RetCode.SUCCESS
    assert result.log == ()
    captured = capsys.readouterr()
    assert "[neml2:newton" not in captured.out
    assert "[neml2:newton" not in captured.err


def test_newton_debug_emits_convergence_log(capsys):
    """``NEML2_LOGS=newton=debug`` emits one ``[neml2:newton] ITERATION`` line per
    step -- from the shared C++ Newton loop, through the log store -- with a
    monotonically non-increasing residual norm, bracketed by begin/end banners."""
    import re

    from neml2 import log

    log.set_default_level("newton", "debug")
    try:
        result = Newton(atol=1e-12, rtol=1e-12, miters=20).solve(_scalar_system())
    finally:
        log.reset_defaults()

    assert result.ret is RetCode.SUCCESS
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "---- begin newton solve ----" in text
    assert "---- end newton solve ----" in text  # clean separator, no reason mangled in
    assert "reason=" in text  # the reason rides its own summary line
    iter_lines = [ln for ln in text.splitlines() if "ITERATION" in ln]
    assert len(iter_lines) >= 2
    assert all(ln.startswith("[neml2:newton") for ln in iter_lines)
    norms = []
    for ln in iter_lines:
        m = re.search(r"\|R\| = ([\d.eE+-]+)", ln)
        assert m is not None
        norms.append(float(m.group(1)))
    assert all(a >= b for a, b in zip(norms, norms[1:], strict=False))  # non-increasing
    assert norms[-1] < norms[0]  # actually made progress


def test_model_channel_logs_kstate_debug(capsys):
    """``model=debug`` dumps the K-state / shape metadata during Jacobian
    assembly (``es/system``: ``_call_model`` + ``_assemble_matrix``) -- a niche
    developer diagnostic, silent at every other level."""
    from neml2 import log

    system = _scalar_system()
    log.set_default_level("model", "debug")
    try:
        system.assemble(need_A=True, need_B=True, need_b=True)
    finally:
        log.reset_defaults()
    text = "".join(capsys.readouterr())
    assert "[neml2:model" in text
    assert "_call_model" in text
    assert "_assemble_matrix" in text


# --------------------------------------------------------------------------- #
# AssembledMatrix.per_instance_matvec (grain-diagonal matvec)                  #
# --------------------------------------------------------------------------- #


def test_per_instance_matvec_single_dense_matches_matmul():
    """For a single DENSE group there is no site axis, so the per-instance matvec
    equals both the plain ``@`` and a raw batched matmul."""
    torch.manual_seed(0)
    lay = AxisLayout([["a", "b"]], {"a": Scalar, "b": Scalar}, {}, ("dense",))
    nblk, batch, m = 3, 2, 2
    A = torch.randn(nblk, batch, m, m, dtype=torch.float64)
    am = AssembledMatrix(lay, lay, [[Tensor(A, batch_ndim=2, sub_batch_ndim=0)]])
    xv = torch.randn(nblk, batch, m, dtype=torch.float64)
    x = AssembledVector(lay, [Tensor(xv, batch_ndim=2, sub_batch_ndim=0)])

    out = am.per_instance_matvec(x)
    ref = (A @ xv.unsqueeze(-1)).squeeze(-1)
    assert out.tensors[0].sub_batch_ndim == 0
    assert torch.allclose(out.tensors[0].data, ref)
    # @ and per_instance agree when nothing is BLOCK.
    assert torch.allclose(out.tensors[0].data, (am @ x).tensors[0].data)


def test_per_instance_matvec_block_group_stays_per_site():
    """A BLOCK output keeps its site axis (grain-diagonal ``B @ v``), whereas ``@``
    aggregates over the contracted site axis."""
    torch.manual_seed(0)
    N, npp, nblk, batch = 4, 6, 3, 2
    lay = AxisLayout([["p"]], {"p": SR2}, {"p": torch.Size([N])}, ("block",))
    App = torch.randn(nblk, batch, N, npp, npp, dtype=torch.float64)
    am = AssembledMatrix(lay, lay, [[Tensor(App, batch_ndim=2, sub_batch_ndim=1)]])
    vp = torch.randn(nblk, batch, N, npp, dtype=torch.float64)
    x = AssembledVector(lay, [Tensor(vp, batch_ndim=2, sub_batch_ndim=1)])

    per_site = (App @ vp.unsqueeze(-1)).squeeze(-1)  # (nblk, batch, N, npp)
    out = am.per_instance_matvec(x)
    assert out.tensors[0].sub_batch_ndim == 1
    assert torch.allclose(out.tensors[0].data, per_site)
    # __matmul__ sums over the site axis -> dense (sub 0).
    agg = am @ x
    assert agg.tensors[0].sub_batch_ndim == 0
    assert torch.allclose(agg.tensors[0].data, per_site.sum(dim=2))


def test_per_instance_matvec_transpose_dense():
    torch.manual_seed(1)
    lay = AxisLayout([["a", "b"]], {"a": Scalar, "b": Scalar}, {}, ("dense",))
    A = torch.randn(3, 2, 2, 2, dtype=torch.float64)
    am = AssembledMatrix(lay, lay, [[Tensor(A, batch_ndim=2, sub_batch_ndim=0)]])
    xv = torch.randn(3, 2, 2, dtype=torch.float64)
    x = AssembledVector(lay, [Tensor(xv, batch_ndim=2, sub_batch_ndim=0)])

    out = am.per_instance_matvec(x, transpose=True)
    ref = (A.transpose(-1, -2) @ xv.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out.tensors[0].data, ref)


def _arrowhead(N=4, npp=6, nss=6, nblk=2, batch=2, seed=0):
    """SPD arrowhead: BLOCK(per-site) primary + DENSE(global) Schur block, and a rhs."""
    torch.manual_seed(seed)
    lay = AxisLayout(
        [["p"], ["s"]],
        {"p": SR2, "s": SR2},
        {"p": torch.Size([N]), "s": torch.Size([])},
        ("block", "dense"),
    )
    Ms = torch.randn(nblk, batch, N, npp, npp, dtype=torch.float64)
    App = Ms @ Ms.transpose(-1, -2) + (npp + 4.0) * torch.eye(npp, dtype=torch.float64)
    Md = torch.randn(nblk, batch, nss, nss, dtype=torch.float64)
    Ass = Md @ Md.transpose(-1, -2) + (nss + 4.0) * torch.eye(nss, dtype=torch.float64)
    Aps = 0.05 * torch.randn(nblk, batch, N, npp, nss, dtype=torch.float64)
    Asp = 0.05 * torch.randn(nblk, batch, N, nss, npp, dtype=torch.float64)
    tp = Tensor(App, batch_ndim=2, sub_batch_ndim=1)
    tps = Tensor(Aps, batch_ndim=2, sub_batch_ndim=1)
    tsp = Tensor(Asp, batch_ndim=2, sub_batch_ndim=1)
    tss = Tensor(Ass, batch_ndim=2, sub_batch_ndim=0)
    am = AssembledMatrix(lay, lay, [[tp, tps], [tsp, tss]])
    bp = torch.randn(nblk, batch, N, npp, dtype=torch.float64)
    bs = torch.randn(nblk, batch, nss, dtype=torch.float64)
    b = AssembledVector(
        lay,
        [Tensor(bp, batch_ndim=2, sub_batch_ndim=1), Tensor(bs, batch_ndim=2, sub_batch_ndim=0)],
    )
    return am, b


def test_per_instance_matvec_arrowhead_inverts_schur():
    """per_instance_matvec is the true arrowhead ``A @ x`` (BLOCK row kept per-site,
    DENSE row aggregating the BLOCK column), so it inverts the Schur solve."""
    am, b = _arrowhead()
    solver = SchurComplement(
        residual_primary_group=0, unknown_primary_group=0, primary_solver=DenseLU()
    )
    x = solver.solve(am, b)
    back = am.per_instance_matvec(x)
    assert torch.allclose(back.tensors[0].data, b.tensors[0].data, atol=1e-8)
    assert torch.allclose(back.tensors[1].data, b.tensors[1].data, atol=1e-8)


# --------------------------------------------------------------------------- #
# AssembledMatrix.transpose / AssembledMatrix.batch / AssembledVector.batch    #
# --------------------------------------------------------------------------- #


def test_assembled_matrix_transpose_roundtrip():
    torch.manual_seed(0)
    rl = AxisLayout([["a"], ["b"]], {"a": Scalar, "b": Scalar}, {}, ("dense", "dense"))
    cl = AxisLayout([["c"], ["d"]], {"c": Scalar, "d": Scalar}, {}, ("dense", "dense"))

    def blk(r, c):
        return Tensor(torch.randn(4, 2, r, c, dtype=torch.float64), batch_ndim=2)

    am = AssembledMatrix(rl, cl, [[blk(2, 3), blk(2, 5)], [blk(7, 3), blk(7, 5)]])
    amt = am.transpose()
    assert amt.row_layout == cl and amt.col_layout == rl
    assert tuple(amt.tensors[0][0].data.shape[-2:]) == (3, 2)
    # block (i, j) of the transpose is block (j, i) of the original, base-transposed
    assert torch.equal(amt.tensors[0][1].data, am.tensors[1][0].data.transpose(-1, -2))
    amtt = amt.transpose()
    for i in range(2):
        for j in range(2):
            assert torch.equal(amtt.tensors[i][j].data, am.tensors[i][j].data)


def test_assembled_matrix_transpose_rejects_two_intmd():
    import pytest  # noqa: PLC0415

    lay = AxisLayout([["a"]], {"a": Scalar}, {}, ("dense",))
    blk = Tensor(torch.zeros(4, 2, 3, 3, 2, 2, dtype=torch.float64), batch_ndim=2, sub_batch_ndim=2)
    am = AssembledMatrix(lay, lay, [[blk]])
    with pytest.raises(NotImplementedError):
        am.transpose()


def test_assembled_matrix_batch_select():
    torch.manual_seed(0)
    lay = AxisLayout([["a"]], {"a": Scalar}, {}, ("dense",))
    blk = Tensor(torch.randn(6, 2, 3, 3, dtype=torch.float64), batch_ndim=2)
    am = AssembledMatrix(lay, lay, [[blk]])
    sl = am.batch[1:4]
    assert sl.tensors[0][0].data.shape[0] == 3
    assert torch.equal(sl.tensors[0][0].data, blk.data[1:4])


def test_assembled_vector_batch_select():
    lay = AxisLayout([["a"]], {"a": Scalar}, {}, ("dense",))
    v = AssembledVector(lay, [Tensor(torch.randn(6, 2, 3, dtype=torch.float64), batch_ndim=2)])
    vs = v.batch[2:]
    assert vs.tensors[0].data.shape[0] == 4
    assert torch.equal(vs.tensors[0].data, v.tensors[0].data[2:])


# --------------------------------------------------------------------------- #
# AxisLayout flat DOF counts + AssembledVector.to_flat / from_flat            #
# --------------------------------------------------------------------------- #


def _block_dense_layout():
    return AxisLayout(
        [["p"], ["s"]],
        {"p": SR2, "s": SR2},
        {"p": torch.Size([4]), "s": torch.Size([])},
        ("block", "dense"),
    )


def test_axis_layout_flat_dof_counts():
    lay = _block_dense_layout()
    # intmd ndim: preserved site axis for BLOCK, folded (0) for DENSE.
    assert lay.group_intmd_ndim(0) == 1
    assert lay.group_intmd_ndim(1) == 0
    # flat size folds the sub-batch extent into the DOF count...
    assert lay.group_flat_size(0) == 4 * 6
    assert lay.group_flat_size(1) == 6
    assert lay.flat_size() == 4 * 6 + 6
    # ...whereas group_size / storage_size stay base-only (per site).
    assert lay.group_size(0) == 6
    assert lay.storage_size() == 12


def test_assembled_vector_to_flat_from_flat_roundtrip():
    torch.manual_seed(0)
    lay = _block_dense_layout()
    vp = torch.randn(3, 2, 4, 6, dtype=torch.float64)
    vs = torch.randn(3, 2, 6, dtype=torch.float64)
    av = AssembledVector(
        lay,
        [Tensor(vp, batch_ndim=2, sub_batch_ndim=1), Tensor(vs, batch_ndim=2, sub_batch_ndim=0)],
    )
    flat = av.to_flat()
    assert flat.shape == (3, 2, lay.flat_size())

    rt = AssembledVector.from_flat(lay, flat)
    assert torch.equal(rt.tensors[0].data, vp)
    assert rt.tensors[0].sub_batch_ndim == 1  # BLOCK site axis recovered
    assert torch.equal(rt.tensors[1].data, vs)
    assert rt.tensors[1].sub_batch_ndim == 0
