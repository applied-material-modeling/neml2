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

"""Tests for the lifted-arrowhead PCR factorization (neml2 pyzag backend).

Structure: (1) lift algebra unit checks, (2) backend PCR-vs-Thomas on synthetic
2-group / single-group-BLOCK systems, (3) end-to-end Taylor forward + adjoint parity.
"""

from pathlib import Path

import pytest
import torch
from pyzag import chunktime, nonlinear

from neml2 import load_nonlinear_system
from neml2.es import AssembledMatrix
from neml2.es.axis_layout import AxisLayout
from neml2.pyzag import NEML2PyzagModel
from neml2.pyzag.operators import NEML2LiftedPCRFactorization, NEML2SolvableBlockOperator
from neml2.pyzag.operators._lifted_pcr import (
    _DiagFactors,
    compose,
    lift_site_diagonal,
    neg,
)
from neml2.pyzag.operators._vector import NEML2BlockVector
from neml2.types import SR2, Tensor

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# lift algebra helpers (test-local dense reconstruction)                       #
# --------------------------------------------------------------------------- #
def _schur_dense(T, N, np_):
    """Reconstruct the dense (N*np, N*np) primary operator D + U W V."""
    D, U, W, V = T["D"], T["U"], T["W"], T["V"]
    M = torch.zeros(N * np_, N * np_)
    for i in range(N):
        M[i * np_ : (i + 1) * np_, i * np_ : (i + 1) * np_] = D[i]
    q = W.shape[-1]
    if q > 0:
        M = M + U.reshape(N * np_, q) @ W @ V.permute(1, 0, 2).reshape(q, N * np_)
    return M


def _rand_lift(N, np_, q):
    return dict(
        D=torch.stack([torch.randn(np_, np_) for _ in range(N)]),
        U=torch.randn(N, np_, q),
        W=torch.randn(q, q),
        V=torch.randn(N, q, np_),
    )


def test_compose_closure():
    N, np_ = 4, 3
    T1, T2 = _rand_lift(N, np_, 2), _rand_lift(N, np_, 3)
    lhs = _schur_dense(compose(T1, T2), N, np_)
    rhs = _schur_dense(T1, N, np_) @ _schur_dense(T2, N, np_)
    assert torch.allclose(lhs, rhs, atol=1e-10)
    assert compose(T1, T2)["W"].shape[-1] == 2 + 3


def test_one_hop_fold_matches_dense():
    N, np_, ns = 4, 3, 2
    torch.manual_seed(3)

    def spd(n):
        M = torch.randn(n, n)
        return M @ M.T + n * torch.eye(n)

    App = torch.stack([spd(np_) + np_ * torch.eye(np_) for _ in range(N)])
    Aps = 0.3 * torch.randn(N, np_, ns)
    Asp = 0.3 * torch.randn(N, ns, np_)
    Ass = spd(ns) + ns * torch.eye(ns)
    fac = _DiagFactors(App, Aps, Asp, Ass, ns)

    Bl = torch.stack([0.5 * torch.randn(np_, np_) for _ in range(N)])
    Br = torch.stack([0.5 * torch.randn(np_, np_) for _ in range(N)])

    # dense reference:  -Bl (A^-1)pp Br
    Nf = N * np_ + ns
    Ad = torch.zeros(Nf, Nf)
    for i in range(N):
        s = slice(i * np_, (i + 1) * np_)
        Ad[s, s] = App[i]
        Ad[s, N * np_ :] = Aps[i]
        Ad[N * np_ :, s] = Asp[i]
    Ad[N * np_ :, N * np_ :] = Ass
    Ainv_pp = torch.linalg.inv(Ad)[: N * np_, : N * np_]
    Bl_d = torch.block_diag(*Bl)
    Br_d = torch.block_diag(*Br)
    ref = -(Bl_d @ Ainv_pp @ Br_d)

    T = neg(compose(compose(lift_site_diagonal(Bl), fac.Ahat), lift_site_diagonal(Br)))
    assert torch.allclose(_schur_dense(T, N, np_), ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# backend PCR vs Thomas on synthetic systems                                   #
# --------------------------------------------------------------------------- #
def _two_group_ops(N, b, nblk, np_, ns, seed):
    torch.manual_seed(seed)
    lay = AxisLayout(
        [["p"], ["s"]],
        {"p": SR2, "s": SR2},
        {"p": torch.Size([N]), "s": torch.Size([])},
        ("block", "dense"),
    )

    def spd_site(n):
        M = torch.randn(nblk, b, N, n, n)
        return M @ M.transpose(-1, -2) + (n + 4.0) * torch.eye(n)

    App = spd_site(np_)
    Aps = 0.2 * torch.randn(nblk, b, N, np_, ns)
    Asp = 0.2 * torch.randn(nblk, b, N, ns, np_)
    Ass = torch.randn(nblk, b, ns, ns)
    Ass = Ass @ Ass.transpose(-1, -2) + (ns + 4.0) * torch.eye(ns)
    A = NEML2SolvableBlockOperator(
        AssembledMatrix(
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
    )
    m = nblk - 1
    Bpp = 0.3 * torch.randn(m, b, N, np_, np_)
    z_ps = torch.zeros(m, b, N, np_, ns)
    z_sp = torch.zeros(m, b, N, ns, np_)
    z_ss = torch.zeros(m, b, ns, ns)
    B = NEML2SolvableBlockOperator(
        AssembledMatrix(
            lay,
            lay,
            [
                [
                    Tensor(Bpp, batch_ndim=2, sub_batch_ndim=1),
                    Tensor(z_ps, batch_ndim=2, sub_batch_ndim=1),
                ],
                [
                    Tensor(z_sp, batch_ndim=2, sub_batch_ndim=1),
                    Tensor(z_ss, batch_ndim=2, sub_batch_ndim=0),
                ],
            ],
        )
    )
    v = NEML2BlockVector([torch.randn(nblk, b, N, np_), torch.randn(nblk, b, ns)], lay, [1, 0])
    return A, B, v


@pytest.mark.parametrize("nblk", [2, 4, 6, 7, 8])
def test_lifted_pcr_matches_thomas_two_group(nblk):
    A, B, v = _two_group_ops(N=5, b=2, nblk=nblk, np_=6, ns=6, seed=nblk)
    xt = chunktime.BidiagonalThomasFactorization(A, B).matvec(v.clone())
    xp = NEML2LiftedPCRFactorization(A, B).matvec(v.clone())
    assert isinstance(xt, NEML2BlockVector) and isinstance(xp, NEML2BlockVector)
    for gt, gp in zip(xt.raw_tensors, xp.raw_tensors, strict=True):
        assert torch.allclose(gt, gp, atol=1e-9)


@pytest.mark.parametrize("nblk", [4, 7, 8])
def test_lifted_pcr_matches_thomas_single_group_block(nblk):
    torch.manual_seed(nblk)
    N, b, np_ = 5, 2, 6
    lay = AxisLayout([["p"]], {"p": SR2}, {"p": torch.Size([N])}, ("block",))
    M = torch.randn(nblk, b, N, np_, np_)
    App = M @ M.transpose(-1, -2) + (np_ + 4.0) * torch.eye(np_)
    A = NEML2SolvableBlockOperator(
        AssembledMatrix(lay, lay, [[Tensor(App, batch_ndim=2, sub_batch_ndim=1)]])
    )
    Bpp = 0.3 * torch.randn(nblk - 1, b, N, np_, np_)
    B = NEML2SolvableBlockOperator(
        AssembledMatrix(lay, lay, [[Tensor(Bpp, batch_ndim=2, sub_batch_ndim=1)]])
    )
    v = NEML2BlockVector([torch.randn(nblk, b, N, np_)], lay, [1])
    xt = chunktime.BidiagonalThomasFactorization(A, B).matvec(v.clone())
    xp = NEML2LiftedPCRFactorization(A, B).matvec(v.clone())
    assert isinstance(xt, NEML2BlockVector) and isinstance(xp, NEML2BlockVector)
    assert torch.allclose(xt.raw_tensors[0], xp.raw_tensors[0], atol=1e-9)


# --------------------------------------------------------------------------- #
# end-to-end: Taylor polycrystal, PCR vs Thomas (forward + adjoint)            #
# --------------------------------------------------------------------------- #
_TAYLOR_MODEL = (
    Path(__file__).parents[3]
    / "tests"
    / "regression"
    / "solid_mechanics"
    / "crystal_plasticity"
    / "taylor"
    / "model.i"
)

_ORIENTATIONS = torch.tensor(
    [
        [-0.269981, -0.299844, -0.86408],
        [0.209546, 0.192014, 0.514051],
        [-0.0251234, -0.0175916, -0.636644],
        [-0.146257, -0.0475218, -0.970804],
        [-0.174458, -0.302169, -0.523373],
    ],
    dtype=torch.float64,
)

_CALIBRATION_PARAMS = ["slip_strength_constant_strength", "voce_hardening_initial_slope"]


def _build_taylor(include_parameters=None, ntime=12, nbatch=1):
    nsys = load_nonlinear_system(str(_TAYLOR_MODEL), "eq_sys")
    factory = NEML2PyzagModel(nsys, include_parameters=include_parameters)
    ngrains = _ORIENTATIONS.shape[0]
    ic = {
        "elastic_strain": torch.zeros(nbatch, ngrains, 6),
        "orientation": _ORIENTATIONS.unsqueeze(0).expand(nbatch, ngrains, 3).contiguous(),
        "slip_hardening": torch.zeros(nbatch, ngrains),
        "deformation_rate": torch.zeros(nbatch, 6),
        "target_cauchy_stress": torch.zeros(nbatch, 6),
    }
    y0 = factory.assemble_state(ic, dynamic_dim=1)
    control = torch.zeros(ntime, nbatch, 6)
    control[..., 0] = 1.0
    prescribed = torch.zeros(ntime, nbatch, 6)
    prescribed[..., 0] = 1e-4
    times = torch.linspace(0.0, 50.0, ntime).reshape(ntime, 1).expand(ntime, nbatch).contiguous()
    vorticity = torch.zeros(ntime, nbatch, 3)
    forces = factory.assemble_forces(
        {"control": control, "prescribed": prescribed, "t": times, "vorticity": vorticity},
        dynamic_dim=2,
    )
    return factory, y0, ntime, forces


def _solver(factory, nchunk, dso):
    return nonlinear.RecursiveNonlinearEquationSolver(
        factory,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        direct_solve_operator=dso,
        nonlinear_solver=chunktime.ChunkNewtonRaphsonLineSearch(
            rtol=1e-8, atol=1e-10, miter=200, linesearch_iter=5
        ),
    )


@pytest.mark.parametrize("nchunk", [1, 2, 4])
def test_taylor_pcr_forward_matches_thomas(nchunk):
    factory, y0, nstep, forces = _build_taylor()
    with torch.no_grad():
        xt = nonlinear.solve(
            _solver(factory, nchunk, chunktime.BidiagonalThomasFactorization), y0, nstep, forces
        )
        xp = nonlinear.solve(
            _solver(factory, nchunk, chunktime.BidiagonalPCRFactorization), y0, nstep, forces
        )
    assert torch.allclose(xt, xp, atol=1e-8)


@pytest.mark.parametrize("nchunk", [1, 2, 4])
def test_taylor_pcr_adjoint_matches_thomas(nchunk):
    def grads(dso):
        factory, y0, nstep, forces = _build_taylor(include_parameters=_CALIBRATION_PARAMS)
        solver = _solver(factory, nchunk, dso)
        solver.zero_grad()
        torch.norm(nonlinear.solve_adjoint(solver, y0, nstep, forces)).backward()
        return {n: p.grad.clone() for n, p in solver.named_parameters() if p.grad is not None}

    gt = grads(chunktime.BidiagonalThomasFactorization)
    gp = grads(chunktime.BidiagonalPCRFactorization)
    assert gt.keys() == gp.keys()
    for k in gt:
        assert torch.allclose(gt[k], gp[k], atol=1e-7, rtol=1e-5)
