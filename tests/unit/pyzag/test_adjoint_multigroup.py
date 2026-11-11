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

"""Adjoint-vs-finite-difference check for the multi-group BLOCK+DENSE Taylor model,
extending ``test_adjoint.py`` to the crystal-plasticity Schur path."""

from pathlib import Path

import pytest
import torch
from pyzag import chunktime, nonlinear

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagModel

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

_CALIBRATION_PARAMS = [
    "slip_strength_constant_strength",
    "voce_hardening_initial_slope",
]


def _make_solver(factory, nchunk):
    return nonlinear.RecursiveNonlinearEquationSolver(
        factory,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        nonlinear_solver=chunktime.ChunkNewtonRaphsonLineSearch(
            rtol=1e-8, atol=1e-10, miter=200, linesearch_iter=5
        ),
    )


def _build_problem(ntime=12, nchunk=1, nbatch=1):
    nsys = load_nonlinear_system(str(_TAYLOR_MODEL), "eq_sys")
    factory = NEML2PyzagModel(nsys, include_parameters=_CALIBRATION_PARAMS)

    ngrains = _ORIENTATIONS.shape[0]
    device = torch.device("cpu")
    dtype = torch.float64

    ic_dict = {
        "elastic_strain": torch.zeros(nbatch, ngrains, 6, device=device, dtype=dtype),
        "orientation": _ORIENTATIONS.to(device, dtype)
        .unsqueeze(0)
        .expand(nbatch, ngrains, 3)
        .contiguous(),
        "slip_hardening": torch.zeros(nbatch, ngrains, device=device, dtype=dtype),
        "deformation_rate": torch.zeros(nbatch, 6, device=device, dtype=dtype),
        "target_cauchy_stress": torch.zeros(nbatch, 6, device=device, dtype=dtype),
    }
    y0 = factory.assemble_state(ic_dict, dynamic_dim=1)

    control = torch.zeros(ntime, nbatch, 6, device=device, dtype=dtype)
    control[..., 0] = 1.0
    prescribed = torch.zeros(ntime, nbatch, 6, device=device, dtype=dtype)
    prescribed[..., 0] = 1e-4
    times = (
        torch.linspace(0.0, 50.0, ntime, device=device, dtype=dtype)
        .reshape(ntime, 1)
        .expand(ntime, nbatch)
        .contiguous()
    )
    vorticity = torch.zeros(ntime, nbatch, 3, device=device, dtype=dtype)
    forces_dict = {
        "control": control,
        "prescribed": prescribed,
        "t": times,
        "vorticity": vorticity,
    }
    forces = factory.assemble_forces(forces_dict, dynamic_dim=2)

    return factory, y0, ntime, forces


def _adjoint_grads(factory, y0, nstep, forces, nchunk):
    solver = _make_solver(factory, nchunk)
    solver.zero_grad()
    res = nonlinear.solve_adjoint(solver, y0, nstep, forces)
    torch.norm(res).backward()
    return {n: torch.Tensor(p.grad) for n, p in solver.named_parameters()}


def _fd_grads(factory, y0, nstep, forces, nchunk, eps=1e-6):
    solver = _make_solver(factory, nchunk)
    res = {}
    with torch.no_grad():
        val0 = torch.norm(nonlinear.solve(solver, y0, nstep, forces))
        for n, p in solver.named_parameters():
            p0 = p.clone()
            dx = torch.max(torch.abs(p0) * eps, torch.full_like(p0, eps))
            p.data = p0 + dx
            val1 = torch.norm(nonlinear.solve(solver, y0, nstep, forces))
            res[n] = (val1 - val0) / dx
            p.data = p0
    return res


@pytest.mark.parametrize("nchunk", [1, 2, 4])
def test_multigroup_adjoint_vs_fd(nchunk):
    torch.manual_seed(42)
    factory, y0, nstep, forces = _build_problem(ntime=12, nchunk=nchunk)

    grads_adjoint = _adjoint_grads(factory, y0, nstep, forces, nchunk)
    grads_fd = _fd_grads(factory, y0, nstep, forces, nchunk)

    assert grads_adjoint.keys() == grads_fd.keys()
    for n in grads_adjoint:
        a = grads_adjoint[n]
        f = grads_fd[n]
        assert torch.allclose(a, f, atol=1e-6, rtol=1e-4), (
            f"param {n!r} (nchunk={nchunk}): adjoint={a.tolist()} "
            f"fd={f.tolist()} absdiff={(a - f).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# Temperature-dependent stiffness: CubicElasticityTensor(E, nu, G) with each
# coefficient a ScalarLinearInterpolation table over temperature, feeding
# GeneralElasticity's promoted ``elastic_stiffness_tensor``. Mirroring the E
# table ordinate as a pyzag parameter exercises the promoted-stiffness
# chain-rule action and the parameter adjoint through the interpolation
# table end to end.
# ---------------------------------------------------------------------------

_TAYLOR_TEMPERATURE_MODEL = _TAYLOR_MODEL.parents[1] / "taylor_temperature" / "model.i"

_TEMPERATURE_PARAMS = ["E_ordinate", "slip_strength_constant_strength"]


def _build_temperature_problem(ntime=12, nbatch=3):
    nsys = load_nonlinear_system(str(_TAYLOR_TEMPERATURE_MODEL), "eq_sys")
    factory = NEML2PyzagModel(nsys, include_parameters=_TEMPERATURE_PARAMS)

    ngrains = _ORIENTATIONS.shape[0]
    dtype = torch.float64
    ic_dict = {
        "elastic_strain": torch.zeros(nbatch, ngrains, 6, dtype=dtype),
        "orientation": _ORIENTATIONS.unsqueeze(0).expand(nbatch, ngrains, 3).contiguous(),
        "slip_hardening": torch.zeros(nbatch, ngrains, dtype=dtype),
        "deformation_rate": torch.zeros(nbatch, 6, dtype=dtype),
        "target_cauchy_stress": torch.zeros(nbatch, 6, dtype=dtype),
    }
    y0 = factory.assemble_state(ic_dict, dynamic_dim=1)

    control = torch.zeros(ntime, nbatch, 6, dtype=dtype)
    control[..., 0] = 1.0
    prescribed = torch.zeros(ntime, nbatch, 6, dtype=dtype)
    prescribed[..., 0] = 1e-4
    s = torch.linspace(0.0, 1.0, ntime, dtype=dtype).unsqueeze(-1)
    times = (50.0 * s).expand(ntime, nbatch).contiguous()
    # Distinct temperature ramps per batch column, spanning several table
    # segments so more than one E ordinate receives a gradient.
    t_start = torch.linspace(350.0, 950.0, nbatch, dtype=dtype)
    temperature = t_start + 200.0 * s
    forces_dict = {
        "control": control,
        "prescribed": prescribed,
        "t": times,
        "temperature": temperature,
        "vorticity": torch.zeros(ntime, nbatch, 3, dtype=dtype),
    }
    forces = factory.assemble_forces(forces_dict, dynamic_dim=2)
    return factory, y0, ntime, forces


def _fd_grads_elementwise(factory, y0, nstep, forces, nchunk, eps=1e-6):
    """Forward FD, perturbing one parameter element at a time
    (``_fd_grads`` above perturbs every element of a parameter at once, which
    is only valid for scalar parameters)."""
    solver = _make_solver(factory, nchunk)
    res = {}
    with torch.no_grad():
        val0 = torch.norm(nonlinear.solve(solver, y0, nstep, forces))
        for n, p in solver.named_parameters():
            p0 = p.clone()
            g = torch.zeros_like(p0).reshape(-1)
            for i in range(g.numel()):
                pert = p0.clone().reshape(-1)
                dx = max(abs(pert[i].item()) * eps, eps)
                pert[i] += dx
                p.data = pert.reshape(p0.shape)
                g[i] = (torch.norm(nonlinear.solve(solver, y0, nstep, forces)) - val0) / dx
            p.data = p0
            res[n] = g.reshape(p0.shape)
    return res


@pytest.mark.parametrize("nchunk", [1, 3])
def test_temperature_dependent_stiffness_adjoint_vs_fd(nchunk):
    factory, y0, nstep, forces = _build_temperature_problem()

    grads_adjoint = _adjoint_grads(factory, y0, nstep, forces, nchunk)
    grads_fd = _fd_grads_elementwise(factory, y0, nstep, forces, nchunk)

    assert grads_adjoint.keys() == grads_fd.keys()
    for n in grads_adjoint:
        a = grads_adjoint[n]
        f = grads_fd[n]
        assert torch.isfinite(a).all(), f"param {n!r}: non-finite adjoint gradient {a.tolist()}"
        assert torch.allclose(a, f, atol=1e-8, rtol=1e-3), (
            f"param {n!r} (nchunk={nchunk}): adjoint={a.tolist()} "
            f"fd={f.tolist()} absdiff={(a - f).abs().max().item():.3e}"
        )
    # Every table segment touched by the temperature ramps contributes.
    (gE,) = (g for n, g in grads_adjoint.items() if n.endswith("E_ordinate"))
    assert (gE.abs() > 0).sum() >= 3
