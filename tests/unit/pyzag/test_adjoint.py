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

"""Adjoint gradients of the factory must match finite differences -- the load-bearing
check on parameter mirroring and per-forward sync (``_update_parameter_values`` /
``_param_targets``)."""

import math

import pytest
import torch
from pyzag import nonlinear

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagFactory


def _per_var_forces(model, force_pieces):
    """Pack ``{var: (*time, *base)}`` into the flat ``(*time, ..., nforce)`` layout."""
    flat_pieces = []
    for name in model.fvars:
        piece = force_pieces[name]
        type_cls = model.flayout.type_of(name)
        var_size = model.flayout.var_size(name)
        if type_cls.BASE_NDIM == 0:
            piece = piece.unsqueeze(-1)
        elif type_cls.BASE_NDIM > 1:
            piece = piece.reshape(*piece.shape[: -type_cls.BASE_NDIM], var_size)
        flat_pieces.append(piece)
    return torch.cat(flat_pieces, dim=-1)


def _ramp_strain(nstep, nbatch):
    """A (nstep, nbatch, 6) Mandel-strain ramp from zero to a fixed end strain."""
    end = torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64)
    base = t.unsqueeze(-1).unsqueeze(-1) * end.reshape(1, 1, 6)
    return base.expand(nstep, nbatch, 6).contiguous()


def _ramp_time(nstep, nbatch):
    """Per-batch log-spaced final time so step rates span several orders of magnitude."""
    end = torch.logspace(-1, -5, nbatch, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64).unsqueeze(-1)
    return t * end


class DerivativeCheck:
    """Base for adjoint-vs-FD checks; subclasses populate the attributes in ``_setup``."""

    model: NEML2PyzagFactory
    nchunk: int
    initial_state: torch.Tensor
    nstep: int
    forces: torch.Tensor
    atol: float
    rtol: float

    def adjoint_grads(self):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        )
        solver.zero_grad()
        res = nonlinear.solve_adjoint(solver, self.initial_state, self.nstep, self.forces)
        torch.norm(res).backward()
        return {n: torch.Tensor(p.grad) for n, p in solver.named_parameters()}

    def fd_grads(self, eps=1.0e-6):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        )
        res = {}
        with torch.no_grad():
            val0 = torch.norm(nonlinear.solve(solver, self.initial_state, self.nstep, self.forces))
            for n, p in solver.named_parameters():
                p0 = p.clone()
                dx = torch.max(torch.abs(p0) * eps, torch.full_like(p0, eps))
                p.data = p0 + dx
                val1 = torch.norm(
                    nonlinear.solve(solver, self.initial_state, self.nstep, self.forces)
                )
                res[n] = (val1 - val0) / dx
                p.data = p0
        return res

    def test_adjoint_vs_fd(self):
        grads_adjoint = self.adjoint_grads()
        grads_fd = self.fd_grads()
        assert grads_adjoint.keys() == grads_fd.keys()
        for n in grads_adjoint:
            assert torch.allclose(grads_adjoint[n], grads_fd[n], atol=self.atol, rtol=self.rtol)


class TestElasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self, models_dir, lincomb_internal_params):
        nmodel = load_nonlinear_system(models_dir / "elastic_model.i", "eq_sys")
        self.model = NEML2PyzagFactory(
            nmodel, exclude_parameters=lincomb_internal_params, compile=False
        )

        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-5

        time = _ramp_time(self.nstep, self.nbatch)
        strain = _ramp_strain(self.nstep, self.nbatch)
        self.forces = _per_var_forces(self.model, {"t": time, "strain": strain})
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate), dtype=torch.float64)


class TestViscoplasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self, models_dir, lincomb_internal_params):
        nmodel = load_nonlinear_system(models_dir / "viscoplastic_model.i", "eq_sys")
        self.model = NEML2PyzagFactory(
            nmodel, exclude_parameters=lincomb_internal_params, compile=False
        )

        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-5

        time = _ramp_time(self.nstep, self.nbatch)
        strain = _ramp_strain(self.nstep, self.nbatch)
        self.forces = _per_var_forces(self.model, {"t": time, "strain": strain})
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate), dtype=torch.float64)


class TestKocksMeckingMixedControlModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self, models_dir, lincomb_internal_params):
        nmodel = load_nonlinear_system(models_dir / "km_mixed_model.i", "eq_sys")
        self.model = NEML2PyzagFactory(
            nmodel,
            exclude_parameters=[
                "yield_zero_value",
                "mu_abscissa",
                "mu_ordinate",
                *lincomb_internal_params,
            ],
            compile=False,
        )

        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-4

        time = _ramp_time(self.nstep, self.nbatch)

        sqrt2 = math.sqrt(2)
        end_condition = torch.tensor(
            [0.1, -50.0, -0.025, 0.15 / sqrt2, 75.0 / sqrt2, 0.05 / sqrt2],
            dtype=torch.float64,
        )
        t = torch.linspace(0.0, 1.0, self.nstep, dtype=torch.float64)
        condition = (
            (t.unsqueeze(-1).unsqueeze(-1) * end_condition.reshape(1, 1, 6))
            .expand(self.nstep, self.nbatch, 6)
            .contiguous()
        )

        control = (
            torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
            .reshape(1, 1, 6)
            .expand(self.nstep, self.nbatch, 6)
            .contiguous()
        )

        t_start = torch.linspace(300.0, 500.0, self.nbatch, dtype=torch.float64)
        t_end = torch.linspace(600.0, 1200.0, self.nbatch, dtype=torch.float64)
        temperature = t.unsqueeze(-1) * (t_end - t_start) + t_start

        self.forces = _per_var_forces(
            self.model,
            {
                "t": time,
                "control": control,
                "mixed_control": condition,
                "temperature": temperature,
            },
        )
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate), dtype=torch.float64)
