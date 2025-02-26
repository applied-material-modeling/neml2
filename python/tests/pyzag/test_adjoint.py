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

"""Test adjoint correctness"""

import pytest
from pathlib import Path
import torch
import neml2
from neml2.tensors import Scalar, SR2, Tensor
from pyzag import nonlinear
import math


class DerivativeCheck:
    def adjoint_grads(self):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        solver.zero_grad()
        res = nonlinear.solve_adjoint(solver, self.initial_state, self.nstep, self.forces)
        val = torch.norm(res)
        val.backward()
        return {n: p.grad for n, p in solver.named_parameters()}

    def fd_grads(self, eps=1.0e-6):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        res = {}
        with torch.no_grad():
            val0 = torch.norm(nonlinear.solve(solver, self.initial_state, self.nstep, self.forces))
            for n, p in solver.named_parameters():
                p0 = p.clone()
                dx = torch.abs(p0) * eps
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
        for n in grads_adjoint.keys():
            assert torch.allclose(grads_adjoint[n], grads_fd[n], atol=self.atol, rtol=self.rtol)


class TestElasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = neml2.reload_model(pwd / "models" / "elastic_model.i", "implicit_rate")
        self.model = neml2.pyzag.NEML2PyzagModel(nmodel)

        # Test configuration
        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-5

        # Prescribed time
        start_time = Scalar.full(0.0)
        end_time = Scalar(torch.logspace(-1, -5, self.nbatch))
        time = Scalar.linspace(start_time, end_time, self.nstep)

        # Prescribed strain
        start_strain = SR2.full(0.0)
        end_strain = SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0)
        strain = SR2.linspace(start_strain, end_strain, self.nstep).batch.unsqueeze(-1)

        # Prescribed forces
        self.forces = self.model.forces_asm.assemble_by_variable(
            {"forces/t": time, "forces/E": strain}
        ).torch()

        # Initial state
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate))


class TestViscoplasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = neml2.reload_model(pwd / "models" / "viscoplastic_model.i", "implicit_rate")
        self.model = neml2.pyzag.NEML2PyzagModel(nmodel)

        # Test configuration
        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-5

        # Prescribed time
        start_time = Scalar.full(0.0)
        end_time = Scalar(torch.logspace(-1, -5, self.nbatch))
        time = Scalar.linspace(start_time, end_time, self.nstep)

        # Prescribed strain
        start_strain = SR2.full(0.0)
        end_strain = SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0)
        strain = SR2.linspace(start_strain, end_strain, self.nstep).batch.unsqueeze(-1)

        # Prescribed forces
        self.forces = self.model.forces_asm.assemble_by_variable(
            {"forces/t": time, "forces/E": strain}
        ).torch()

        # Initial state
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate))


class TestKocksMeckingMixedControlModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = neml2.reload_model(pwd / "models" / "km_mixed_model.i", "implicit_rate")
        self.model = neml2.pyzag.NEML2PyzagModel(
            nmodel, exclude_parameters=["yield_zero_sy", "mu_X", "mu_Y"]
        )

        # Test configuration
        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-4

        # Prescribed time
        start_time = Scalar.full(0.0)
        end_time = Scalar(torch.logspace(-1, -5, self.nbatch))
        time = Scalar.linspace(start_time, end_time, self.nstep)

        # Prescribed strain/stress
        sqrt2 = math.sqrt(2)  # For Mandel notation
        start_condition = SR2.full(0.0)
        end_condition = SR2.fill(0.1, -50, -0.025, 0.15 / sqrt2, 75.0 / sqrt2, 0.05 / sqrt2)
        condition = SR2.linspace(start_condition, end_condition, self.nstep).batch.unsqueeze(-1)

        # Prescribed control
        # 1st and 4th components are 1.0 (stress-controlled)
        control = Tensor(torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]), 0)

        # Prescribed temperature
        start_temperature = Scalar.linspace(Scalar.full(300), Scalar.full(500), self.nbatch)
        end_temperature = Scalar.linspace(Scalar.full(600), Scalar.full(1200), self.nbatch)
        temperature = Scalar.linspace(start_temperature, end_temperature, self.nstep)

        # Prescribed forces
        self.forces = self.model.forces_asm.assemble_by_variable(
            {
                "forces/t": time,
                "forces/control": control,
                "forces/fixed_values": condition,
                "forces/T": temperature,
            }
        ).torch()

        # Initial state
        self.initial_state = torch.zeros((self.nbatch, self.model.nstate))
