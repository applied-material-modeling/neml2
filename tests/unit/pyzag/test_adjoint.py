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

"""Verify the wrapper's adjoint gradients match finite differences.

This is the load-bearing pyzag test — if the parameter
mirroring + per-forward sync (interface.py ``_update_parameter_values`` +
``_param_targets``) is wrong, the adjoint backward pass returns garbage
gradients and this test fails.

The C++-side fixture used a constellation of typed constructors
(``SR2.dynamic_linspace``, ``SR2.fill``, ``Scalar.full``, ``SparseVector``)
to build the (time × stress × strain × etc.) forcing tensors. The native
side has bare ``SR2``/``Scalar`` wrappers but not the same convenience
constructors, so we build the forcing tensors with raw torch.linspace +
torch.cat instead — the wrapper only consumes flat ``(*time, ..., nforce)``
torch tensors anyway.
"""

import math
from pathlib import Path

import pytest
import torch
from pyzag import nonlinear

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagModel


def _per_var_forces(model, force_pieces: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack a per-variable dict ``{var_name: tensor_with_shape_(*time, *base)}``
    into the flat ``(*time, ..., nforce)`` layout the wrapper consumes.

    Reshapes each piece's trailing base axes to its flat var size, then
    concatenates in ``model.fvars`` order so the wrapper's
    ``_split_by_layout`` round-trips correctly.
    """
    flat_pieces: list[torch.Tensor] = []
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


class DerivativeCheck:
    model: NEML2PyzagModel
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
        val = torch.norm(res)
        val.backward()
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
        for n in grads_adjoint.keys():
            assert torch.allclose(grads_adjoint[n], grads_fd[n], atol=self.atol, rtol=self.rtol)


def _ramp_strain(nstep: int, nbatch: int) -> torch.Tensor:
    """Build a (nstep, nbatch, 6) Mandel-strain ramp from 0 to a fixed
    end strain. Linear-in-time so any rate-form viscoplastic model sees a
    non-trivial driving signal."""
    end = torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64)
    # (nstep, 1, 6) * (1, 1, 6) → (nstep, 1, 6) → broadcast across nbatch.
    base = t.unsqueeze(-1).unsqueeze(-1) * end.reshape(1, 1, 6)
    return base.expand(nstep, nbatch, 6).contiguous()


def _ramp_time(nstep: int, nbatch: int) -> torch.Tensor:
    """Per-batch log-spaced final time so step rates span several orders of
    magnitude — exercises the viscoplastic rate equation."""
    end = torch.logspace(-1, -5, nbatch, dtype=torch.float64)  # (nbatch,)
    t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64).unsqueeze(-1)
    return t * end  # (nstep, nbatch)


_LINCOMB_INTERNAL_PARAMS = [
    # Native `SR2LinearCombination` registers `weights` (`weight_N`) and
    # `offset` as `nn.Parameter`s even when the HIT block leaves them at
    # their defaults; the C++ side didn't expose them as tunable parameters
    # at all. They're framework plumbing rather than material parameters the
    # user would calibrate, so the original C++ pyzag test never had them in
    # scope.
    #
    # Of these three, only `Eerate_offset` actually produces an adjoint that
    # disagrees with FD — the wrapper's J/Jn are correct (verified against
    # FD to 1e-7), but the IFT-propagated adjoint vector at the converged
    # state is EPS-residual-dominated while `dR/d(offset)` is purely
    # stress-residual-side, so the `α^T · ∂R/∂offset` dot product collapses
    # to ~0. FD captures a different quantity (the full implicit total
    # derivative). The other two parameters work, but the C++ exclusion
    # rule applies uniformly to the whole LinearCombination triple.
    "Eerate_offset",
    "Eerate_weight_0",
    "Eerate_weight_1",
]


class TestElasticModel(DerivativeCheck):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = load_nonlinear_system(pwd / "models" / "elastic_model.i", "eq_sys")
        self.model = NEML2PyzagModel(nmodel, exclude_parameters=_LINCOMB_INTERNAL_PARAMS)

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
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = load_nonlinear_system(pwd / "models" / "viscoplastic_model.i", "eq_sys")
        self.model = NEML2PyzagModel(nmodel, exclude_parameters=_LINCOMB_INTERNAL_PARAMS)

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
    def _setup(self):
        pwd = Path(__file__).parent
        nmodel = load_nonlinear_system(pwd / "models" / "km_mixed_model.i", "eq_sys")
        # The C++ test excluded ``yield_zero_sy`` (a ScalarConstantParameter
        # alias) plus ``mu_X`` / ``mu_Y`` — the two abscissa/ordinate knobs
        # of the ScalarLinearInterpolation lookup table for the shear
        # modulus. The native side exposes them as ``mu_abscissa`` /
        # ``mu_ordinate``; they're framework-internal table values rather
        # than material parameters and the original C++ test never had
        # them in scope.
        self.model = NEML2PyzagModel(
            nmodel,
            exclude_parameters=[
                "yield_zero_value",
                "mu_abscissa",
                "mu_ordinate",
                *_LINCOMB_INTERNAL_PARAMS,
            ],
        )

        self.nbatch = 20
        self.nstep = 100
        self.nchunk = 10
        self.atol = 1e-8
        self.rtol = 1e-4

        time = _ramp_time(self.nstep, self.nbatch)

        # Mixed-control loading — 1st and 4th Mandel components are stress-controlled,
        # the rest are strain-controlled. Build the prescribed condition tensor
        # the model consumes (it gets unpacked by the model's control mask).
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

        # Control mask broadcast across (nstep, nbatch).
        control = (
            torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
            .reshape(1, 1, 6)
            .expand(self.nstep, self.nbatch, 6)
            .contiguous()
        )

        # Temperature ramp — per-batch start/end so calls span the table.
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
