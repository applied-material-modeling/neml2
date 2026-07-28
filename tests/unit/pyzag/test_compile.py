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

"""``neml2.compile`` must be transparent to the pyzag pipeline: a compiled full
solve reproduces the gold trajectory, and the adjoint gradients match eager (the
AOTAutograd donated-buffer / retain-graph interaction that neml2 cannot test on
its own).

Generic ``torch.compile``-vs-eager numerical parity for a single model call is
neml2's own responsibility and is deliberately not re-checked here."""

from __future__ import annotations

import pytest
import torch
from pyzag import chunktime, nonlinear

import neml2
from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagFactory

_ALL_MODELS = ["elastic_model", "viscoplastic_model", "km_mixed_model"]


def _gold_state_forces(gold_dir, pmodel, input_name):
    ref = torch.load(str(gold_dir / f"{input_name}.pt"), weights_only=True)
    forces = torch.cat([ref["input"][v] for v in pmodel.fvars], -1)
    state = torch.cat([ref["output"][v] for v in pmodel.svars], -1)
    return state, forces


def _solve(pmodel, state, forces) -> torch.Tensor:
    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
    )
    with torch.no_grad():
        return nonlinear.solve(solver, state[0], forces.shape[0], forces)


def _is_compiled(model) -> bool:
    return getattr(model, "_neml2_compiled_forward", False)


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compiled_forward_solve_matches_gold(models_dir, gold_dir, compare_tolerances, input_name):
    """A full pyzag solve through a ``compile=True`` factory reproduces the gold trajectory.

    Exercises ``NEML2PyzagFactory._compile`` (dynamo cache config + in-place
    ``neml2.compile``) and the compiled residual driving the chunked Newton
    solver end to end -- integration that neml2's own compile tests do not cover.
    """
    nmodel = load_nonlinear_system(models_dir / f"{input_name}.i", "eq_sys")
    compiled = NEML2PyzagFactory(nmodel, compile=True)
    assert _is_compiled(compiled.sys.model)

    state, forces = _gold_state_forces(gold_dir, compiled, input_name)
    compiled_traj = _solve(compiled, state, forces)

    tol = compare_tolerances.get(input_name, {})
    assert torch.allclose(state, compiled_traj, **tol)  # pyright: ignore[reportArgumentType]


def test_adjoint_matches_eager_under_compile(models_dir):
    """solve_adjoint + backward through a compiled residual gives the eager gradients.

    Guards the AOTAutograd donated-buffer interaction with pyzag's retain-graph
    adjoint -- a failure mode outside the scope of neml2's own compile tests.
    """

    def run(compile_it):
        nmodel = load_nonlinear_system(models_dir / "viscoplastic_model.i", "eq_sys")
        pmodel = NEML2PyzagFactory(nmodel, compile=False)
        if compile_it:
            neml2.compile(pmodel)
        nstep, nbatch = 20, 4
        end = torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64)
        strain = (t.reshape(-1, 1, 1) * end.reshape(1, 1, 6)).expand(nstep, nbatch, 6).contiguous()
        time = (t.unsqueeze(-1) * torch.logspace(-1, -3, nbatch, dtype=torch.float64)).contiguous()
        pieces = {"t": time.unsqueeze(-1), "strain": strain}
        forces = torch.cat([pieces[v] for v in pmodel.fvars], dim=-1)
        state0 = torch.zeros((nbatch, pmodel.nstate), dtype=torch.float64)

        solver = nonlinear.RecursiveNonlinearEquationSolver(
            pmodel,
            step_generator=nonlinear.StepGenerator(10),
            predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        )
        solver.zero_grad()
        res = nonlinear.solve_adjoint(solver, state0, nstep, forces)
        torch.norm(res).backward()
        return {n: p.grad.clone() for n, p in solver.named_parameters() if p.grad is not None}

    eager_grads = run(False)
    compiled_grads = run(True)
    assert eager_grads.keys() == compiled_grads.keys()
    assert eager_grads
    for name in eager_grads:
        assert torch.allclose(eager_grads[name], compiled_grads[name], rtol=1e-6, atol=1e-8), (
            f"gradient mismatch for {name}"
        )
