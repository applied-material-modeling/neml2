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

"""Tests for ``neml2.compile`` -- in-process ``torch.compile`` acceleration of the
pyzag residual/Jacobian evaluation.

The contract under test: compiling a ``NEML2PyzagModel`` (or its underlying
``ModelNonlinearSystem`` / residual ``Model``) is a transparent, in-place
acceleration -- the forward solve and the residual/Jacobian outputs must match the
eager computation (and the checked-in gold), the adjoint training path must keep
working, and the forward guard must stay armed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch._dynamo as dynamo
from pyzag import chunktime, nonlinear

import neml2
from neml2 import load_nonlinear_system
from neml2.es import ModelNonlinearSystem
from neml2.models.model import Model
from neml2.pyzag import NEML2PyzagModel

_HERE = Path(__file__).parent

# Mirror test_NEML2PyzagModel.test_compare's per-scenario tolerances: the gold
# trajectories were baked by the retired C++ pipeline, so float op-order drift is
# expected.
_COMPARE_TOLERANCES = {
    "km_mixed_model": {"rtol": 1e-4, "atol": 1e-8},
}

_ALL_MODELS = ["elastic_model", "viscoplastic_model", "km_mixed_model"]

# Eager-vs-compiled parity is a float-reassociation sanity check, not bit-equality.
# Inductor reorders float ops and the drift is platform-dependent (macOS Accelerate
# vs Linux MKL/OpenBLAS). The residual is evaluated at the converged gold state, so
# it is near zero and its tiny absolute diffs are atol-dominated; the Jacobian needs
# the relative budget. Use a generous tolerance -- gross bugs (metadata loss, wrong
# op) differ by orders of magnitude, and the precise correctness checks are the
# forward-solve-vs-gold and adjoint-vs-eager tests.
_PARITY_TOL = {"rtol": 1e-4, "atol": 1e-6}


def _load_pmodel(input_name: str) -> NEML2PyzagModel:
    nmodel = load_nonlinear_system(_HERE / "models" / f"{input_name}.i", "eq_sys")
    return NEML2PyzagModel(nmodel)


def _gold_state_forces(pmodel: NEML2PyzagModel, input_name: str):
    ref = torch.load(str(_HERE / "gold" / f"{input_name}.pt"), weights_only=True)
    forces = torch.cat([ref["input"][v] for v in pmodel.fvars], -1)
    state = torch.cat([ref["output"][v] for v in pmodel.svars], -1)
    return state, forces


def _solve(pmodel: NEML2PyzagModel, state: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
    )
    with torch.no_grad():
        return nonlinear.solve(solver, state[0], forces.shape[0], forces)


# ── dispatch ──────────────────────────────────────────────────────────────────


def _is_compiled(model) -> bool:
    """True iff neml2.compile has replaced the model's forward in place."""
    return getattr(model, "_neml2_compiled_forward", False)


def test_compile_dispatch_and_inplace():
    """neml2.compile handles Model / ModelNonlinearSystem / NEML2PyzagModel, in place."""
    pmodel = _load_pmodel("elastic_model")
    assert not _is_compiled(pmodel.sys.model)

    ret = neml2.compile(pmodel)
    assert ret is pmodel  # in-place, returns the same object
    # The pyzag .model alias and the system's residual are the same object, now compiled.
    assert pmodel.sys.model is pmodel.model
    assert _is_compiled(pmodel.sys.model)

    # Bare Model and bare system are also accepted.
    sys2 = load_nonlinear_system(_HERE / "models" / "elastic_model.i", "eq_sys")
    assert isinstance(sys2, ModelNonlinearSystem)
    assert neml2.compile(sys2) is sys2
    assert _is_compiled(sys2.model)

    sys3 = load_nonlinear_system(_HERE / "models" / "elastic_model.i", "eq_sys")
    model = sys3.model
    assert isinstance(model, Model)
    assert neml2.compile(model) is model
    assert _is_compiled(model)


def test_compile_rejects_unsupported():
    with pytest.raises(TypeError):
        neml2.compile(object())


# ── correctness: compiled forward solve matches gold and eager ──────────────────


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compiled_forward_solve_matches_gold(input_name):
    eager = _load_pmodel(input_name)
    state, forces = _gold_state_forces(eager, input_name)
    eager_traj = _solve(eager, state, forces)

    compiled = _load_pmodel(input_name)
    neml2.compile(compiled)
    compiled_traj = _solve(compiled, state, forces)

    tol = _COMPARE_TOLERANCES.get(input_name, {})
    # Compiled vs gold (same tolerance the eager path is held to).
    assert torch.allclose(state, compiled_traj, **tol)  # pyright: ignore[reportArgumentType]
    # Compiled vs eager: should be numerically identical bar reassociation.
    assert torch.allclose(eager_traj, compiled_traj, rtol=1e-9, atol=1e-11)


# ── correctness: residual + Jacobian parity at the forward boundary ─────────────


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compiled_residual_jacobian_parity(input_name):
    """eager vs compiled pmodel.forward(state, forces): r, stack([Jn, J]) match."""
    eager = _load_pmodel(input_name)
    state, forces = _gold_state_forces(eager, input_name)
    # One block of the time history (block + lookback rows).
    s = state[:11].clone()
    f = forces[:11].clone()

    with torch.no_grad():
        r_e, J_e = eager.forward(s, f)

    compiled = _load_pmodel(input_name)
    neml2.compile(compiled)
    with torch.no_grad():
        r_c, J_c = compiled.forward(s, f)

    # ``_PARITY_TOL`` is dict[str, float]; pyright's strict ``**`` spread check
    # rejects float values for allclose's ``equal_nan: bool`` param even though our
    # dict only carries rtol / atol (matches test_NEML2PyzagModel.test_compare).
    assert torch.allclose(r_e, r_c, **_PARITY_TOL)  # pyright: ignore[reportArgumentType]
    assert torch.allclose(J_e, J_c, **_PARITY_TOL)  # pyright: ignore[reportArgumentType]


# ── compilation actually engages ────────────────────────────────────────────────


def test_compile_engages():
    dynamo.reset()
    pmodel = _load_pmodel("elastic_model")
    neml2.compile(pmodel)
    state, forces = _gold_state_forces(pmodel, "elastic_model")
    with torch.no_grad():
        pmodel.forward(state[:11], forces[:11])
    # At least one frame compiled successfully through the residual evaluation.
    assert dynamo.utils.counters["frames"]["ok"] >= 1


# ── adjoint training is not broken by compilation ───────────────────────────────


def test_adjoint_matches_eager_under_compile():
    """solve_adjoint + backward through a compiled residual yields the same parameter
    gradients as the eager residual (compile must be autograd-transparent)."""

    def run(compile_it: bool):
        nmodel = load_nonlinear_system(_HERE / "models" / "viscoplastic_model.i", "eq_sys")
        pmodel = NEML2PyzagModel(nmodel)
        if compile_it:
            neml2.compile(pmodel)
        nstep, nbatch = 20, 4
        end = torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nstep, dtype=torch.float64)
        strain = (t.reshape(-1, 1, 1) * end.reshape(1, 1, 6)).expand(nstep, nbatch, 6).contiguous()
        time = (t.unsqueeze(-1) * torch.logspace(-1, -3, nbatch, dtype=torch.float64)).contiguous()
        # Pack forces in fvars order.
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
    assert eager_grads  # non-empty
    for name in eager_grads:
        assert torch.allclose(eager_grads[name], compiled_grads[name], rtol=1e-6, atol=1e-8), (
            f"gradient mismatch for {name}"
        )


# ── the forward guard is still armed through the compiled path ──────────────────


def test_guard_still_armed_under_compile():
    from neml2.models._guard import _armed

    pmodel = _load_pmodel("elastic_model")
    neml2.compile(pmodel)
    state, forces = _gold_state_forces(pmodel, "elastic_model")
    with torch.no_grad():
        pmodel.forward(state[:11], forces[:11])
    # Outside any forward window the guard is disarmed (balanced enter/exit).
    assert not _armed("autograd")
    assert not _armed("einsum")
