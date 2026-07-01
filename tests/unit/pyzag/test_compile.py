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

"""``neml2.compile`` must be a transparent, in-place acceleration of the pyzag
residual/Jacobian evaluation: the forward solve and residual/Jacobian outputs
match eager (and gold), the adjoint path keeps working, and the guard stays armed."""

from __future__ import annotations

import pytest
import torch
import torch._dynamo as dynamo
from pyzag import chunktime, nonlinear

import neml2
from neml2 import load_nonlinear_system
from neml2.es import ModelNonlinearSystem
from neml2.models.model import Model
from neml2.pyzag import NEML2PyzagFactory

_ALL_MODELS = ["elastic_model", "viscoplastic_model", "km_mixed_model"]

# Eager-vs-compiled parity is a float-reassociation sanity check, not bit-equality.
_PARITY_TOL = {"rtol": 1e-4, "atol": 1e-6}


def _load_factory(models_dir, input_name) -> NEML2PyzagFactory:
    nmodel = load_nonlinear_system(models_dir / f"{input_name}.i", "eq_sys")
    return NEML2PyzagFactory(nmodel, compile=False)


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


def _assert_blocks_close(am_e, am_c, tol):
    """Every defined (i, j) block of two assembled matrices agrees within ``tol``."""
    for i in range(am_e.row_layout.ngroup()):
        for j in range(am_e.col_layout.ngroup()):
            be, bc = am_e.tensors[i][j], am_c.tensors[i][j]
            if be.defined() and bc.defined():
                assert torch.allclose(be.torch(), bc.torch(), rtol=tol["rtol"], atol=tol["atol"])


def test_compile_dispatch_and_inplace(models_dir):
    """neml2.compile accepts Model / ModelNonlinearSystem / NEML2PyzagFactory, in place."""
    pmodel = _load_factory(models_dir, "elastic_model")
    assert not _is_compiled(pmodel.sys.model)

    assert neml2.compile(pmodel) is pmodel
    assert pmodel.sys.model is pmodel.model
    assert _is_compiled(pmodel.sys.model)

    sys2 = load_nonlinear_system(models_dir / "elastic_model.i", "eq_sys")
    assert isinstance(sys2, ModelNonlinearSystem)
    assert neml2.compile(sys2) is sys2
    assert _is_compiled(sys2.model)

    sys3 = load_nonlinear_system(models_dir / "elastic_model.i", "eq_sys")
    model = sys3.model
    assert isinstance(model, Model)
    assert neml2.compile(model) is model
    assert _is_compiled(model)


def test_compile_rejects_unsupported():
    with pytest.raises(TypeError):
        neml2.compile(object())


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compiled_forward_solve_matches_gold(models_dir, gold_dir, compare_tolerances, input_name):
    """A compiled forward solve matches both the gold trajectory and the eager solve."""
    eager = _load_factory(models_dir, input_name)
    state, forces = _gold_state_forces(gold_dir, eager, input_name)
    eager_traj = _solve(eager, state, forces)

    compiled = _load_factory(models_dir, input_name)
    neml2.compile(compiled)
    compiled_traj = _solve(compiled, state, forces)

    tol = compare_tolerances.get(input_name, {})
    assert torch.allclose(state, compiled_traj, **tol)  # pyright: ignore[reportArgumentType]
    assert torch.allclose(eager_traj, compiled_traj, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compiled_residual_jacobian_parity(models_dir, gold_dir, input_name):
    """Eager and compiled evaluate_raw agree on the residual and both Jacobian blocks."""
    eager = _load_factory(models_dir, input_name)
    state, forces = _gold_state_forces(gold_dir, eager, input_name)
    s = state[:11].clone()
    f = forces[:11].clone()

    with torch.no_grad():
        r_e, J_e = eager.evaluate_raw(s, [f])

    compiled = _load_factory(models_dir, input_name)
    neml2.compile(compiled)
    with torch.no_grad():
        r_c, J_c = compiled.evaluate_raw(s, [f])

    assert torch.allclose(r_e, r_c, **_PARITY_TOL)  # pyright: ignore[reportArgumentType]
    _assert_blocks_close(J_e.diag_am, J_c.diag_am, _PARITY_TOL)
    _assert_blocks_close(J_e.sub_am, J_c.sub_am, _PARITY_TOL)


def test_compile_engages(models_dir, gold_dir):
    """Evaluating a compiled factory drives at least one dynamo frame to completion."""
    dynamo.reset()
    pmodel = _load_factory(models_dir, "elastic_model")
    neml2.compile(pmodel)
    state, forces = _gold_state_forces(gold_dir, pmodel, "elastic_model")
    with torch.no_grad():
        pmodel.evaluate_raw(state[:11], [forces[:11]])
    assert dynamo.utils.counters["frames"]["ok"] >= 1


def test_adjoint_matches_eager_under_compile(models_dir):
    """solve_adjoint + backward through a compiled residual gives the eager gradients."""

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


def test_guard_still_armed_under_compile(models_dir, gold_dir):
    """The forward guard is disarmed again after a compiled evaluation."""
    from neml2.models._guard import _armed

    pmodel = _load_factory(models_dir, "elastic_model")
    neml2.compile(pmodel)
    state, forces = _gold_state_forces(gold_dir, pmodel, "elastic_model")
    with torch.no_grad():
        pmodel.evaluate_raw(state[:11], [forces[:11]])
    assert not _armed("autograd")
    assert not _armed("einsum")


def test_construct_with_compile_true(models_dir, gold_dir):
    """The default ``compile=True`` constructor compiles the residual at __init__
    (exercising ``NEML2PyzagFactory._compile`` and its dynamo/AOTAutograd config)
    and still evaluates in parity with an eager-constructed factory."""
    nmodel = load_nonlinear_system(models_dir / "elastic_model.i", "eq_sys")
    compiled = NEML2PyzagFactory(nmodel, compile=True)
    assert _is_compiled(compiled.sys.model)

    state, forces = _gold_state_forces(gold_dir, compiled, "elastic_model")
    s, f = state[:11].clone(), forces[:11].clone()
    with torch.no_grad():
        r_c, _ = compiled.evaluate_raw(s, [f])
    assert torch.isfinite(r_c).all()

    eager = _load_factory(models_dir, "elastic_model")
    with torch.no_grad():
        r_e, _ = eager.evaluate_raw(s, [f])
    assert torch.allclose(r_c, r_e, **_PARITY_TOL)  # pyright: ignore[reportArgumentType]
