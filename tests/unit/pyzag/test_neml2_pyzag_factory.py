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

import pytest
import torch
from pyzag import chunktime, nonlinear

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagFactory

_ALL_MODELS = ["elastic_model", "viscoplastic_model", "km_mixed_model"]


def _load_factory(models_dir, input_name, **kwargs):
    nmodel = load_nonlinear_system(models_dir / f"{input_name}.i", "eq_sys")
    return NEML2PyzagFactory(nmodel, compile=False, **kwargs)


def test_definition(models_dir, lincomb_internal_params):
    """Excluding nu + the LinearCombination plumbing leaves exactly the material parameters."""
    pmodel = _load_factory(
        models_dir, "correct_model", exclude_parameters=["elasticity_nu", *lincomb_internal_params]
    )

    assert set(dict(pmodel.named_parameters())) == {
        "elasticity_E",
        "flow_rate_eta",
        "flow_rate_n",
        "isoharden_K",
        "yield_surface_sy",
    }
    for pname, mirror in pmodel.named_parameters():
        if pname not in pmodel._param_targets:
            continue
        module, leaf = pmodel._param_targets[pname]
        underlying = module._parameters[leaf]
        assert underlying is not None
        assert torch.allclose(mirror, underlying)
    assert pmodel.nstate == 7
    assert pmodel.nforce == 7
    assert pmodel.lookback == 1


def test_change_parameter_shape(models_dir):
    """Reshaping a mirror parameter propagates to the bound model leaf on the next sync."""
    pmodel = _load_factory(models_dir, "correct_model", exclude_parameters=["elasticity_nu"])
    module, leaf = pmodel._param_targets["elasticity_E"]

    def _bound():
        v = getattr(module, leaf)
        return v.data if hasattr(v, "data") and not isinstance(v, torch.Tensor) else v

    pmodel.elasticity_E.data = torch.tensor(1.2e5).expand(10)
    pmodel._update_parameter_values()
    assert torch.allclose(_bound(), pmodel.get_parameter("elasticity_E"))
    assert _bound().shape == (10,)

    pmodel.elasticity_E.data = torch.tensor(1.3e5)
    pmodel._update_parameter_values()
    assert torch.allclose(_bound(), pmodel.get_parameter("elasticity_E"))
    assert _bound().shape == ()


def test_default_trains_all_parameters(models_dir, lincomb_internal_params):
    """With neither list, every HIT-named parameter is mirrored."""
    pmodel = _load_factory(models_dir, "correct_model")
    assert set(dict(pmodel.named_parameters())) == {
        "elasticity_E",
        "elasticity_nu",
        "flow_rate_eta",
        "flow_rate_n",
        "isoharden_K",
        "yield_surface_sy",
        *lincomb_internal_params,
    }


def test_include_parameters(models_dir):
    """``include_parameters`` mirrors only the listed names."""
    targets = [
        "elasticity_E",
        "flow_rate_eta",
        "flow_rate_n",
        "isoharden_K",
        "yield_surface_sy",
    ]
    pmodel = _load_factory(models_dir, "correct_model", include_parameters=targets)
    assert set(dict(pmodel.named_parameters())) == set(targets)


def test_include_and_exclude_are_mutually_exclusive(models_dir):
    nmodel = load_nonlinear_system(models_dir / "correct_model.i", "eq_sys")
    with pytest.raises(ValueError, match="mutually exclusive"):
        NEML2PyzagFactory(
            nmodel,
            include_parameters=["elasticity_E"],
            exclude_parameters=["elasticity_nu"],
        )


def test_include_unknown_parameter_raises(models_dir):
    nmodel = load_nonlinear_system(models_dir / "correct_model.i", "eq_sys")
    with pytest.raises(ValueError, match="unknown parameter"):
        NEML2PyzagFactory(nmodel, include_parameters=["elasticity_E", "not_a_param"])


def test_evaluate_raw_accepts_multiple_forces(models_dir, gold_dir):
    """A split list of force tensors is concatenated and gives the same (r, J)."""
    pmodel = _load_factory(models_dir, "viscoplastic_model")
    ref = torch.load(str(gold_dir / "viscoplastic_model.pt"), weights_only=True)
    forces = torch.cat([ref["input"][v] for v in pmodel.fvars], -1)
    state = torch.cat([ref["output"][v] for v in pmodel.svars], -1)
    s = state[:11].clone()
    f = forces[:11].clone()

    with torch.no_grad():
        r1, j1 = pmodel.evaluate_raw(s, [f])
        k = f.shape[-1] // 2
        r2, j2 = pmodel.evaluate_raw(s, [f[..., :k], f[..., k:]])

    assert torch.equal(r1, r2)
    assert torch.equal(j1.diag_am.tensors[0][0].data, j2.diag_am.tensors[0][0].data)


@pytest.mark.parametrize("input_name", _ALL_MODELS)
def test_compare(models_dir, gold_dir, compare_tolerances, input_name):
    """The native forward solve reproduces the checked-in gold trajectory."""
    pmodel = _load_factory(models_dir, input_name)

    ref = torch.load(str(gold_dir / f"{input_name}.pt"), weights_only=True)
    forces = torch.cat([ref["input"][v] for v in pmodel.fvars], -1)
    state = torch.cat([ref["output"][v] for v in pmodel.svars], -1)
    nstep = forces.shape[0]

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),  # pyright: ignore[reportArgumentType]
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
    )
    with torch.no_grad():
        results = nonlinear.solve(solver, state[0], nstep, forces)

    tol = compare_tolerances.get(input_name, {})
    assert torch.allclose(state, results, **tol)  # pyright: ignore[reportArgumentType]
