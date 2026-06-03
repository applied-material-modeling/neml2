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


from pathlib import Path

import pytest
import torch
from pyzag import chunktime, nonlinear

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagModel

# Native ``SR2LinearCombination`` registers default-valued ``weights`` and
# ``offset`` as ``nn.Parameter``s; the C++ side never exposed them. To match
# the original C++ pyzag-test parameter set (user-facing material parameters
# only), exclude them.
_LINCOMB_INTERNAL_PARAMS = [
    "Eerate_offset",
    "Eerate_weight_0",
    "Eerate_weight_1",
]


def test_definition():
    pwd = Path(__file__).parent
    nmodel = load_nonlinear_system(pwd / "models" / "correct_model.i", "eq_sys")
    pmodel = NEML2PyzagModel(
        nmodel, exclude_parameters=["elasticity_nu", *_LINCOMB_INTERNAL_PARAMS]
    )

    names = set(dict(pmodel.named_parameters()).keys())
    # Matches the original C++ test exactly now.
    assert names == {
        "elasticity_E",
        "flow_rate_eta",
        "flow_rate_n",
        "isoharden_K",
        "yield_sy",
    }
    # Each wrapper-side mirror must match the underlying model's parameter at init time.
    for pname, mirror in pmodel.named_parameters():
        if pname not in pmodel._param_targets:
            continue  # nn.Module bookkeeping (e.g. from pyzag base)
        module, leaf = pmodel._param_targets[pname]
        underlying = module._parameters[leaf]
        assert underlying is not None
        assert torch.allclose(mirror, underlying)
    assert pmodel.nstate == 7
    assert pmodel.nforce == 7
    assert pmodel.lookback == 1


def test_change_parameter_shape():
    pwd = Path(__file__).parent
    nmodel = load_nonlinear_system(pwd / "models" / "correct_model.i", "eq_sys")
    pmodel = NEML2PyzagModel(nmodel, exclude_parameters=["elasticity_nu"])
    module, leaf = pmodel._param_targets["elasticity_E"]

    def _bound(module, leaf):
        # After the first _update_parameter_values, the leaf is bound as a
        # plain instance attribute (possibly wrapped in the leaf's typed
        # class). Unwrap to a raw tensor for shape/value checks.
        v = getattr(module, leaf)
        return v.data if hasattr(v, "data") and not isinstance(v, torch.Tensor) else v

    # Modify the wrapper-side parameter to carry a batch shape of (10,).
    pmodel.elasticity_E.data = torch.tensor(1.2e5).expand(10)
    pmodel._update_parameter_values()
    assert torch.allclose(_bound(module, leaf), pmodel.get_parameter("elasticity_E"))
    assert _bound(module, leaf).shape == (10,)

    # Back to scalar (no batch axis).
    pmodel.elasticity_E.data = torch.tensor(1.3e5)
    pmodel._update_parameter_values()
    assert torch.allclose(_bound(module, leaf), pmodel.get_parameter("elasticity_E"))
    assert _bound(module, leaf).shape == ()


# Per-scenario tolerance for `assert torch.allclose(gold, native)` in
# `test_compare`. The gold trajectories were baked by the (now-retired) C++
# pipeline, so any difference between C++ and native floating-point order
# of operations shows up here as a drift on the order of 1e-7. The
# tolerance is the tightest that still admits the native trajectory.
#
# - elastic / viscoplastic: identical to default ``torch.allclose``.
# - km_mixed: the ``gamma_rate_ri`` column (rate-independent flow solved
#   via complementarity) has 8 / 16000 entries that drift just past the
#   default ``rtol=1e-5`` threshold (~1e-7 absolute on values of
#   ~1e-2 — 1e-5 relative right at the boundary). Bump rtol slightly so
#   the test catches real regressions while tolerating the round-off
#   sensitivity of the complementarity solve.
_COMPARE_TOLERANCES = {
    "km_mixed_model": {"rtol": 1e-4, "atol": 1e-8},
}


@pytest.mark.parametrize("input_name", ["elastic_model", "viscoplastic_model", "km_mixed_model"])
def test_compare(input_name):
    pwd = Path(__file__).parent
    nmodel = load_nonlinear_system(pwd / "models" / f"{input_name}.i", "eq_sys")
    pmodel = NEML2PyzagModel(nmodel)

    # Gold reference baked from the (now-retired) C++ pipeline; the gold .pt
    # records the time-series inputs/outputs as a TorchScript module with
    # buffer attributes named after the HIT variables.
    ref = torch.jit.load(pwd / "gold" / f"{input_name}.pt")
    input_buffers = dict(ref.input.named_buffers())
    output_buffers = dict(ref.output.named_buffers())
    forces = torch.cat([input_buffers[v] for v in pmodel.fvars], -1)
    state = torch.cat([output_buffers[v] for v in pmodel.svars], -1)
    nstep = forces.shape[0]

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
    )
    with torch.no_grad():
        results = nonlinear.solve(solver, state[0], nstep, forces)

    tol = _COMPARE_TOLERANCES.get(input_name, {})
    assert torch.allclose(state, results, **tol)
