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
from pathlib import Path

import torch
import neml2
from pyzag import nonlinear, chunktime


def test_definition():
    pwd = Path(__file__).parent
    nmodel = neml2.load_model(pwd / "models" / "correct_model.i", "implicit_rate")
    pmodel = neml2.pyzag.NEML2PyzagModel(nmodel, exclude_parameters=["elasticity_nu"])

    assert set(dict(pmodel.named_parameters()).keys()) == {
        "elasticity_E",
        "flow_rate_eta",
        "flow_rate_n",
        "isoharden_K",
        "yield_sy",
    }
    for pname, param in pmodel.named_parameters():
        assert torch.allclose(param, pmodel.model.get_parameter(pname).torch())
    assert pmodel.nstate == 7
    assert pmodel.nforce == 7
    assert pmodel.lookback == 1


def test_change_parameter_shape():
    pwd = Path(__file__).parent
    nmodel = neml2.load_model(pwd / "models" / "correct_model.i", "implicit_rate")
    pmodel = neml2.pyzag.NEML2PyzagModel(nmodel, exclude_parameters=["elasticity_nu"])

    # Modify the parameter, batch shape = (10,)
    pmodel.elasticity_E.data = torch.tensor(1.2e5).expand(10)
    pmodel._update_parameter_values()
    assert torch.allclose(pmodel.model.elasticity_E.torch(), pmodel.get_parameter("elasticity_E"))
    assert pmodel.model.elasticity_E.tensor().batch.shape == (10,)

    # Modify again, batch shape = ()
    pmodel.elasticity_E.data = torch.tensor(1.3e5)
    pmodel._update_parameter_values()
    assert torch.allclose(pmodel.model.elasticity_E.torch(), pmodel.get_parameter("elasticity_E"))
    assert pmodel.model.elasticity_E.tensor().batch.shape == ()


@pytest.mark.parametrize("input", ["elastic_model", "viscoplastic_model", "km_mixed_model"])
def test_compare(input):
    pwd = Path(__file__).parent
    nmodel = neml2.load_model(pwd / "models" / "{}.i".format(input), "implicit_rate")
    pmodel = neml2.pyzag.NEML2PyzagModel(nmodel)

    # Reference to compare against
    ref = torch.jit.load(pwd / "gold" / "{}.pt".format(input))
    input = dict(ref.input.named_buffers())
    output = dict(ref.output.named_buffers())
    forces = torch.cat([input[v] for v in pmodel.nl_sys.gmap()], -1)
    state = torch.cat([output[v] for v in pmodel.nl_sys.umap()], -1)
    nstep = forces.shape[0]

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
    )
    with torch.no_grad():
        results = nonlinear.solve(solver, state[0], nstep, forces)

    assert torch.allclose(state, results)
