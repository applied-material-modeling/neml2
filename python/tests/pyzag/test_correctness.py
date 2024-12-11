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

"""Test we get the same answers in pyzag with parallel time as we do in NEML2 with sequential time"""

import unittest
import os.path

import torch

torch.set_default_dtype(torch.double)

import neml2
from pyzag import nonlinear, chunktime


nchunk = 10

models = ["elastic_model", "viscoplastic_model", "complex_model"]


class TestCorrectness(unittest.TestCase):
    def test_correctness(self):
        for i, model in enumerate(models):
            with self.subTest(i=i):
                self.run_model(model)

    def run_model(self, model_name):
        nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), model_name + ".i"),
            "implicit_rate",
        )
        model = neml2.pyzag.NEML2PyzagModel(nmodel)

        results = torch.jit.load(
            os.path.join(os.path.dirname(__file__), "result_" + model_name + ".pt")
        )

        modules = dict(results.named_modules())
        input = dict(modules["input"].named_buffers())
        output = dict(modules["output"].named_buffers())

        forces = torch.cat(
            [
                input["forces/" + n]
                for n in model.model.input_axis().subaxis("forces").variable_names()
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                output["state/" + n]
                for n in model.model.input_axis().subaxis("state").variable_names()
            ],
            dim=-1,
        )

        solver = nonlinear.RecursiveNonlinearEquationSolver(
            model,
            step_generator=nonlinear.StepGenerator(nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
            nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-8, atol=1.0e-10),
        )
        with torch.no_grad():
            results = nonlinear.solve(
                solver,
                state[0],
                len(forces),
                forces,
            )

        self.assertTrue(torch.allclose(state, results))
