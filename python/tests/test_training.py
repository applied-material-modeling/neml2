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
import math
import torch
import neml2


def test_parameter_gradient():
    """
    Test that we can compute parameter gradients through a model.
    This test case is setup mimicking pyzag workflow where the input and output are represented as dense, flat vector.
    """
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_training.i", "model")

    # Initialize the model with the correct batch shape
    B = (2, 5)

    # Define the input
    ndof = model.input_axis().size()
    x = torch.linspace(0, 0.2, ndof).expand(*B, -1)
    x = neml2.Tensor(x, len(B))

    # Disassemble the flat input vector into model variables
    input_vars = ["forces/E", "forces/t", "old_forces/E", "old_forces/t", "old_state/S", "state/S"]
    input_var_sizes = [(6,), (), (6,), (), (6,), (6,)]
    X = neml2.HVector(input_var_sizes)
    X.disassemble(x)

    # Say I want to get the parameter gradient on the flow viscosity
    p = model.flow_rate_eta
    p.requires_grad_(True)

    # Evaluate the model and the loss function
    y = model.value(neml2.bind(input_vars, X))

    # Assemble the model output variables back into a flat vector
    output_vars = ["state/S"]
    output_var_sizes = [(6,)]
    Y = neml2.HVector([y[v] for v in output_vars], output_var_sizes)
    [y_f, _] = Y.assemble()

    # Calculate the loss function
    f = torch.norm(y_f.torch())

    # Get the parameter gradient
    f.backward()
    assert math.isclose(p.grad.item(), 0.023917, rel_tol=1e-6, abs_tol=1e-6)
