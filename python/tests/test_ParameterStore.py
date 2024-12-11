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


def test_named_parameters():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.named_parameters()["E"]
    nu = model.named_parameters()["nu"]

    # Parameters should have the correct value
    assert torch.allclose(E.torch(), torch.tensor(100.0, dtype=torch.float64))
    assert torch.allclose(nu.torch(), torch.tensor(0.3, dtype=torch.float64))


def test_get_parameter():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.get_parameter("E")
    nu = model.get_parameter("nu")

    # Parameters should have the correct value
    assert torch.allclose(E.torch(), torch.tensor(100.0, dtype=torch.float64))
    assert torch.allclose(nu.torch(), torch.tensor(0.3, dtype=torch.float64))


def test_set_parameter():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.get_parameter("E")
    nu = model.get_parameter("nu")

    # This model has two parameters
    model.set_parameter("E", neml2.Scalar.full(200.0))
    model.set_parameter("nu", neml2.Scalar.full(0.2))

    # Parameters should have the correct value
    assert torch.allclose(E.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(nu.torch(), torch.tensor(0.2, dtype=torch.float64))


def test_set_parameters():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.get_parameter("E")
    nu = model.get_parameter("nu")

    # This model has two parameters
    model.set_parameters(
        {
            "E": neml2.Scalar.full(200.0),
            "nu": neml2.Scalar.full(0.2),
        }
    )

    # Parameters should have the correct value
    assert torch.allclose(E.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(nu.torch(), torch.tensor(0.2, dtype=torch.float64))


def test_change_batch_dimensions():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    E = model.get_parameter("E")
    nu = model.get_parameter("nu")

    # This model has two parameters
    model.set_parameters(
        {
            "E": neml2.Scalar.full((5, 2), 200.0),
            "nu": neml2.Scalar.full((5, 2), 0.2),
        }
    )

    # Should have correct shape
    assert E.tensor().batch.shape == (5, 2)
    assert nu.tensor().batch.shape == (5, 2)

    # Now go back
    model.set_parameters(
        {
            "E": neml2.Scalar.full(200.0),
            "nu": neml2.Scalar.full(0.2),
        }
    )

    assert E.tensor().batch.shape == tuple()
    assert nu.tensor().batch.shape == tuple()


def test_parameter_derivative():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStore.i", "model")

    # Strain of batch shape (5,2)
    e = neml2.SR2(torch.tensor([0.1, 0.2, 0.05, 0, 0, 0]).expand(5, 2, 6))

    # The input vector only contains strain
    x = neml2.LabeledVector(neml2.Tensor(e), [model.input_axis()])

    # Setup variable views
    model.reinit(x.batch.shape)

    # This model has two parameters
    E = model.named_parameters()["E"]
    nu = model.named_parameters()["nu"]

    # Model parameters do not require grad by default
    assert not E.requires_grad
    assert not nu.requires_grad

    # Set parameters to requires_grad=True
    E.requires_grad_(True)
    nu.requires_grad_(True)
    assert E.requires_grad
    assert nu.requires_grad

    # Forward
    y = model.value(x)
    assert y.torch().requires_grad

    # dy/dp * x
    y.torch().backward(gradient=x.torch())
    assert torch.allclose(E.grad, torch.tensor(1.1105769561746945, dtype=torch.float64))
    assert torch.allclose(
        nu.grad, torch.tensor(503.51332861533353, dtype=torch.float64)
    )
