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
    model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    X = model.named_parameters()["X"]
    Y = model.named_parameters()["Y"]

    # Parameters should have the correct value
    assert torch.allclose(
        X.torch(), torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    )
    assert torch.allclose(
        Y.torch(), torch.tensor([2.0, -1.0, 5.0, 10.0], dtype=torch.float64)
    )


def test_get_parameter():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    X = model.get_parameter("X")
    Y = model.get_parameter("Y")

    # Parameters should have the correct value
    assert torch.allclose(
        X.torch(), torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    )
    assert torch.allclose(
        Y.torch(), torch.tensor([2.0, -1.0, 5.0, 10.0], dtype=torch.float64)
    )


def test_set_parameter():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    X = model.get_parameter("X")
    Y = model.get_parameter("Y")

    # This model has two parameters
    model.set_parameter("X", neml2.Scalar.full(200.0))
    model.set_parameter("Y", neml2.Scalar.full(0.2))

    # Parameters should have the correct value
    assert torch.allclose(X.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(Y.torch(), torch.tensor(0.2, dtype=torch.float64))


def test_set_parameters():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    # Setup variable views with batch shape (5,2)
    model.reinit([5, 2])

    # This model has two parameters
    X = model.get_parameter("X")
    Y = model.get_parameter("Y")

    # This model has two parameters
    model.set_parameters(
        {
            "X": neml2.Scalar.full(200.0),
            "Y": neml2.Scalar.full(0.2),
        }
    )

    # Parameters should have the correct value
    assert torch.allclose(X.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(Y.torch(), torch.tensor(0.2, dtype=torch.float64))


def test_parameter_derivative():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    inp = neml2.Scalar(torch.ones((5, 2)) * 1.2)

    # The input vector only contains input
    x = neml2.LabeledVector(neml2.Tensor(inp), [model.input_axis()])

    # Setup variable views
    model.reinit(x.batch.shape)

    # This model has two parameters
    X = model.named_parameters()["X"]
    Y = model.named_parameters()["Y"]

    # Model parameters do not require grad by default
    assert not X.requires_grad
    assert not Y.requires_grad

    # Set parameters to requires_grad=True
    X.requires_grad_(True)
    Y.requires_grad_(True)
    assert X.requires_grad
    assert Y.requires_grad

    # Forward
    y = model.value(x)
    assert y.torch().requires_grad

    # dy/dp * x
    y.torch().backward(gradient=x.torch())

    assert X.grad is not None
    assert Y.grad is not None
