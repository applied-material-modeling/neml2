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
from neml2.tensors import Scalar, TensorType


def test_get_model():
    pwd = Path(__file__).parent
    factory = neml2.load_input(pwd / "test_Model.i")
    model = factory.get_model("model")
    assert model


def test_input_type():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_Model.i", "model")
    assert model.input_type("t") == TensorType.Scalar
    assert model.input_type("t~1") == TensorType.Scalar
    assert model.input_type("foo_rate") == TensorType.Scalar
    assert model.input_type("foo") == TensorType.Scalar
    assert model.input_type("foo~1") == TensorType.Scalar
    assert model.input_type("bar_rate") == TensorType.Scalar
    assert model.input_type("bar") == TensorType.Scalar
    assert model.input_type("bar~1") == TensorType.Scalar


def test_output_type():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_Model.i", "model")
    assert model.output_type("foo_bar_residual") == TensorType.Scalar


def test_forward():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_Model.i", "model")

    # Note input variables can have different batch shapes
    x = {
        "t": Scalar.full((1, 2), (), 0.1),
        "t~1": Scalar.zeros((3, 1, 1)),
        "foo_rate": Scalar.full(0.1),
        "foo": Scalar.full(1.5),
        "foo~1": Scalar.full((5, 2), (), 2),
        "bar_rate": Scalar.full(-0.2),
        "bar": Scalar.full(1.5),
        "bar~1": Scalar.full(2.0),
    }

    def check_y(y):
        val = y["foo_bar_residual"].torch()
        assert val.shape == (3, 5, 2)
        assert torch.allclose(val, torch.full((3, 5, 2), -0.99))

    def check_dy_dx(dy_dx):
        val = dy_dx["foo_bar_residual"]["t"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(0.1))

        val = dy_dx["foo_bar_residual"]["t~1"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(-0.1))

        val = dy_dx["foo_bar_residual"]["bar~1"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(-1.0))

        val = dy_dx["foo_bar_residual"]["foo~1"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(-1.0))

        val = dy_dx["foo_bar_residual"]["bar"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(1.0))

        val = dy_dx["foo_bar_residual"]["bar_rate"].torch()
        assert val.shape == (3, 1, 2)
        assert torch.allclose(val, torch.tensor(-0.1))

        val = dy_dx["foo_bar_residual"]["foo"].torch()
        assert val.shape == ()
        assert torch.allclose(val, torch.tensor(1.0))

        val = dy_dx["foo_bar_residual"]["foo_rate"].torch()
        assert val.shape == (3, 1, 2)
        assert torch.allclose(val, torch.tensor(-0.1))

    y = model.value(x)
    check_y(y)

    dy_dx = model.dvalue(x)
    check_dy_dx(dy_dx)

    y, dy_dx = model.value_and_dvalue(x)
    check_y(y)
    check_dy_dx(dy_dx)
