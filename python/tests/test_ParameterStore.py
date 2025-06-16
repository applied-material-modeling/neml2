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
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # This model has two parameters
    E = model.named_parameters()["E"]
    nu = model.named_parameters()["nu"]

    # Parameters should have the correct value
    assert torch.allclose(E.torch(), torch.tensor(100.0, dtype=torch.float64))
    assert torch.allclose(nu.torch(), torch.tensor(0.3, dtype=torch.float64))


def test_get_parameter():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # Parameters should have the correct value
    assert torch.allclose(model.E.torch(), torch.tensor(100.0, dtype=torch.float64))
    assert torch.allclose(model.nu.torch(), torch.tensor(0.3, dtype=torch.float64))


def test_set_parameter():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # This model has two parameters
    model.E = neml2.Scalar.full(200.0)
    model.nu = neml2.Scalar.full(0.2)

    # Parameters should have the correct value
    assert torch.allclose(model.E.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(model.nu.torch(), torch.tensor(0.2, dtype=torch.float64))


def test_set_parameters():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_ParameterStore.i", "model")

    # This model has two parameters
    model.set_parameters(
        {
            "E": neml2.Scalar.full(200.0),
            "nu": neml2.Scalar.full(0.2),
        }
    )

    # Parameters should have the correct value
    assert torch.allclose(model.E.torch(), torch.tensor(200.0, dtype=torch.float64))
    assert torch.allclose(model.nu.torch(), torch.tensor(0.2, dtype=torch.float64))
