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

import unittest
from pathlib import Path
import torch
import neml2


class TestLinearCombination(unittest.TestCase):
    def setUp(self):
        pwd = Path(__file__).parent
        self.model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model2")

    def test_parameter_derivative(self):
        # Batch shape 5 (shape 5,4)
        inp = torch.tensor(
            [
                [0.2, 2.5, 0.3, 0.1],
                [-0.5, 0.4, 0.2, -1.4],
                [2.5, 1.8, 0.6, 3.2],
                [-10.0, 11.0, 12.0, 6.8],
                [1.0, 2.0, 3.0, 7.2],
            ]
        )

        # The input vector only contains input
        x = neml2.LabeledVector(neml2.Tensor(inp, 1), [self.model.input_axis()])

        # Setup variable views
        self.model.reinit(x.batch.shape)

        coefs = self.model.get_parameter("c")
        coefs.requires_grad_(True)

        # Forward
        y = self.model.value(x)
        v = torch.norm(y.torch())
        v.backward()

        self.assertIsNotNone(coefs.grad)


class TestLinearInterpolation(unittest.TestCase):
    def setUp(self):
        pwd = Path(__file__).parent
        self.model = neml2.reload_model(pwd / "test_ParameterStoreVector.i", "model")

    def test_named_parameters(self):
        # Setup variable views with batch shape (5,2)
        self.model.reinit([5, 2])

        # This model has two parameters
        X = self.model.named_parameters()["X"]
        Y = self.model.named_parameters()["Y"]

        # Parameters should have the correct value
        self.assertTrue(
            torch.allclose(
                X.torch(), torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.allclose(
                Y.torch(), torch.tensor([2.0, -1.0, 5.0, 10.0], dtype=torch.float64)
            )
        )

    def test_get_parameter(self):
        # Setup variable views with batch shape (5,2)
        self.model.reinit([5, 2])

        # This model has two parameters
        X = self.model.get_parameter("X")
        Y = self.model.get_parameter("Y")

        # Parameters should have the correct value
        self.assertTrue(
            torch.allclose(
                X.torch(), torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.allclose(
                Y.torch(), torch.tensor([2.0, -1.0, 5.0, 10.0], dtype=torch.float64)
            )
        )

    def test_set_parameter(self):
        # Setup variable views with batch shape (5,2)
        self.model.reinit([5, 2])

        # This model has two parameters
        X = self.model.get_parameter("X")
        Y = self.model.get_parameter("Y")

        # This model has two parameters
        self.model.set_parameter("X", neml2.Scalar.full(200.0))
        self.model.set_parameter("Y", neml2.Scalar.full(0.2))

        # Parameters should have the correct value
        self.assertTrue(
            torch.allclose(X.torch(), torch.tensor(200.0, dtype=torch.float64))
        )
        self.assertTrue(
            torch.allclose(Y.torch(), torch.tensor(0.2, dtype=torch.float64))
        )

    def test_set_parameters(self):
        # Setup variable views with batch shape (5,2)
        self.model.reinit([5, 2])

        # This model has two parameters
        X = self.model.get_parameter("X")
        Y = self.model.get_parameter("Y")

        # This model has two parameters
        self.model.set_parameters(
            {
                "X": neml2.Scalar.full(200.0),
                "Y": neml2.Scalar.full(0.2),
            }
        )

        # Parameters should have the correct value
        self.assertTrue(
            torch.allclose(X.torch(), torch.tensor(200.0, dtype=torch.float64))
        )
        self.assertTrue(
            torch.allclose(Y.torch(), torch.tensor(0.2, dtype=torch.float64))
        )

    def test_parameter_derivative(self):
        inp = torch.tensor(
            [[0.2, 2.5, 0.3, 1.2, 0.5], [0.1, 2.5, 1.2, 1.3, 1.5]]
        ).unsqueeze(-1)

        # The input vector only contains input
        x = neml2.LabeledVector(neml2.Tensor(inp, 2), [self.model.input_axis()])

        # Setup variable views
        self.model.reinit(x.batch.shape)

        # This model has two parameters
        X = self.model.named_parameters()["X"]
        Y = self.model.named_parameters()["Y"]

        # Model parameters do not require grad by default
        self.assertFalse(X.requires_grad)
        self.assertFalse(Y.requires_grad)

        # Set parameters to requires_grad=True
        X.requires_grad_(True)
        Y.requires_grad_(True)
        self.assertTrue(X.requires_grad)
        self.assertTrue(Y.requires_grad)

        # Forward
        y = self.model.value(x)
        self.assertTrue(y.torch().requires_grad)

        # dy/dp * x
        y.torch().backward(gradient=x.torch())

        self.assertIsNotNone(X.grad)
        self.assertIsNotNone(Y.grad)
