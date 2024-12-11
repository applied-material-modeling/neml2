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

"""Test basic setup of NEML2 models"""

import unittest
import os.path

import neml2

import torch

torch.set_default_dtype(torch.double)


class TestChangeParameterShape(unittest.TestCase):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "correct_model.i"), "implicit_rate"
        )
        self.pmodel = neml2.pyzag.NEML2PyzagModel(
            self.nmodel, exclude_parameters=["elasticity.nu"]
        )

    def test_update_parameter(self):
        self.pmodel.elasticity_E.data = torch.tensor(1.2e5).expand(10)
        self.pmodel._update_parameter_values()
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )
        self.assertEqual(
            self.pmodel.model.get_parameter("elasticity.E").tensor().batch.shape, (10,)
        )

        # Now go back
        self.pmodel.elasticity_E.data = torch.tensor(1.3e5)
        self.pmodel._update_parameter_values()
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )
        self.assertEqual(
            self.pmodel.model.get_parameter("elasticity.E").tensor().batch.shape,
            tuple(),
        )


class TestCorrectlyDefinedModel(unittest.TestCase):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "correct_model.i"), "implicit_rate"
        )
        self.pmodel = neml2.pyzag.NEML2PyzagModel(
            self.nmodel, exclude_parameters=["elasticity.nu"]
        )

    def test_parameter_names(self):
        """Check that names and parameter values are the same"""
        self.assertEqual(
            set([n for n, _ in self.pmodel.named_parameters()]),
            set(
                [
                    "elasticity_E",
                    "flow_rate_eta",
                    "flow_rate_n",
                    "isoharden_K",
                    "yield_sy",
                ]
            ),
        )

        for name, val in self.pmodel.named_parameters():
            self.assertTrue(
                torch.allclose(
                    val,
                    self.pmodel.model.named_parameters()[
                        self.pmodel.parameter_name_map[name]
                    ].torch(),
                )
            )

    def test_update_parameter(self):
        self.assertTrue(torch.allclose(self.pmodel.elasticity_E, torch.tensor(1e5)))
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )
        self.pmodel.elasticity_E.data = torch.tensor(1.2e5)
        self.pmodel._update_parameter_values()
        self.assertTrue(torch.allclose(self.pmodel.elasticity_E, torch.tensor(1.2e5)))
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )

    def test_nstate(self):
        self.assertEqual(self.pmodel.nstate, 7)

    def test_nforce(self):
        self.assertEqual(self.pmodel.nforce, 7)


class TestSetVectorParameter(unittest.TestCase):

    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "complex_model.i"), "implicit_rate"
        )
        self.pmodel = neml2.pyzag.NEML2PyzagModel(self.nmodel)

    def test_set_vector(self):
        self.assertTrue(
            torch.allclose(
                self.pmodel.mu_Y, self.nmodel.named_parameters()["mu.Y"].torch()
            )
        )
        self.pmodel.mu_Y = torch.nn.Parameter(torch.ones_like(self.pmodel.mu_Y))
        self.pmodel._update_parameter_values()
        self.assertTrue(torch.allclose(self.pmodel.mu_Y, torch.tensor(1.0)))
        self.assertTrue(
            torch.allclose(
                self.nmodel.named_parameters()["mu.Y"].torch(), torch.tensor(1.0)
            )
        )

    def test_set_scalar(self):
        self.assertTrue(torch.allclose(self.pmodel.elasticity_E, torch.tensor(1e5)))
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )
        self.pmodel.elasticity_E.data = torch.tensor(1.2e5)
        self.pmodel._update_parameter_values()
        self.assertTrue(torch.allclose(self.pmodel.elasticity_E, torch.tensor(1.2e5)))
        self.assertTrue(
            torch.allclose(
                self.pmodel.model.named_parameters()["elasticity.E"].torch(),
                self.pmodel.elasticity_E,
            )
        )
