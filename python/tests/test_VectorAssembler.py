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
import neml2
from neml2.tensors import Scalar, Tensor
import torch


def test_assemble_by_variable():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    axis = model.input_axis()
    assembler = neml2.VectorAssembler(axis)
    v = assembler.assemble_by_variable(
        {
            "state/bar": Scalar.full(1.0),
            "state/foo": Scalar.full((2, 3), 2.0),
            "state/foo_rate": Scalar.full((6, 1, 1), 3.0),
        }
    )
    assert v.batch.shape == (6, 2, 3)
    assert v.base.shape == (8,)
    assert torch.allclose(v.base[4].torch(), torch.tensor([1.0]))
    assert torch.allclose(v.base[6].torch(), torch.tensor([2.0]))
    assert torch.allclose(v.base[7].torch(), torch.tensor([3.0]))


def test_split_by_variable():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    axis = model.input_axis()
    assembler = neml2.VectorAssembler(axis)
    v = Tensor(torch.linspace(0, 1, 8), 0)
    vals = assembler.split_by_variable(v)
    assert torch.allclose(vals["forces/t"].torch(), v.base[0].torch())
    assert torch.allclose(vals["old_forces/t"].torch(), v.base[1].torch())
    assert torch.allclose(vals["old_state/bar"].torch(), v.base[2].torch())
    assert torch.allclose(vals["old_state/foo"].torch(), v.base[3].torch())
    assert torch.allclose(vals["state/bar"].torch(), v.base[4].torch())
    assert torch.allclose(vals["state/bar_rate"].torch(), v.base[5].torch())
    assert torch.allclose(vals["state/foo"].torch(), v.base[6].torch())
    assert torch.allclose(vals["state/foo_rate"].torch(), v.base[7].torch())


def test_split_by_subaxis():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    axis = model.input_axis()
    assembler = neml2.VectorAssembler(axis)
    v = Tensor(torch.linspace(0, 1, 8), 0)
    vals = assembler.split_by_subaxis(v)
    assert torch.allclose(vals["forces"].torch(), v.base[0].torch())
    assert torch.allclose(vals["old_forces"].torch(), v.base[1].torch())
    assert torch.allclose(vals["old_state"].torch(), v.base[2:4].torch())
    assert torch.allclose(vals["state"].torch(), v.base[4:8].torch())
