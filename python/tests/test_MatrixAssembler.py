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
from neml2.tensors import Tensor
import torch


def test_assemble_by_variable():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    xaxis = model.input_axis()
    yaxis = model.output_axis()
    assembler = neml2.MatrixAssembler(yaxis, xaxis)
    M = assembler.assemble_by_variable(
        {
            "residual/foo_bar": {
                "state/bar": Tensor.full((1, 1), 1.0),
                "state/foo": Tensor.full((2, 3), (1, 1), 2.0),
                "state/foo_rate": Tensor.full((6, 1, 1), (1, 1), 3.0),
            }
        }
    )
    assert M.batch.shape == (6, 2, 3)
    assert M.base.shape == (1, 8)
    assert torch.allclose(M.base[0, 4].torch(), torch.tensor([1.0]))
    assert torch.allclose(M.base[0, 6].torch(), torch.tensor([2.0]))
    assert torch.allclose(M.base[0, 7].torch(), torch.tensor([3.0]))


def test_split_by_variable():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    xaxis = model.input_axis()
    yaxis = model.output_axis()
    assembler = neml2.MatrixAssembler(yaxis, xaxis)
    M = Tensor(torch.linspace(0, 1, 8).reshape(1, 8), 0)
    vals = assembler.split_by_variable(M)["residual/foo_bar"]
    assert torch.allclose(vals["forces/t"].torch(), M.base[0, 0].torch())
    assert torch.allclose(vals["old_forces/t"].torch(), M.base[0, 1].torch())
    assert torch.allclose(vals["old_state/bar"].torch(), M.base[0, 2].torch())
    assert torch.allclose(vals["old_state/foo"].torch(), M.base[0, 3].torch())
    assert torch.allclose(vals["state/bar"].torch(), M.base[0, 4].torch())
    assert torch.allclose(vals["state/bar_rate"].torch(), M.base[0, 5].torch())
    assert torch.allclose(vals["state/foo"].torch(), M.base[0, 6].torch())
    assert torch.allclose(vals["state/foo_rate"].torch(), M.base[0, 7].torch())


def test_split_by_subaxis():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    xaxis = model.input_axis()
    yaxis = model.output_axis()
    assembler = neml2.MatrixAssembler(yaxis, xaxis)
    M = Tensor(torch.linspace(0, 1, 8).reshape(1, 8), 0)
    vals = assembler.split_by_subaxis(M)["residual"]
    assert torch.allclose(vals["forces"].torch(), M.base[0, 0].torch())
    assert torch.allclose(vals["old_forces"].torch(), M.base[0, 1].torch())
    assert torch.allclose(vals["old_state"].torch(), M.base[0, 2:4].torch())
    assert torch.allclose(vals["state"].torch(), M.base[0, 4:8].torch())
