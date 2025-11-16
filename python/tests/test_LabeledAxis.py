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


def test_size():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    input_axis = model.input_axis()
    output_axis = model.output_axis()

    assert input_axis.size() == 8
    assert input_axis.size("forces") == 1
    assert input_axis.size("forces/t") == 1
    assert input_axis.size("old_forces") == 1
    assert input_axis.size("old_forces/t") == 1
    assert input_axis.size("old_state") == 2
    assert input_axis.size("old_state/bar") == 1
    assert input_axis.size("old_state/foo") == 1
    assert input_axis.size("state") == 4
    assert input_axis.size("state/bar") == 1
    assert input_axis.size("state/bar_rate") == 1
    assert input_axis.size("state/foo") == 1
    assert input_axis.size("state/foo_rate") == 1
    assert output_axis.size() == 1
    assert output_axis.size("residual") == 1
    assert output_axis.size("residual/foo_bar") == 1


def test_slice():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    input_axis = model.input_axis()
    output_axis = model.output_axis()

    assert input_axis.slice("forces") == slice(0, 1, 1)
    assert input_axis.slice("forces/t") == slice(0, 1, 1)
    assert input_axis.slice("old_forces") == slice(1, 2, 1)
    assert input_axis.slice("old_forces/t") == slice(1, 2, 1)
    assert input_axis.slice("old_state") == slice(2, 4, 1)
    assert input_axis.slice("old_state/bar") == slice(2, 3, 1)
    assert input_axis.slice("old_state/foo") == slice(3, 4, 1)
    assert input_axis.slice("state") == slice(4, 8, 1)
    assert input_axis.slice("state/bar") == slice(4, 5, 1)
    assert input_axis.slice("state/bar_rate") == slice(5, 6, 1)
    assert input_axis.slice("state/foo") == slice(6, 7, 1)
    assert input_axis.slice("state/foo_rate") == slice(7, 8, 1)
    assert output_axis.slice("residual") == slice(0, 1, 1)
    assert output_axis.slice("residual/foo_bar") == slice(0, 1, 1)


def test_variable_accessors():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    axis = model.input_axis()

    assert axis.nvariable() == 8
    assert axis.has_variable("forces/t")
    assert not axis.has_variable("nonexistent")
    assert axis.variable_id("forces/t") == 0
    assert axis.variable_id("old_forces/t") == 1
    assert axis.variable_id("old_state/bar") == 2
    assert axis.variable_id("old_state/foo") == 3
    assert axis.variable_id("state/bar") == 4
    assert axis.variable_id("state/bar_rate") == 5
    assert axis.variable_id("state/foo") == 6
    assert axis.variable_id("state/foo_rate") == 7
    assert axis.variable_names() == [
        "forces/t",
        "old_forces/t",
        "old_state/bar",
        "old_state/foo",
        "state/bar",
        "state/bar_rate",
        "state/foo",
        "state/foo_rate",
    ]
    assert axis.variable_slices() == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
        slice(3, 4, 1),
        slice(4, 5, 1),
        slice(5, 6, 1),
        slice(6, 7, 1),
        slice(7, 8, 1),
    ]
    assert axis.variable_slice("forces/t") == slice(0, 1, 1)
    assert axis.variable_slice("old_forces/t") == slice(1, 2, 1)
    assert axis.variable_slice("old_state/bar") == slice(2, 3, 1)
    assert axis.variable_slice("old_state/foo") == slice(3, 4, 1)
    assert axis.variable_slice("state/bar") == slice(4, 5, 1)
    assert axis.variable_slice("state/bar_rate") == slice(5, 6, 1)
    assert axis.variable_slice("state/foo") == slice(6, 7, 1)
    assert axis.variable_slice("state/foo_rate") == slice(7, 8, 1)
    assert axis.variable_sizes() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert axis.variable_size("forces/t") == 1
    assert axis.variable_size("old_forces/t") == 1
    assert axis.variable_size("old_state/bar") == 1
    assert axis.variable_size("old_state/foo") == 1
    assert axis.variable_size("state/bar") == 1
    assert axis.variable_size("state/bar_rate") == 1
    assert axis.variable_size("state/foo") == 1
    assert axis.variable_size("state/foo_rate") == 1


def test_subaxis_accessors():
    pwd = Path(__file__).parent
    model = neml2.load_model(pwd / "test_LabeledAxis.i", "model")
    axis = model.input_axis()

    assert axis.nsubaxis() == 4
    assert axis.has_subaxis("forces")
    assert not axis.has_subaxis("nonexistent")
    assert axis.subaxis_id("forces") == 0
    assert axis.subaxis_id("old_forces") == 1
    assert axis.subaxis_id("old_state") == 2
    assert axis.subaxis_id("state") == 3
    assert axis.subaxis_names() == ["forces", "old_forces", "old_state", "state"]
    assert axis.subaxis_slices() == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 4, 1),
        slice(4, 8, 1),
    ]
    assert axis.subaxis_slice("forces") == slice(0, 1, 1)
    assert axis.subaxis_slice("old_forces") == slice(1, 2, 1)
    assert axis.subaxis_slice("old_state") == slice(2, 4, 1)
    assert axis.subaxis_slice("state") == slice(4, 8, 1)
    assert axis.subaxis_sizes() == [1, 1, 2, 4]
    assert axis.subaxis_size("forces") == 1
    assert axis.subaxis_size("old_forces") == 1
    assert axis.subaxis_size("old_state") == 2
    assert axis.subaxis_size("state") == 4
