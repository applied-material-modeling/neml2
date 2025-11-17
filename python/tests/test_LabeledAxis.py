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
