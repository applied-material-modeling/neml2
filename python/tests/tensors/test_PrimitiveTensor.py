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
from common import *
import neml2


def test_named_ctors(tensor_options):
    s = TensorShape(dynamic=(2, 3), intmd=(2, 1), base=(3, 3))

    # empty
    A = neml2.R2.empty(**tensor_options)
    assert A.batch.dim() == 0
    assert A.base.shape == s.base_shape

    A = neml2.R2.empty(s.dynamic_shape, s.intmd_shape, **tensor_options)
    assert_tensor_shape(A, s)

    # zeros
    A = neml2.R2.zeros(**tensor_options)
    assert A.batch.dim() == 0
    assert A.base.shape == s.base_shape
    assert torch.allclose(A.torch(), torch.zeros(s.base_shape, **tensor_options))

    A = neml2.R2.zeros(s.dynamic_shape, s.intmd_shape, **tensor_options)
    assert_tensor_shape(A, s)
    assert torch.allclose(A.torch(), torch.zeros(s.shape, **tensor_options))

    # ones
    A = neml2.R2.ones(**tensor_options)
    assert A.batch.dim() == 0
    assert A.base.shape == s.base_shape
    assert torch.allclose(A.torch(), torch.ones(s.base_shape, **tensor_options))

    A = neml2.R2.ones(s.dynamic_shape, s.intmd_shape, **tensor_options)
    assert_tensor_shape(A, s)
    assert torch.allclose(A.torch(), torch.ones(s.shape, **tensor_options))

    # full
    A = neml2.R2.full(1.1, **tensor_options)
    assert A.batch.dim() == 0
    assert A.base.shape == s.base_shape
    assert torch.allclose(A.torch(), torch.full(s.base_shape, 1.1, **tensor_options))

    A = neml2.R2.full(s.dynamic_shape, s.intmd_shape, 2.3, **tensor_options)
    assert_tensor_shape(A, s)
    assert torch.allclose(A.torch(), torch.full(s.shape, 2.3, **tensor_options))
