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

# fixtures
from common import *
import neml2


@pytest.fixture
def tensor_shape_A():
    """
    Sample tensor shape A, unbatched
    """
    dynamic_shape = ()
    intmd_shape = ()
    base_shape = (2,)
    return TensorShape(dynamic_shape, intmd_shape, base_shape)


@pytest.fixture
def tensor_shape_B():
    """
    Sample tensor shape B, non-trivially broadcastable with A
    """
    dynamic_shape = (2, 1)
    intmd_shape = (2,)
    base_shape = (3, 1, 2)
    return TensorShape(dynamic_shape, intmd_shape, base_shape)


@pytest.fixture
def tensor_shape_C():
    """
    Sample tensor shape C, non-trivially broadcastable with A and B
    """
    dynamic_shape = (2, 1, 3)
    intmd_shape = (3, 1)
    base_shape = (1, 2)
    return TensorShape(dynamic_shape, intmd_shape, base_shape)


@pytest.fixture(params=["tensor_shape_A", "tensor_shape_B", "tensor_shape_C"], ids=["A", "B", "C"])
def tensor_shape(request):
    return request.getfixturevalue(request.param)


def test_empty_ctor():
    A = neml2.Tensor()
    assert not A.defined()
    # assert A.dtype == torch.get_default_dtype()


def test_ctor(tensor_shape: TensorShape, tensor_options):
    # From a torch.Tensor
    A0 = torch.rand(*tensor_shape.shape, **tensor_options)
    A = neml2.Tensor(A0, tensor_shape.dynamic_dim, tensor_shape.intmd_dim)
    assert torch.allclose(A.torch(), A0)
    assert_tensor_shape(A, tensor_shape)

    # From another Tensor
    sample = make_sample(tensor_shape, **tensor_options)
    A = neml2.Tensor(sample)
    assert torch.allclose(A.torch(), sample.torch())
    assert_tensor_shape(A, tensor_shape)


def test_view(tensor_shape: TensorShape, tensor_options):
    sample = make_sample(tensor_shape, **tensor_options)
    sample0 = sample.clone()

    # __getitem__
    assert torch.allclose(sample.dynamic[None].torch(), sample.torch()[None])
    assert torch.allclose(sample.dynamic[...].torch(), sample.torch()[...])
    if sample.dynamic.dim() > 0:
        assert torch.allclose(sample.dynamic[0].torch(), sample.torch()[0])
        assert torch.allclose(sample.dynamic[0:5:2].torch(), sample.torch()[0:5:2])
        assert torch.allclose(sample.dynamic[:, 0].torch(), sample.torch()[:, 0])

    # __setitem__
    sample0.dynamic[...] = sample.torch() + 1.3
    assert torch.allclose(sample0.torch(), sample.torch() + 1.3)

    # expand
    target_dynamic_shape = (7, 2) + tensor_shape.dynamic_shape
    target_shape = target_dynamic_shape + tensor_shape.static_shape
    B = sample.dynamic.expand(target_dynamic_shape)
    assert torch.allclose(B.torch(), sample.torch().expand(target_shape))

    # unsqueeze
    B = sample.dynamic.unsqueeze(0)
    assert torch.allclose(B.torch(), sample.torch().unsqueeze(0))
    B = sample.dynamic.unsqueeze(-1)
    assert torch.allclose(B.torch(), sample.torch().unsqueeze(tensor_shape.dynamic_dim))

    # transpose
    if tensor_shape.batch_dim >= 2:
        B = sample.dynamic.transpose(0, 1)
        assert torch.allclose(B.torch(), torch.transpose(sample.torch(), 0, 1))


def test_binary_ops(
    tensor_shape_A: TensorShape,
    tensor_shape_B: TensorShape,
    tensor_shape_C: TensorShape,
    tensor_options,
):
    A = make_sample(tensor_shape_A, **tensor_options)
    B = make_sample(tensor_shape_B, **tensor_options)
    C = make_sample(tensor_shape_C, **tensor_options)

    # add
    assert_unary_op(lambda x: x + 0.5, A)
    assert_unary_op(lambda x: x + 0.5, B)
    assert_unary_op(lambda x: 0.5 + x, A)
    assert_unary_op(lambda x: 0.5 + x, B)
    assert_binary_op(lambda x, y: x + y, A, B)
    assert_binary_op(lambda x, y: x + y, B, C)
    # sub
    assert_unary_op(lambda x: x - 0.5, A)
    assert_unary_op(lambda x: x - 0.5, B)
    assert_unary_op(lambda x: 0.5 - x, A)
    assert_unary_op(lambda x: 0.5 - x, B)
    assert_binary_op(lambda x, y: x - y, A, B)
    assert_binary_op(lambda x, y: x - y, B, C)
    # mul
    assert_unary_op(lambda x: x * 0.5, A)
    assert_unary_op(lambda x: x * 0.5, B)
    assert_unary_op(lambda x: 0.5 * x, A)
    assert_unary_op(lambda x: 0.5 * x, B)
    assert_binary_op(lambda x, y: x * y, A, B)
    assert_binary_op(lambda x, y: x * y, B, C)
    # div
    assert_unary_op(lambda x: x / 0.5, A)
    assert_unary_op(lambda x: x / 0.5, B)
    assert_unary_op(lambda x: 0.5 / x, A)
    assert_unary_op(lambda x: 0.5 / x, B)
    assert_binary_op(lambda x, y: x / y, A, B)
    assert_binary_op(lambda x, y: x / y, B, C)
    # pow
    assert_unary_op(lambda x: x**0.5, A)
    assert_unary_op(lambda x: x**0.5, B)
    assert_unary_op(lambda x: 0.5**x, A)
    assert_unary_op(lambda x: 0.5**x, B)
    assert_binary_op(lambda x, y: x**y, A, B)
    assert_binary_op(lambda x, y: x**y, B, C)


def test_unary_ops(tensor_shape: TensorShape, tensor_options):
    sample = make_sample(tensor_shape, **tensor_options)

    # neg
    assert_unary_op(lambda x: -x, sample)


def test_named_ctors(tensor_shape: TensorShape, tensor_options):
    sample = make_sample(tensor_shape, **tensor_options)

    # empty_like
    A = neml2.Tensor.empty_like(sample)
    assert_tensor_shape(A, tensor_shape)

    # zeros_like
    A = neml2.Tensor.zeros_like(A)
    A0 = torch.zeros_like(sample.torch())
    assert_tensor_shape(A, tensor_shape)
    assert torch.allclose(A.torch(), A0)

    # ones_like
    A = neml2.Tensor.ones_like(A)
    assert_tensor_shape(A, tensor_shape)
    assert torch.allclose(A.torch(), torch.ones_like(sample.torch()))

    # full_like
    A = neml2.Tensor.full_like(sample, 1.1)
    assert_tensor_shape(A, tensor_shape)
    assert torch.allclose(A.torch(), torch.full_like(sample.torch(), 1.1))
