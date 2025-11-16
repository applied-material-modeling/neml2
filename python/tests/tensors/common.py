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
import torch
import neml2
from typing import NamedTuple
from functools import reduce


class TensorShapeBase(NamedTuple):
    shape: tuple[int, ...]
    dynamic_shape: tuple[int, ...]
    intmd_shape: tuple[int, ...]
    base_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    static_shape: tuple[int, ...]

    dim: int
    dynamic_dim: int
    intmd_dim: int
    base_dim: int
    batch_dim: int
    static_dim: int


class TensorShape(TensorShapeBase):
    def __new__(cls, dynamic: tuple[int, ...], intmd: tuple[int, ...], base: tuple[int, ...]):
        batch = dynamic + intmd
        static = intmd + base
        shape = dynamic + intmd + base

        dynamic_dim = len(dynamic)
        intmd_dim = len(intmd)
        base_dim = len(base)
        batch_dim = len(batch)
        static_dim = len(static)
        dim = len(shape)

        obj = super().__new__(
            cls,
            shape,
            dynamic,
            intmd,
            base,
            batch,
            static,
            dim,
            dynamic_dim,
            intmd_dim,
            base_dim,
            batch_dim,
            static_dim,
        )
        return obj

    @classmethod
    def broadcast(cls, s1, s2):
        return TensorShape(
            torch.broadcast_shapes(s1.dynamic_shape, s2.dynamic_shape),
            torch.broadcast_shapes(s1.intmd_shape, s2.intmd_shape),
            torch.broadcast_shapes(s1.base_shape, s2.base_shape),
        )


def _stringify_fixture(v):
    return str(v)


@pytest.fixture(params=[torch.float64], ids=_stringify_fixture)
def dtype(request):
    """dtype being tested"""
    return request.param


@pytest.fixture(params=["cpu", "cuda:0"], ids=_stringify_fixture)
def device(request):
    """
    device being tested

    CUDA is skipped if not available
    """
    if request.param == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(params=[False], ids=["no grad"])
def requires_grad(request):
    """
    requires_grad
    """
    return request.param


@pytest.fixture
def tensor_options(dtype, device, requires_grad):
    """tensor options being tested"""
    return {"dtype": dtype, "device": device, "requires_grad": requires_grad}


def make_sample(tensor_shape: TensorShape, **tensor_options):
    """
    Create a sample Tensor with given batch and base shapes
    """
    numel = reduce(lambda x, y: x * y, tensor_shape.shape, 1)
    data = torch.arange(numel, **tensor_options).reshape(tensor_shape.shape)
    data -= data.mean() * 1.1
    return neml2.Tensor(data, tensor_shape.dynamic_dim, tensor_shape.intmd_dim)


def assert_tensor_shape(A, tensor_shape: TensorShape):
    assert A.dynamic.shape == tensor_shape.dynamic_shape
    assert A.intmd.shape == tensor_shape.intmd_shape
    assert A.base.shape == tensor_shape.base_shape
    assert A.batch.shape == tensor_shape.batch_shape
    assert A.static.shape == tensor_shape.static_shape
    assert A.dynamic.dim() == tensor_shape.dynamic_dim
    assert A.intmd.dim() == tensor_shape.intmd_dim
    assert A.base.dim() == tensor_shape.base_dim
    assert A.batch.dim() == tensor_shape.batch_dim
    assert A.static.dim() == tensor_shape.static_dim


def assert_unary_op(func, x):
    r = func(x)
    s = TensorShape(
        x.dynamic.shape,
        x.intmd.shape,
        x.base.shape,
    )
    assert_tensor_shape(r, s)


def assert_binary_op(func, x, y):
    r = func(x, y)
    s = TensorShape.broadcast(
        TensorShape(
            x.dynamic.shape,
            x.intmd.shape,
            x.base.shape,
        ),
        TensorShape(
            y.dynamic.shape,
            y.intmd.shape,
            y.base.shape,
        ),
    )
    assert_tensor_shape(r, s)
