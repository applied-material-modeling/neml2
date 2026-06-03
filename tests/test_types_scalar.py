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

import math

import torch
from torch import nn
from torch.utils import _pytree as pytree

from neml2.types import SR2, Scalar, exp, sqrt


def test_construction_and_metadata():
    s = Scalar(torch.tensor(1.5))
    assert s.data.item() == 1.5
    assert s.dtype == torch.float64
    assert s.BASE_NDIM == 0
    assert s.BASE_SHAPE == ()
    assert s.batch_shape == torch.Size([])


def test_batched_scalar():
    s = Scalar(torch.arange(4.0))
    assert s.batch_shape == torch.Size([4])


def test_from_value_inherits_options():
    other = Scalar(torch.zeros(3, dtype=torch.float32))
    s = Scalar.from_value(2.0, like=other)
    assert s.dtype == torch.float32
    assert s.data.item() == 2.0


def test_unary_ops():
    s = Scalar(torch.tensor(4.0))
    assert sqrt(s).data.item() == 2.0
    assert math.isclose(exp(s).data.item(), math.exp(4.0))
    assert (s**3).data.item() == 64.0
    assert abs(-s).data.item() == 4.0


def test_arithmetic_with_scalar_and_python_numbers():
    a, b = Scalar(torch.tensor(3.0)), Scalar(torch.tensor(2.0))
    assert (a + b).data.item() == 5.0
    assert (a - b).data.item() == 1.0
    assert (a * b).data.item() == 6.0
    assert (a / b).data.item() == 1.5
    # Reflected ops against floats
    assert (1.0 + a).data.item() == 4.0
    assert (1.0 - a).data.item() == -2.0
    assert (2.0 * a).data.item() == 6.0
    assert (12.0 / a).data.item() == 4.0


def test_pytree_round_trip():
    s = Scalar(torch.tensor([1.0, 2.0, 3.0]))
    leaves, spec = pytree.tree_flatten(s)
    assert len(leaves) == 1 and torch.equal(leaves[0], s.data)
    s2 = pytree.tree_unflatten(leaves, spec)
    assert isinstance(s2, Scalar)
    assert torch.equal(s2.data, s.data)


def test_torch_export_round_trip():
    """Confirms `register_dataclass` makes Scalar a first-class export I/O type."""

    class Identity(nn.Module):
        def forward(self, x: Scalar) -> Scalar:
            return Scalar(x.data * 1.0)

    s = Scalar(torch.tensor([1.0, 2.0]))
    ep = torch.export.export(Identity(), (s,))
    # 1 user input (the flattened tensor), 1 user output.
    assert len(ep.graph_signature.user_inputs) == 1
    assert len(ep.graph_signature.user_outputs) == 1
    out = ep.module()(s)
    assert isinstance(out, Scalar)
    assert torch.equal(out.data, s.data)


def test_scalar_times_sr2_dispatches_via_rmul():
    """The cross-type contract: `Scalar * SR2` goes through SR2.__rmul__."""
    s = Scalar(torch.tensor(2.0))
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    result = s * a
    assert isinstance(result, SR2)
    assert torch.equal(result.data, a.data * 2.0)
