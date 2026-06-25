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

import torch
from torch import nn
from torch.utils import _pytree as pytree

from neml2.types import SR2, Scalar, dev, norm, tr, unit, vol


def test_identity_packing():
    I = SR2.identity()
    assert torch.equal(I.data, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert I.BASE_NDIM == 1
    assert I.BASE_SHAPE == (6,)


def test_zeros_and_batch_shape():
    z = SR2.zeros(3, 4)
    assert z.shape == torch.Size([3, 4, 6])
    assert z.batch_shape == torch.Size([3, 4])


def test_tr_vol_dev():
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert tr(a).data.item() == 6.0
    # vol(a) = (tr(a) / 3) * I
    assert torch.equal(vol(a).data, torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0, 0.0]))
    # dev(a) = a - vol(a)
    assert torch.equal(dev(a).data, a.data - vol(a).data)
    # dev is traceless
    assert torch.isclose(tr(dev(a)).data, torch.tensor(0.0))


def test_norm_and_unit():
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    # Frobenius in Mandel = L2 of components (the sqrt(2) packing already does the off-diag scaling)
    expected = torch.sqrt(torch.tensor(1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0))
    assert torch.isclose(norm(a).data, expected)
    u = unit(a)
    assert torch.isclose(norm(u).data, torch.tensor(1.0))


def test_norm_eps_protects_zero():
    # ``eps`` is added unsquared under the sqrt (matches v2 ``sqrt(norm_sq(a) + eps)``),
    # so ``norm(0, eps) == sqrt(eps)``.
    z = SR2(torch.zeros(6))
    assert torch.isclose(norm(z, eps=1e-12).data, torch.tensor(1e-6))


def test_arithmetic_and_negation():
    a = SR2(torch.arange(6.0))
    b = SR2(torch.ones(6))
    assert torch.equal((a + b).data, a.data + 1.0)
    assert torch.equal((a - b).data, a.data - 1.0)
    assert torch.equal((-a).data, -a.data)


def test_scalar_times_sr2_and_float_times_sr2():
    a = SR2(torch.arange(6.0))
    s = Scalar(torch.tensor(3.0))
    # SR2 * Scalar -> SR2.__mul__
    r1 = a * s
    # Scalar * SR2 -> SR2.__rmul__ (Scalar.__mul__ returns NotImplemented for SR2)
    r2 = s * a
    # float * SR2 -> SR2.__rmul__
    r3 = 3.0 * a
    expected = a.data * 3.0
    for r in (r1, r2, r3):
        assert isinstance(r, SR2)
        assert torch.equal(r.data, expected)


def test_division_by_scalar():
    a = SR2(torch.arange(6.0))
    s = Scalar(torch.tensor(2.0))
    r = a / s
    assert isinstance(r, SR2)
    assert torch.equal(r.data, a.data / 2.0)


def test_batched_broadcast():
    batch_a = SR2(torch.arange(12.0).reshape(2, 6))
    s = Scalar(torch.tensor([2.0, 3.0]))
    r = batch_a * s
    # Each batch row scaled independently
    assert torch.equal(r.data[0], batch_a.data[0] * 2.0)
    assert torch.equal(r.data[1], batch_a.data[1] * 3.0)


def test_pytree_round_trip():
    a = SR2(torch.arange(12.0).reshape(2, 6))
    leaves, spec = pytree.tree_flatten(a)
    assert len(leaves) == 1
    a2 = pytree.tree_unflatten(leaves, spec)
    assert isinstance(a2, SR2)
    assert torch.equal(a2.data, a.data)


def test_torch_export_round_trip():
    class Identity(nn.Module):
        def forward(self, x: SR2) -> SR2:
            return SR2(x.data * 1.0)

    a = SR2(torch.arange(6.0))
    ep = torch.export.export(Identity(), (a,))
    assert len(ep.graph_signature.user_inputs) == 1
    assert len(ep.graph_signature.user_outputs) == 1
    out = ep.module()(a)
    assert isinstance(out, SR2)
    assert torch.equal(out.data, a.data)
