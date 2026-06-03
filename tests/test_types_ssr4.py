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

from neml2.types import SR2, SSR4, Scalar, dev, outer, vol


def test_identity_sym_acts_as_eye():
    I = SSR4.identity_sym()
    assert I.shape == torch.Size([6, 6])
    assert torch.equal(I.data, torch.eye(6))


def test_identity_vol_projects_isotropic_part():
    I_vol = SSR4.identity_vol()
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.equal((I_vol @ a).data, vol(a).data)


def test_identity_dev_projects_deviatoric_part():
    I_dev = SSR4.identity_dev()
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.allclose((I_dev @ a).data, dev(a).data)


def test_identity_full_acts_as_vol_3x():
    """`SSR4.identity()` is δ_{ij}δ_{kl}: maps A to tr(A) * I."""
    I = SSR4.identity()
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    out = I @ a
    expected = torch.tensor([6.0, 6.0, 6.0, 0.0, 0.0, 0.0])  # tr(a) = 6
    assert torch.allclose(out.data, expected)


def test_ssr4_addition():
    A = SSR4.identity_sym()
    B = SSR4.identity_vol()
    C = A + B
    assert torch.equal(C.data, A.data + B.data)


def test_ssr4_scaled_by_scalar_and_float():
    A = SSR4.identity_sym()
    s = Scalar(torch.tensor(3.0))
    assert torch.equal((s * A).data, 3.0 * A.data)
    assert torch.equal((A * s).data, 3.0 * A.data)
    assert torch.equal((3.0 * A).data, 3.0 * A.data)


def test_ssr4_matmul_ssr4_chains():
    A = SSR4.identity_vol()
    B = SSR4.identity_vol()
    # I_vol @ I_vol == I_vol (projection)
    AB = A @ B
    assert torch.allclose(AB.data, A.data)


def test_outer_produces_symmetric_ssr4():
    a = SR2(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    O = outer(a, a)
    assert isinstance(O, SSR4)
    assert torch.equal(O.data, O.data.transpose(-2, -1))


def test_pytree_round_trip():
    A = SSR4.identity_sym()
    leaves, spec = pytree.tree_flatten(A)
    A2 = pytree.tree_unflatten(leaves, spec)
    assert isinstance(A2, SSR4)
    assert torch.equal(A2.data, A.data)


def test_torch_export_round_trip():
    class Identity(nn.Module):
        def forward(self, x: SSR4) -> SSR4:
            return SSR4(x.data * 1.0)

    A = SSR4.identity_sym()
    ep = torch.export.export(Identity(), (A,))
    assert len(ep.graph_signature.user_inputs) == 1
    assert len(ep.graph_signature.user_outputs) == 1
    out = ep.module()(A)
    assert isinstance(out, SSR4)
    assert torch.equal(out.data, A.data)
