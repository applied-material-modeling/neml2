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

"""Unit tests for ``neml2.types.linalg``."""

from __future__ import annotations

import torch

from neml2.types import R2, SR2, Vec
from neml2.types.functions import r2_from_sr2
from neml2.types.linalg import diag, eigh, transpose

torch.set_default_dtype(torch.double)


# ============================================================================
# eigh
# ============================================================================
def test_eigh_returns_typed_wrappers():
    """eigh must return Vec and R2, never raw tensors (Hard rule #1)."""
    s = SR2.fill(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    vals, vecs = eigh(s)
    assert isinstance(vals, Vec)
    assert isinstance(vecs, R2)


def test_eigh_diagonal_matches_diagonals():
    s = SR2.fill(2.0, -1.0, 3.0, 0.0, 0.0, 0.0)
    vals, _ = eigh(s)
    assert torch.allclose(vals.data, torch.tensor([-1.0, 2.0, 3.0]), atol=1e-14)


def test_eigh_reconstruction_machine_precision():
    s = SR2.fill(1.0, 2.0, -1.0, 0.3, -0.2, 0.5)
    full = r2_from_sr2(s).data
    vals, vecs = eigh(s)
    recon = vecs.data @ torch.diag_embed(vals.data) @ vecs.data.transpose(-2, -1)
    assert torch.allclose(recon, full, atol=1e-13)


def test_eigh_eigenvectors_orthonormal():
    s = SR2.fill(1.5, -0.7, 2.3, 0.4, 0.2, -0.3)
    _, vecs = eigh(s)
    VVt = vecs.data @ vecs.data.transpose(-2, -1)
    assert torch.allclose(VVt, torch.eye(3, dtype=torch.float64), atol=1e-13)


def test_eigh_eigenvalues_ascending():
    s = SR2.fill(5.0, -3.0, 1.0, 0.0, 0.0, 0.0)
    vals, _ = eigh(s)
    assert torch.all(vals.data[1:] >= vals.data[:-1])


def test_eigh_mandel_weight_consistency():
    """Eigh of an SR2 must agree with torch.linalg.eigh of the
    independently-built physical 3x3 — catches Mandel-weight mishandling."""
    s = SR2.fill(1.0, 2.0, -0.5, 0.3, -0.4, 0.6)
    vals_ours, _ = eigh(s)
    xx, yy, zz, yz, xz, xy = 1.0, 2.0, -0.5, 0.3, -0.4, 0.6
    full_indep = torch.tensor([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    vals_ref = torch.linalg.eigvalsh(full_indep)
    assert torch.allclose(vals_ours.data, vals_ref, atol=1e-13)


def test_eigh_autograd_matches_finite_difference():
    _SQRT2 = 2.0**0.5

    def eig2_of(md: torch.Tensor) -> torch.Tensor:
        s = SR2(md)
        vals, _ = eigh(s)
        return vals.data[2]

    md0 = torch.tensor(
        [1.0, 2.0, -0.5, 0.3 * _SQRT2, -0.4 * _SQRT2, 0.6 * _SQRT2],
        requires_grad=True,
    )
    eig2_of(md0).backward()
    assert md0.grad is not None
    grad_auto = md0.grad.clone()

    delta = 1e-6
    md_fixed = md0.detach()
    grad_fd = torch.zeros_like(md_fixed)
    for i in range(6):
        mp = md_fixed.clone()
        mp[i] += delta
        mm = md_fixed.clone()
        mm[i] -= delta
        grad_fd[i] = (eig2_of(mp).item() - eig2_of(mm).item()) / (2 * delta)
    rel = (grad_auto - grad_fd).abs() / grad_fd.abs().clamp_min(1e-12)
    assert torch.all(rel < 1e-4)


def test_eigh_eigenvalue_gradients_finite_at_2fold_degeneracy():
    """Mazars-relevant: at axisymmetric loading (λ_2 = λ_3 exactly),
    eigenvalue-only gradients must be finite."""
    md = torch.tensor([-1.0e-3, +2.0e-4, +2.0e-4, 0.0, 0.0, 0.0], requires_grad=True)
    s = SR2(md)
    vals, _ = eigh(s)
    assert torch.isclose(vals.data[1], vals.data[2], atol=1e-15)
    (vals.data**2).sum().backward()
    assert md.grad is not None
    assert torch.isfinite(md.grad).all()


def test_eigh_eigenvalue_gradients_finite_at_3fold_degeneracy():
    """Hydrostatic state."""
    md = torch.tensor([1.0e-3, 1.0e-3, 1.0e-3, 0.0, 0.0, 0.0], requires_grad=True)
    s = SR2(md)
    vals, _ = eigh(s)
    assert torch.allclose(vals.data, vals.data[0], atol=1e-15)
    (vals.data**2).sum().backward()
    assert md.grad is not None
    assert torch.isfinite(md.grad).all()


def test_eigh_eigenvector_gradients_documented_nan_at_degeneracy():
    """DOCUMENTED LIMITATION: eigenvector-explicit gradients are NaN at
    degeneracy. If a future degeneracy-safe eigh is added, UPDATE this
    test to assert finiteness + update the eigh docstring."""
    md = torch.tensor([-1.0e-3, +2.0e-4, +2.0e-4, 0.0, 0.0, 0.0], requires_grad=True)
    s = SR2(md)
    _, vecs = eigh(s)
    (vecs.data**2).sum().backward()
    assert md.grad is not None
    assert torch.isnan(md.grad).any(), (
        "Eigenvector gradients are NO LONGER NaN at degenerate eigenvalues. "
        "If you intentionally added a degeneracy-safe eigh, update this "
        "test and the eigh docstring."
    )


def test_eigh_preserves_sub_batch_metadata():
    md = torch.zeros(4, 6)
    md[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = SR2(md, sub_batch_ndim=0)
    vals, vecs = eigh(s)
    assert vals.sub_batch_ndim == 0
    assert vecs.sub_batch_ndim == 0
    assert vals.data.shape == (4, 3)
    assert vecs.data.shape == (4, 3, 3)


# ============================================================================
# transpose
# ============================================================================
def test_transpose_returns_R2():
    r = R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)
    rt = transpose(r)
    assert isinstance(rt, R2)


def test_transpose_swaps_axes():
    r = R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)
    rt = transpose(r)
    # Original is [[1,2,3],[4,5,6],[7,8,9]]; transpose is [[1,4,7],[2,5,8],[3,6,9]]
    expected = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    assert torch.allclose(rt.data, expected)


def test_transpose_symmetric_R2_is_idempotent():
    """If R2 happens to be symmetric, transpose returns the same data."""
    r = R2.fill(1, 2, 3, 2, 5, 6, 3, 6, 9)
    rt = transpose(r)
    assert torch.allclose(rt.data, r.data)


def test_transpose_twice_is_identity():
    r = R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert torch.allclose(transpose(transpose(r)).data, r.data)


# ============================================================================
# diag
# ============================================================================
def test_diag_returns_Vec():
    r = R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)
    d = diag(r)
    assert isinstance(d, Vec)


def test_diag_extracts_diagonal():
    r = R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)
    d = diag(r)
    assert torch.allclose(d.data, torch.tensor([1.0, 5.0, 9.0]))


def test_diag_of_identity_is_ones():
    r = R2.identity()
    d = diag(r)
    assert torch.allclose(d.data, torch.tensor([1.0, 1.0, 1.0]))


def test_diag_preserves_sub_batch_metadata():
    data = torch.zeros(4, 3, 3)
    data[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data[:, 1, 1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    r = R2(data, sub_batch_ndim=0)
    d = diag(r)
    assert d.sub_batch_ndim == 0
    assert d.data.shape == (4, 3)
    assert torch.allclose(d.data[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(d.data[:, 1], torch.tensor([5.0, 6.0, 7.0, 8.0]))


# ============================================================================
# Composition — eigenprojection JVP identity (used by Mazars classes)
# ============================================================================
def test_eigenprojection_identity():
    r"""diag(V^T S V) gives the per-eigenvalue tangents: when S = V·Λ·V^T,
    diag(V^T S V) = Λ. This is the building block of Mazars JVPs."""
    s = SR2.fill(1.0, 2.0, -1.0, 0.3, -0.2, 0.5)
    vals, vecs = eigh(s)
    s_full = r2_from_sr2(s)
    rotated = transpose(vecs) @ s_full @ vecs
    eigval_via_rotation = diag(rotated)
    assert torch.allclose(eigval_via_rotation.data, vals.data, atol=1e-12)
