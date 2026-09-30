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

r"""Linear-algebra operations on typed tensor wrappers.

This module mirrors the v2 C++ ``neml2/tensors/functions/linalg/`` namespace,
adapted to v3's Python-native typed-wrapper API. All functions here accept
and return typed wrappers (``Scalar``, ``Vec``, ``R2``, ``SR2``, ...) so the
no-raw-tensor rule (CLAUDE.md §"Hard rules") is preserved at every call site
outside ``neml2/types/``.

Currently provides:

  * :func:`eigh` — eigendecomposition of an ``SR2`` symmetric tensor.
  * :func:`transpose` — transpose of an ``R2``.
  * :func:`diag` — extract the diagonal of an ``R2`` as a ``Vec``.

These three together let downstream models compose eigenvalue chain rules
purely on the typed surface — see ``models/solid_mechanics/damage/`` for
the Mazars-family models that use them.
"""

from __future__ import annotations

import torch

from ._base import wrap_like
from .functions import r2_from_sr2
from .r2 import R2
from .sr2 import SR2
from .vec import Vec

__all__ = ["eigh", "transpose", "diag"]


def eigh(s: SR2) -> tuple[Vec, R2]:
    r"""Eigendecomposition of a symmetric second-order tensor.

    Computes :math:`\mathbf{s} = \mathbf{V}\,\mathrm{diag}(\boldsymbol{\lambda})\,\mathbf{V}^\top`
    of a Mandel-stored ``SR2``. Returns the principal values as a ``Vec``
    (3 components, in **ascending** order matching :func:`torch.linalg.eigh`)
    and the corresponding orthonormal principal directions as the columns
    of an ``R2``.

    Parameters
    ----------
    s
        Input symmetric second-order tensor.

    Returns
    -------
    eigvals
        ``Vec`` of three eigenvalues, ascending. Inherits ``s``'s sub-batch
        metadata.
    eigvecs
        ``R2`` whose columns are the unit-length principal directions.
        Column :math:`i` corresponds to eigenvalue :math:`\lambda_i`.

    Notes
    -----
    Autograd through this function is provided by
    :func:`torch.linalg.eigh`'s built-in derivative, with one important
    asymmetry at **degenerate eigenvalues** (any two or more eigenvalues
    equal — e.g. hydrostatic stress or axisymmetric loading with equal
    laterals):

    * **Eigenvalue gradients are well-defined** even at degeneracy.
      :math:`\partial \lambda_i/\partial S = v_i \otimes v_i` uses the
      value of the eigenvector but not its derivative; any representative
      from the degenerate eigenspace gives the same answer.
    * **Eigenvector gradients are NOT well-defined at degeneracy.**
      :math:`\partial v_i/\partial S` contains terms
      :math:`1/(\lambda_i - \lambda_j)` that diverge when eigenvalues
      coincide. ``torch.linalg.eigh`` returns NaN there.

    Practical consequence: callers that consume only eigenvalues
    (Mazars-family damage models — Macaulay brackets on principal
    strains, principal-stress sign decomposition) are SAFE at degenerate
    states. Callers that propagate gradients through the eigenvectors
    (e.g. anisotropic damage tensors of the form :math:`D = \sum_i D_i
    (n_i \otimes n_i)`) will hit NaN at degeneracy and need a
    degeneracy-aware alternative (Itskov 2002 / Dui-Chen 2004
    closed-form derivatives, or invariant-based reformulations).

    See ``tests/unit/types/test_linalg.py`` for empirical confirmation
    of both behaviours.

    Implementation note: Mandel-to-full conversion reuses
    :func:`r2_from_sr2`, so the off-diagonal :math:`\sqrt{2}` weights
    are handled consistently with the rest of the type system.
    """
    full = r2_from_sr2(s)
    eigvals, eigvecs = torch.linalg.eigh(full.data)
    return wrap_like(Vec, eigvals, s), wrap_like(R2, eigvecs, s)


def transpose(r: R2) -> R2:
    r"""Transpose of a second-order tensor.

    Returns :math:`\mathbf{R}^\top`. Sub-batch and K-region metadata
    propagate from the input. Useful for principal-frame triple products
    such as :math:`\mathbf{V}^\top \mathbf{S} \mathbf{V}` that arise in
    eigenprojection chain rules.
    """
    return wrap_like(R2, r.data.transpose(-2, -1), r)


def diag(r: R2) -> Vec:
    r"""Extract the diagonal of a second-order tensor as a 3-vector.

    Returns the ``Vec`` whose components are
    :math:`(R_{00}, R_{11}, R_{22})`. Sub-batch and K-region metadata
    propagate from the input.

    Used by eigenvalue chain rules: when :math:`\mathbf{R} =
    \mathbf{V}^\top \mathbf{S} \mathbf{V}` (an :math:`\mathbf{S}` rotated
    into the eigenframe of some other tensor), :func:`diag` extracts the
    principal-direction projections :math:`(v_i^\top \mathbf{S} v_i)_i`
    which are exactly the eigenvalue-tangent contributions in JVP form.
    """
    d = torch.diagonal(r.data, dim1=-2, dim2=-1)
    return wrap_like(Vec, d, r)
