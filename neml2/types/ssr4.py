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

"""SSR4 — symmetric rank-4 tensor in Mandel packing.

Base shape ``(6, 6)``; both index pairs symmetric, packed in Mandel basis so
``SSR4 : SR2`` is a plain matrix-vector product.

Arithmetic operators and ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. SSR4-specific:
several identity / projector factories (full identity, symmetric identity,
volumetric / deviatoric / cubic-symmetry projectors) and a polymorphic
:meth:`__matmul__` that contracts with either ``SR2`` (returning ``SR2``)
or ``SSR4`` (returning ``SSR4``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from neml2.types._base import align_sub_batch
from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register
from neml2.types.sr2 import SR2


@dataclass(frozen=True, eq=False)
class SSR4(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 6, 6)`` in Mandel packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 2
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (6, 6)

    # ---- identity / projector factories ----

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Full identity δ_{ij}δ_{kl} in Mandel packing.

        In the Mandel basis this is a 3x3 block of 1.0 in the top-left
        (volumetric) corner with all off-diagonal/deviatoric entries zero,
        i.e. it acts as ``A -> tr(A) * I`` rather than ``A -> A``.
        Distinct from :meth:`identity_sym` (which is the (6,6) eye).
        """
        I = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        return cls(I.unsqueeze(-1) * I.unsqueeze(-2))

    @classmethod
    def identity_sym(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Symmetric identity ``I_sym : A = A`` for any SR2 $A$ — the (6,6) eye."""
        return cls(torch.eye(6, dtype=dtype, device=device))

    @classmethod
    def identity_vol(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Volumetric projection ``vol(A) = I_vol : A``."""
        I = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).unsqueeze(-1)
        return cls((I @ I.transpose(-2, -1)) / 3.0)

    @classmethod
    def identity_dev(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Deviatoric projection ``dev(A) = I_dev : A`` — equals ``I_sym - I_vol``."""
        return SSR4.identity_sym(dtype=dtype, device=device) - SSR4.identity_vol(
            dtype=dtype, device=device
        )

    @classmethod
    def identity_C1(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Cubic-symmetry projector $I_C1$: top-left 3x3 identity block of the
        Mandel (6, 6) matrix; selects the cubic on-diagonal normal-stress
        coefficient (matches C++ ``SSR4::identity_C1``).
        """
        M = torch.zeros((6, 6), dtype=dtype, device=device)
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        M[2, 2] = 1.0
        return cls(M)

    @classmethod
    def identity_C2(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Cubic-symmetry projector $I_C2$: the three normal-stress off-diagonal
        ones in the top-left 3x3 Mandel block; selects the cubic off-diagonal
        normal-stress coefficient (matches C++ ``SSR4::identity_C2``).
        """
        M = torch.zeros((6, 6), dtype=dtype, device=device)
        M[0, 1] = 1.0
        M[0, 2] = 1.0
        M[1, 0] = 1.0
        M[1, 2] = 1.0
        M[2, 0] = 1.0
        M[2, 1] = 1.0
        return cls(M)

    @classmethod
    def identity_C3(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> SSR4:
        """Cubic-symmetry projector $I_C3$: bottom-right 3x3 identity block of
        the Mandel (6, 6) matrix; selects the cubic shear coefficient (matches
        C++ ``SSR4::identity_C3``).
        """
        M = torch.zeros((6, 6), dtype=dtype, device=device)
        M[3, 3] = 1.0
        M[4, 4] = 1.0
        M[5, 5] = 1.0
        return cls(M)

    # ---- mixed-type matmul (stays here because it changes return type) ----

    def __matmul__(self, other):
        """SSR4 @ SR2 → SR2 (contraction ``C : A``) or SSR4 @ SSR4 → SSR4."""
        if isinstance(other, SR2):
            [aa, bb], sb = align_sub_batch(self, other)
            # ``"...ij,...j->...i"`` is a 6×6 · 6 matvec — explicit matmul (no einsum).
            return SR2(
                (aa.data @ bb.data.unsqueeze(-1)).squeeze(-1),
                sub_batch_ndim=sb,
            )
        if isinstance(other, SSR4):
            [aa, bb], sb = align_sub_batch(self, other)
            # ``"...ij,...jk->...ik"`` is a 6×6 · 6×6 matmul.
            return SSR4(
                aa.data @ bb.data,
                sub_batch_ndim=sb,
            )
        return NotImplemented


register(SSR4)
