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

"""MRP — rotation stored as modified Rodrigues parameters (MRPs).

Base shape ``(3,)``. The three components are ``n * tan(theta/4)`` where ``n``
is the rotation axis (unit vector) and ``theta`` the rotation angle. This is
the same convention as ``include/neml2/tensors/MRP.h``; it differs from the
standard Rodrigues parameters ``n * tan(theta/2)`` (which can be obtained by
the inverse map). The zero vector is the identity rotation.

MRPs aren't a vector space in the rotation-composition sense, but the
underlying 3-vector storage supports the usual scalar arithmetic. The Newton
residual in ``WR2ImplicitExponentialTimeIntegration`` uses ``-`` between MRPs
as elementwise 3-vector subtraction (correct as long as both sides are MRPs
of the same orientation up to roundoff). Composition itself goes through the
free :func:`compose` in :mod:`functions`.

Arithmetic operators and ``zeros``/``ones``/``full``/``empty``/``fill``
factories are inherited from :class:`PrimitiveTensor`. The only MRP-specific
factory is :meth:`identity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register

if TYPE_CHECKING:
    from neml2.types.scalar import Scalar
    from neml2.types.vec import Vec


@dataclass(frozen=True, eq=False)
class MRP(PrimitiveTensor):
    """Wraps a `torch.Tensor` of shape ``(..., 3)`` in MRP packing."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    sub_batch_state: tuple = ()
    sub_batch_meta: tuple = ()
    k_ndim: int = 0
    k_state: tuple = ()
    k_pairing: tuple = ()
    BASE_NDIM: ClassVar[int] = 1
    BASE_SHAPE: ClassVar[tuple[int, ...]] = (3,)

    @classmethod
    def identity(
        cls, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> MRP:
        """The identity rotation — the zero MRP vector."""
        return cls(torch.zeros(3, dtype=dtype, device=device))

    # ---- orientation constructors (plain batch; mirror v2 ``Rot``) ----

    @classmethod
    def rand(
        cls,
        *shape: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> MRP:
        """``shape`` orientations sampled uniformly on SO(3).

        Shoemake's method (``Rot::rand`` in ``src/neml2/tensors/Rot.cxx``): three
        uniform deviates build a uniform unit quaternion, converted to the MRP
        ``q_vec / (1 + q_w)``.
        """
        u = torch.rand(*shape, 3, dtype=dtype or torch.float64, device=device)
        u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]
        tau = 2.0 * math.pi
        w = torch.sqrt(1.0 - u0) * torch.sin(tau * u1)
        x = torch.sqrt(1.0 - u0) * torch.cos(tau * u1)
        y = torch.sqrt(u0) * torch.sin(tau * u2)
        z = torch.sqrt(u0) * torch.cos(tau * u2)
        return cls(torch.stack([x, y, z], dim=-1) / (1.0 + w).unsqueeze(-1))

    @classmethod
    def from_axis_angle(cls, n: Vec, theta: Scalar) -> MRP:
        """MRP for a rotation of ``theta`` about axis ``n`` (need not be unit).

        Matches ``Rot::axis_angle``: ``(n / |n|) * tan(theta / 4)``.
        """
        nn = n.data / torch.linalg.vector_norm(n.data, dim=-1, keepdim=True)
        return cls(nn * torch.tan(theta.data / 4.0).unsqueeze(-1), sub_batch_ndim=n.sub_batch_ndim)

    @classmethod
    def from_axis_angle_standard(cls, n: Vec, theta: Scalar) -> MRP:
        """MRP from a standard axis-angle (``Rot::axis_angle_standard``): the
        ``n * tan(theta / 2)`` Rodrigues vector mapped to MRP packing."""
        nn = n.data / torch.linalg.vector_norm(n.data, dim=-1, keepdim=True)
        vn = nn * torch.tan(theta.data / 2.0).unsqueeze(-1)  # standard Rodrigues
        nsq = (vn * vn).sum(dim=-1, keepdim=True)
        return cls(vn / (torch.sqrt(1.0 + nsq) + 1.0), sub_batch_ndim=n.sub_batch_ndim)

    @classmethod
    def rotation_from_to(cls, v1: Vec, v2: Vec) -> MRP:
        """MRP of the shortest rotation taking direction ``v1`` to ``v2``.

        Matches ``Rot::rotation_from_to``.
        """
        a, b = v1.data, v2.data
        n = torch.linalg.cross(a, b, dim=-1)
        c = (a * b).sum(dim=-1, keepdim=True)
        srp = n / (1.0 + c)
        nsrp = (srp * srp).sum(dim=-1, keepdim=True)
        return cls(srp / (torch.sqrt(1.0 + nsrp) + 1.0), sub_batch_ndim=v1.sub_batch_ndim)


register(MRP)
