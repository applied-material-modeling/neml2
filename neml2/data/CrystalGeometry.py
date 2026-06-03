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

"""Python-native crystallography Data base class.

Mirrors the C++ ``neml2::crystallography::CrystalGeometry`` ``[Data]`` block.
Each instance carries the precomputed slip-system geometry — symmetric Schmid
tensors ``M : SR2``, skew Schmid ``W : WR2``, full Schmid ``A : R2``, slip-group
offsets — shared by reference across every CP slip leaf (``ResolvedShear``,
``PlasticDeformationRate``, ``PlasticVorticity``).

The Python class is NOT a :class:`~neml2.model.Model`. It has no
``forward`` and never participates in chain-rule propagation — every Schmid
tensor is a constant the moment the input HIT file is loaded. Concrete
crystal systems (e.g. :class:`~neml2.data.CubicCrystal.CubicCrystal`)
construct a :class:`CrystalGeometry` from their own symmetry operators.
"""

from __future__ import annotations

import math
from typing import cast

import torch

from ..types import (
    R2,
    SR2,
    WR2,
    MillerIndex,
    skew,
    sym,
)

__all__ = ["CrystalGeometry"]


# ---------------------------------------------------------------------------
# Helpers — Miller -> Cartesian and symmetry-equivalent direction families
# ---------------------------------------------------------------------------


def _miller_to_cartesian(mi: MillerIndex, lattice: torch.Tensor) -> torch.Tensor:
    """Convert a single Miller index to a Cartesian ``(3,)`` raw tensor.

    ``lattice`` is the ``(3, 3)`` matrix whose rows are the three lattice
    vectors (or reciprocal lattice vectors). Mirrors the C++
    ``miller_to_cartesian`` static in ``CrystalGeometry.cxx`` after a GCD
    normalisation of the index.
    """
    # GCD normalisation — divide by gcd of components (handles e.g. (2,2,0) -> (1,1,0)).
    d = mi.data.to(dtype=torch.int64)
    if d.ndim != 1 or d.shape[0] != 3:
        raise ValueError(f"_miller_to_cartesian expects a single MillerIndex (3,), got {d.shape}")
    g = math.gcd(math.gcd(int(d[0].item()), int(d[1].item())), int(d[2].item()))
    if g == 0:
        g = 1
    dr = d.to(dtype=lattice.dtype) / float(g)
    # vdot of each lattice vector (row) with dr — yields per-axis components.
    return torch.einsum("ij,j->i", lattice, dr)


def _make_reciprocal_lattice(A: torch.Tensor) -> torch.Tensor:
    """Reciprocal lattice from real lattice (rows are vectors).

    $A$ is ``(3, 3)`` with rows ``a1, a2, a3``; output rows are
    $b1 = (a2 x a3) / (a1 . (a2 x a3))$ etc. — same convention as the C++
    ``make_reciprocal_lattice`` helper.
    """
    a1, a2, a3 = A[0], A[1], A[2]
    c23 = torch.linalg.cross(a2, a3)
    c31 = torch.linalg.cross(a3, a1)
    c12 = torch.linalg.cross(a1, a2)
    b1 = c23 / torch.dot(a1, c23)
    b2 = c31 / torch.dot(a2, c31)
    b3 = c12 / torch.dot(a3, c12)
    return torch.stack([b1, b2, b3], dim=0)


def _unique_bidirectional(ops: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Apply each symmetry op to ``inp`` and return unique vectors up to sign.

    ``ops`` is ``(N, 3, 3)``; ``inp`` is ``(3,)``; output is ``(K, 3)`` with
    ``K <= N``. Mirrors ``neml2::crystallography::unique_bidirectional``.
    """
    candidates = torch.einsum("nij,j->ni", ops, inp)  # (N, 3)
    unique: list[torch.Tensor] = [candidates[0]]
    for i in range(1, candidates.shape[0]):
        vi = candidates[i]
        # Compare against every stored vector, with sign ambiguity.
        keep = True
        for u in unique:
            if torch.allclose(u, vi) or torch.allclose(u, -vi):
                keep = False
                break
        if keep:
            unique.append(vi)
    return torch.stack(unique, dim=0)


def _unit(v: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Unit-norm a ``(*, 3)`` raw vector."""
    n = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps * eps)
    return v / n


def _setup_schmid_tensors(
    lattice: torch.Tensor,
    sym_ops: torch.Tensor,
    slip_directions: MillerIndex,
    slip_planes: MillerIndex,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Compute per-slip-system Cartesian directions, planes, burgers, group offsets.

    Mirrors ``CrystalGeometry::setup_schmid_tensors`` in
    ``src/neml2/models/crystallography/CrystalGeometry.cxx``. Returns:

    * ``cartesian_slip_directions``: ``(nslip, 3)``, unit vectors.
    * ``cartesian_slip_planes``: ``(nslip, 3)``, unit vectors.
    * ``burgers``: ``(nslip,)``, Burgers vector lengths.
    * ``offsets``: ``[0, n_grp_0, n_grp_0+n_grp_1, ..., nslip]``.

    Each input Miller family ``(d_i, n_i)`` produces one slip group; the
    perpendicular-pair filter discards symmetry images where the rotated
    direction is no longer perpendicular to the rotated plane normal.
    """
    B = _make_reciprocal_lattice(lattice)

    cmds = slip_directions.data
    cmps = slip_planes.data
    if cmds.ndim == 1:
        cmds = cmds.unsqueeze(0)
    if cmps.ndim == 1:
        cmps = cmps.unsqueeze(0)
    if cmds.shape != cmps.shape:
        raise ValueError(
            f"slip_directions and slip_planes must have matching shape; got "
            f"{tuple(cmds.shape)} vs {tuple(cmps.shape)}"
        )

    n_families = cmds.shape[0]
    dirs_acc: list[torch.Tensor] = []
    planes_acc: list[torch.Tensor] = []
    burgers_acc: list[torch.Tensor] = []
    offsets: list[int] = [0]

    for i in range(n_families):
        cmd = MillerIndex(cmds[i])
        cmp_ = MillerIndex(cmps[i])
        d_cart = _miller_to_cartesian(cmd, lattice)
        p_cart = _miller_to_cartesian(cmp_, B)

        direction_options = _unique_bidirectional(sym_ops, d_cart)
        plane_options = _unique_bidirectional(sym_ops, p_cart)

        last = offsets[-1]
        for j in range(direction_options.shape[0]):
            di = direction_options[j]
            # Dot di against every rotated plane normal.
            dps = torch.einsum("ni,i->n", plane_options, di)
            # Keep planes where |di . n_k| ~= 0 (perpendicular).
            zero = torch.zeros_like(dps)
            mask = torch.isclose(dps.abs(), zero)
            for k in torch.where(mask)[0].tolist():
                pi = plane_options[k]
                dirs_acc.append(_unit(di))
                planes_acc.append(_unit(pi))
                burgers_acc.append(torch.linalg.vector_norm(di))
                last += 1
        offsets.append(last)

    if not dirs_acc:
        raise RuntimeError(
            "No slip systems generated — check Miller indices and crystal class consistency."
        )

    cart_dirs = torch.stack(dirs_acc, dim=0)
    cart_planes = torch.stack(planes_acc, dim=0)
    burgers = torch.stack(burgers_acc, dim=0)
    return cart_dirs, cart_planes, burgers, offsets


# ---------------------------------------------------------------------------
# CrystalGeometry — the Data class consumers see
# ---------------------------------------------------------------------------


class CrystalGeometry(torch.nn.Module):
    r"""Slip-system geometry shared by all CP leaves in a single crystal model.

    Constructed once at HIT-load time by the factory; CP slip leaves
    (``ResolvedShear``, ``PlasticDeformationRate``, ``PlasticVorticity``)
    receive it by reference through their ``from_hit`` constructors and read
    $M$ / $W$ / $A$ in their forward pass.

    ``nn.Module`` is the parent so that assigning a ``CrystalGeometry`` to a
    CP leaf via ``self._cg = crystal_geometry`` registers it as a sub-module —
    ``leaf.to('cuda')`` then moves the Schmid / lattice buffers via the
    standard PyTorch buffer-propagation path. Without this, the buffers stay
    on CPU and Inductor trips a device-mismatch on the first ``bmm`` against
    a CUDA input during ``torch.export``. Wrapper-typed views ($A$, $M$,
    $W$, ``sym_ops``, ``slip_directions``, ``slip_planes``) are reconstructed
    on the fly from the underlying buffers so they always carry the current
    device.

    Buffer attributes
    -----------------
    sym_ops_data : torch.Tensor
        ``(nsym, 3, 3)`` proper-rotation symmetry operators.
    lattice_vectors : torch.Tensor
        ``(3, 3)`` real-space lattice vectors (rows).
    reciprocal_lattice_vectors : torch.Tensor
        ``(3, 3)`` reciprocal-lattice vectors (rows).
    cartesian_slip_directions, cartesian_slip_planes : torch.Tensor
        ``(nslip, 3)`` unit-vector representations.
    burgers : torch.Tensor
        ``(nslip,)`` Burgers vector lengths.
    A_data, M_data, W_data : torch.Tensor
        Raw storage for the wrapper-typed Schmid tensors (``(nslip, 3, 3)``,
        ``(nslip, 6)``, ``(nslip, 3)`` respectively).
    slip_directions_data, slip_planes_data : torch.Tensor
        Raw Miller-index storage.
    slip_offsets : list[int]
        ``[0, ..., nslip]`` group boundaries; the size of group ``i`` is
        ``slip_offsets[i+1] - slip_offsets[i]``. Plain Python list; not a buffer.

    Wrapper-typed properties
    ------------------------
    sym_ops : R2
    A : R2 (``sub_batch_ndim=1``)
    M : SR2 (``sub_batch_ndim=1``)
    W : WR2 (``sub_batch_ndim=1``)
    slip_directions, slip_planes : MillerIndex
    """

    def __init__(
        self,
        *,
        sym_ops: R2,
        lattice_vectors: torch.Tensor,
        slip_directions: MillerIndex,
        slip_planes: MillerIndex,
    ) -> None:
        super().__init__()
        # Raw-tensor buffers — these are what ``.to(device)`` moves.
        self.register_buffer("sym_ops_data", sym_ops.data)
        self.register_buffer("lattice_vectors", lattice_vectors)
        self.register_buffer(
            "reciprocal_lattice_vectors", _make_reciprocal_lattice(lattice_vectors)
        )
        self.register_buffer("slip_directions_data", slip_directions.data)
        self.register_buffer("slip_planes_data", slip_planes.data)

        cart_dirs, cart_planes, burgers, offsets = _setup_schmid_tensors(
            lattice_vectors, sym_ops.data, slip_directions, slip_planes
        )
        self.register_buffer("cartesian_slip_directions", cart_dirs)
        self.register_buffer("cartesian_slip_planes", cart_planes)
        self.register_buffer("burgers", burgers)
        self.slip_offsets = offsets

        # Schmid tensors live with sub_batch_ndim=1 (the leading nslip axis is
        # the per-site sub-batch dim — see D-049). Stored as raw buffers; the
        # ``A`` / ``M`` / ``W`` properties below rewrap on access.
        A_data = torch.einsum("ni,nj->nij", _unit(cart_dirs), _unit(cart_planes))
        A_r2 = R2(A_data, sub_batch_ndim=1)
        self.register_buffer("A_data", A_data)
        self.register_buffer("M_data", sym(A_r2).data)
        self.register_buffer("W_data", skew(A_r2).data)

    # ---- wrapper-typed views over the raw buffers ----
    # Each rebuilds the wrapper from the current buffer state so the wrapper's
    # device tracks the module's device after ``.to(...)``.

    @property
    def sym_ops(self) -> R2:
        return R2(cast(torch.Tensor, self.sym_ops_data))

    @property
    def A(self) -> R2:
        return R2(cast(torch.Tensor, self.A_data), sub_batch_ndim=1)

    @property
    def M(self) -> SR2:
        return SR2(cast(torch.Tensor, self.M_data), sub_batch_ndim=1)

    @property
    def W(self) -> WR2:
        return WR2(cast(torch.Tensor, self.W_data), sub_batch_ndim=1)

    @property
    def slip_directions(self) -> MillerIndex:
        return MillerIndex(cast(torch.Tensor, self.slip_directions_data))

    @property
    def slip_planes(self) -> MillerIndex:
        return MillerIndex(cast(torch.Tensor, self.slip_planes_data))

    # ---- shape accessors ----

    @property
    def nslip(self) -> int:
        return self.slip_offsets[-1]

    @property
    def nslip_groups(self) -> int:
        return len(self.slip_offsets) - 1

    def nslip_in_group(self, i: int) -> int:
        if not 0 <= i < self.nslip_groups:
            raise IndexError(f"slip group {i} out of range [0, {self.nslip_groups})")
        return self.slip_offsets[i + 1] - self.slip_offsets[i]
