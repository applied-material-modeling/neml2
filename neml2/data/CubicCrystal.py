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

"""Python-native ``CubicCrystal`` Data block.

Mirrors the C++ ``neml2::crystallography::CubicCrystal`` ``[Data]`` block: a
specialization of :class:`~neml2.data.CrystalGeometry.CrystalGeometry`
that fixes the symmetry operators to the 24 proper rotations of the cubic
"432" point group and builds the lattice from a single lattice parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..factory import register_neml2_object
from ..schema import HitSchema, parameter
from ..types import (
    R2,
    MillerIndex,
    Scalar,
)
from .CrystalGeometry import CrystalGeometry

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile

__all__ = ["CubicCrystal", "cubic_symmetry_operators"]


# ---------------------------------------------------------------------------
# Cubic symmetry — 24 proper-rotation quaternions for the "432" orbifold
# ---------------------------------------------------------------------------

# Constants from src/neml2/tensors/crystallography.cxx (header crystallography.h).
_o = 1.0
_z = 0.0
_h = 0.5
_a = 0.7071067811865476  # sqrt(2)/2
# _b = sqrt(3)/2 — used by hexagonal, not cubic; not needed here.

# Quaternion convention: (w, x, y, z) per src/neml2/tensors/Quaternion.cxx.
_CUBIC_QUATERNIONS: list[tuple[float, float, float, float]] = [
    (_o, _z, _z, _z),
    (_h, _h, _h, _h),
    (-_h, _h, _h, _h),
    (_h, -_h, _h, _h),
    (_h, _h, -_h, _h),
    (-_h, -_h, -_h, _h),
    (_h, -_h, -_h, _h),
    (-_h, -_h, _h, _h),
    (-_h, _h, -_h, _h),
    (_z, _z, _o, _z),
    (_z, _z, _z, _o),
    (_z, _o, _z, _z),
    (_z, -_a, _z, _a),
    (_z, _a, _z, _a),
    (_a, _z, _a, _z),
    (_a, _z, -_a, _z),
    (_z, _z, -_a, _a),
    (_a, _a, _z, _z),
    (_a, -_a, _z, _z),
    (_z, _z, _a, _a),
    (_z, -_a, _a, _z),
    (_a, _z, _z, -_a),
    (_z, _a, _a, _z),
    (_a, _z, _z, _a),
]


def _quaternion_to_R(q: torch.Tensor) -> torch.Tensor:
    """Convert ``(*, 4)`` quaternion ``(w, x, y, z)`` to ``(*, 3, 3)`` rotation matrix.

    Matches ``Quaternion::rotation_matrix`` in ``src/neml2/tensors/Quaternion.cxx``.
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    xx = x * x
    yy = y * y
    zz = z * z
    rows = [
        torch.stack([1 - 2 * yy - 2 * zz, 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * xx - 2 * zz, 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * xx - 2 * yy], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def cubic_symmetry_operators(
    *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
) -> R2:
    """Return the 24 cubic (432) proper-rotation symmetry operators as ``R2``.

    Shape: ``(24, 3, 3)`` with ``sub_batch_ndim=1`` so the leading axis acts
    as a per-operator sub-batch. Mirrors
    ``neml2::crystallography::symmetry("432")``.
    """
    qs = torch.tensor(_CUBIC_QUATERNIONS, dtype=dtype, device=device)  # (24, 4)
    R = _quaternion_to_R(qs)  # (24, 3, 3)
    return R2(R, sub_batch_ndim=1)


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


@register_neml2_object("CubicCrystal")
class CubicCrystal:
    """A specialization of the general CrystalGeometry class defining a cubic crystal system."""

    SECTION = "Data"

    # Construction-only options. ``from_hit`` owns the parsing (the lattice
    # parameter may be a literal float or a [Tensors] cross-reference, and the
    # symmetry operators are fixed to the cubic "432" group rather than read
    # from HIT). This schema drives the syntax catalog.
    hit = HitSchema(
        parameter("lattice_parameter", Scalar, "The lattice parameter"),
        parameter(
            "slip_directions",
            MillerIndex,
            "A list of Miller indices defining the slip directions",
        ),
        parameter(
            "slip_planes",
            MillerIndex,
            "A list of Miller indices defining the slip planes",
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> CrystalGeometry:
        lp_raw = node.param_str("lattice_parameter")
        try:
            lp = float(lp_raw)
        except ValueError:
            # Cross-reference into [Tensors]: same convention as
            # declare_typed_parameter mode 2. Reduce to a Python float — the
            # lattice parameter is scalar by construction.
            t = factory.get_tensor(lp_raw)
            data = t.data if hasattr(t, "data") else t
            lp = float(data.reshape(-1)[0].item())

        dtype = torch.float64
        lattice = lp * torch.eye(3, dtype=dtype)
        sym_ops = cubic_symmetry_operators(dtype=dtype)

        sd_name = node.param_str("slip_directions")
        sp_name = node.param_str("slip_planes")
        sd = factory.get_tensor(sd_name)
        sp = factory.get_tensor(sp_name)
        if not isinstance(sd, MillerIndex):
            raise TypeError(
                f"[Data/{node.path()}] slip_directions={sd_name!r} did not evaluate to "
                f"MillerIndex (got {type(sd).__name__})."
            )
        if not isinstance(sp, MillerIndex):
            raise TypeError(
                f"[Data/{node.path()}] slip_planes={sp_name!r} did not evaluate to "
                f"MillerIndex (got {type(sp).__name__})."
            )

        return CrystalGeometry(
            sym_ops=sym_ops,
            lattice_vectors=lattice,
            slip_directions=sd,
            slip_planes=sp,
        )
