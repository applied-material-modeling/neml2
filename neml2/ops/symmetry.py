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

"""Crystal point-group symmetry operators in orbifold notation.

Python-native port of ``neml2::crystallography::symmetry`` from v2
(``src/neml2/tensors/crystallography.cxx``). :func:`symmetry` returns the
rotational symmetry operators of a crystal class as a batch of rotation
matrices (an :class:`~neml2.types.R2` with one sub-batch axis indexing the
operators).

The operators are built from the base tetragonal / hexagonal / cubic unit
quaternion tables and converted to rotation matrices via
:func:`~neml2.types.quaternion_rotation_matrix` -- everything stays in the
typed :mod:`neml2.types` wrappers.
"""

from __future__ import annotations

import math

import torch

from neml2.types import R2, Quaternion, cat, quaternion_rotation_matrix

# Quaternion component constants used by the operator tables (v2 ``o/z/a/h/b``).
_O = 1.0
_Z = 0.0
_H = 0.5
_A = 1.0 / math.sqrt(2.0)  # 1/sqrt(2)
_B = math.sqrt(3.0) / 2.0  # sqrt(3)/2

#: Orbifold-notation crystal classes supported by :func:`symmetry`.
ORBIFOLDS = ("432", "23", "622", "32", "6", "3", "42", "4", "222", "2", "1")


def _quat_table(
    rows: list[list[float]],
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> Quaternion:
    """A unit-quaternion operator table as a ``Quaternion`` with sub_batch_ndim=1."""
    return Quaternion(
        torch.tensor(rows, dtype=dtype or torch.float64, device=device), sub_batch_ndim=1
    )


def _tetragonal(dtype, device) -> Quaternion:
    o, z, a = _O, _Z, _A
    return _quat_table(
        [
            [o, z, z, z],
            [z, z, o, z],
            [z, o, z, z],
            [z, z, z, o],
            [a, z, z, -a],
            [a, z, z, a],
            [z, a, a, z],
            [z, -a, a, z],
        ],
        dtype,
        device,
    )


def _hexagonal(dtype, device) -> Quaternion:
    o, z, h, b = _O, _Z, _H, _B
    return _quat_table(
        [
            [o, z, z, z],
            [-h, z, z, b],
            [h, z, z, b],
            [b, z, z, -h],
            [z, z, z, o],
            [b, z, z, h],
            [z, -h, b, z],
            [z, o, z, z],
            [z, h, b, z],
            [z, b, h, z],
            [z, z, o, z],
            [z, b, -h, z],
        ],
        dtype,
        device,
    )


def _cubic(dtype, device) -> Quaternion:
    o, z, h, a = _O, _Z, _H, _A
    return _quat_table(
        [
            [o, z, z, z],
            [h, h, h, h],
            [-h, h, h, h],
            [h, -h, h, h],
            [h, h, -h, h],
            [-h, -h, -h, h],
            [h, -h, -h, h],
            [-h, -h, h, h],
            [-h, h, -h, h],
            [z, z, o, z],
            [z, z, z, o],
            [z, o, z, z],
            [z, -a, z, a],
            [z, a, z, a],
            [a, z, a, z],
            [a, z, -a, z],
            [z, z, -a, a],
            [a, a, z, z],
            [a, -a, z, z],
            [z, z, a, a],
            [z, -a, a, z],
            [a, z, z, -a],
            [z, a, a, z],
            [a, z, z, a],
        ],
        dtype,
        device,
    )


def _operators(
    orbifold: str, dtype: torch.dtype | None, device: torch.device | str | None
) -> Quaternion:
    """The unit-quaternion operator set for ``orbifold`` (v2 ``symmetry`` subsets)."""
    if orbifold == "432":
        return _cubic(dtype, device)
    if orbifold == "23":
        return _cubic(dtype, device).sub_batch[0:12]
    if orbifold == "622":
        return _hexagonal(dtype, device)
    if orbifold == "32":
        hexa = _hexagonal(dtype, device)
        return cat([hexa.sub_batch[0:3].sub_batch, hexa.sub_batch[9:12].sub_batch], dim=0)
    if orbifold == "6":
        return _hexagonal(dtype, device).sub_batch[0:6]
    if orbifold == "3":
        return _hexagonal(dtype, device).sub_batch[0:3]
    if orbifold == "42":
        return _tetragonal(dtype, device)
    if orbifold == "4":
        tetra = _tetragonal(dtype, device)
        return cat([tetra.sub_batch[0:1].sub_batch, tetra.sub_batch[3:6].sub_batch], dim=0)
    if orbifold == "222":
        return _tetragonal(dtype, device).sub_batch[0:4]
    if orbifold == "2":
        return _tetragonal(dtype, device).sub_batch[0:2]
    if orbifold == "1":
        return _tetragonal(dtype, device).sub_batch[0:1]
    raise ValueError(f"Unknown crystal class {orbifold!r}; expected one of {ORBIFOLDS}")


def symmetry(
    orbifold: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> R2:
    """Rotational symmetry operators of a crystal class, as rotation matrices.

    ``orbifold`` is the orbifold-notation point group (one of :data:`ORBIFOLDS`):
    ``432``/``23`` (cubic), ``622``/``32``/``6``/``3`` (hexagonal/trigonal),
    ``42``/``4``/``222``/``2`` (tetragonal/orthorhombic/monoclinic), ``1``
    (triclinic). Returns an :class:`~neml2.types.R2` whose single sub-batch axis
    indexes the operators (same set and order as v2
    ``neml2::crystallography::symmetry``).
    """
    return quaternion_rotation_matrix(_operators(orbifold, dtype, device))


__all__ = ["symmetry", "ORBIFOLDS"]
