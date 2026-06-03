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

"""Pytree registration for the typed tensor wrappers.

`torch.utils._pytree.register_dataclass` handles both the export-time flatten
(consumed by `torch.export.export`) and the broader pytree walk surface
(consumed by `nn.Module` state_dict traversal, optimizer code, etc.). The
public `torch.export.register_dataclass` is a thin alias for this — calling
either is equivalent, but we go through the private path because it accepts
``field_names`` which the public alias does not.

The wrappers carry a second non-leaf field, ``sub_batch_ndim``, which tracks
the number of trailing batch axes that act as a structured "sub-batch"
region (the Python-native analogue of C++ `intmd_dim`). It is excluded from
the pytree via ``drop_field_names`` so the only flat leaf is ``data``;
roundtrips through `torch.export`'s flatten/unflatten reset it to the
dataclass default of ``0`` — which is fine because the exported graph
operates on raw tensors whose shapes already encode the sub-batch axes.

All `torch._inductor` / `torch.utils._pytree` internal-API drift is contained
to this one file per RISK.md R-001.
"""

from __future__ import annotations

from torch.utils import _pytree as _torch_pytree


def register(cls: type, *extra_drop_fields: str) -> None:
    """Register a typed wrapper class as a pytree node.

    The class is expected to be a `@dataclass(frozen=True, eq=False)` with a
    ``data: torch.Tensor`` field and a ``sub_batch_ndim: int = 0`` field.
    Flatten yields the one tensor; unflatten reconstructs the wrapper around
    it with ``sub_batch_ndim`` reset to the default.

    ``extra_drop_fields`` lets wrappers with additional non-leaf metadata
    fields drop those too.
    """
    _torch_pytree.register_dataclass(
        cls,
        field_names=["data"],
        drop_field_names=["sub_batch_ndim", *extra_drop_fields],
        serialized_type_name=f"neml2.types.{cls.__name__}",
    )
