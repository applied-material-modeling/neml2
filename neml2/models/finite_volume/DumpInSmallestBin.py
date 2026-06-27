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

"""Python-native mirror of C++ ``finite_volume/DumpInSmallestBin.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar
from ...types.functions import cat
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("DumpInSmallestBin")
class DumpInSmallestBin(Model):
    """Dump the source magnitude into the smallest cell-center bin.

    ``magnitude`` is a global Scalar (no sub-batch); ``cell_centers`` carries
    the per-cell sub-batch axis (``sub_batch_ndim=1``, size $N$); the output
    ``dumped_source`` has the same per-cell sub-batch with first-cell value =
    ``magnitude`` and zero elsewhere. ``magnitude`` therefore couples densely
    with the per-cell output -- declared in :attr:`list_deriv`.
    """

    hit = HitSchema(
        input("magnitude", Scalar, "Source magnitude."),
        output(
            "dumped_source",
            Scalar,
            "Source dumped into the smallest bin.",
            default="state/dumped_source",
        ),
        parameter("cell_centers", Scalar, "Cell center locations.", allow_promotion=True),
    )

    # magnitude is a global Scalar (sub_batch=0); the output carries the
    # per-cell sub-batch axis from cell_centers -- so the edge INTRODUCES
    # the ``"cell"`` label on the output.

    cell_centers: Scalar

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        mag = inputs[0]
        # cell_centers carries an N-cell axis. The HIT-stored parameter loses
        # its declared sub_batch_ndim metadata (the tensor is stored as a bare
        # nn.Parameter and re-wrapped with sub_batch_ndim=0 on access), so
        # retag here to view that trailing axis as the per-cell sub-batch.
        centers = self._get_param(
            "cell_centers", promoted_params=(), type_cls=Scalar
        ).sub_batch.retag(1)
        N = int(centers.sub_batch_shape[-1])

        # Promote mag (sub_batch=0) to a per-cell Scalar of size 1 along a new
        # cell axis -- the C++ ``mag_raw.unsqueeze(...)`` branch when
        # ``_magnitude.intmd_dim() == 0``.
        mag_cell = mag.sub_batch.unsqueeze(0)  # Scalar(sub_batch=1, size=1)
        # Zero tail occupying cells [1, N): same dtype/device as centers (now
        # retagged with sub_batch_ndim=1), new sub-batch axis of size N-1.
        zero_tail = Scalar.zeros_like(centers, sub_batch_shape=(N - 1,))
        src = cat([mag_cell.sub_batch, zero_tail.sub_batch], dim=0)  # (sub_batch=1, size=N)

        if v is None:
            return src

        # D-062 pushforward. Forward is linear in ``mag`` (mag occupies cell 0,
        # zeros elsewhere), so the action mirrors the forward structure on the
        # tangent ``V``: lift ``V`` to a per-cell Scalar of size 1, concat with
        # a zero tail. ``V`` has leading-K dynamic axis; both ``sub_batch.unsqueeze``
        # and the free ``cat`` over region views broadcast over the
        # dynamic batch (including leading K) cleanly.
        def mag_action(V: Scalar) -> Scalar:
            V_cell = V.sub_batch.unsqueeze(0)
            V_tail = Scalar.zeros_like(V_cell, sub_batch_shape=(N - 1,))
            return cat([V_cell.sub_batch, V_tail.sub_batch], dim=0)

        return src, self.apply_chain_rule(v, "dumped_source", {"magnitude": mag_action}, output=src)


__all__ = ["DumpInSmallestBin"]
