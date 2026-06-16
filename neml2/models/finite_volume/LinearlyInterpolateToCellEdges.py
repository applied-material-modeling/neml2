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

"""Python-native mirror of C++ ``finite_volume/LinearlyInterpolateToCellEdges.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameter
from ...types import Scalar
from ...types.functions import fullify
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("LinearlyInterpolateToCellEdges")
class LinearlyInterpolateToCellEdges(Model):
    """Linear interpolation from cell centers to interior cell edges.

    ``cell_values`` has size $N$; ``cell_centers`` has size $N$;
    ``cell_edges`` has size ``N+1`` (interior + boundary edges); output has
    size ``M = N-1`` (interior edges only).
    """

    hit = HitSchema(
        parameter(
            "cell_values", Scalar, "Cell-centered values to interpolate.", allow_nonlinear=True
        ),
        parameter("cell_centers", Scalar, "Cell center positions."),
        parameter("cell_edges", Scalar, "Cell edge positions."),
        output(
            "edge_values",
            Scalar,
            "Linearly interpolated cell-edge values.",
            attr="_out_name",
        ),
    )

    cell_values: Scalar
    cell_centers: Scalar
    cell_edges: Scalar
    _out_name: str

    def _weights(self) -> tuple[Scalar, Scalar]:
        """Typed stencil weights ``w_left``, ``w_right`` over the M=N-1 interior edges.

        ``register_typed_parameter`` stores the raw tensor data, so
        ``self.cell_centers`` / ``self.cell_edges`` arrive with
        ``sub_batch_ndim=0``. Retag to expose the cell axis as
        ``sub_batch`` so slicing along it is typed (and so the typed
        arithmetic with ``cell_values`` below aligns on sub_batch).
        """
        centers = self.cell_centers.sub_batch.retag(1)
        edges = self.cell_edges.sub_batch.retag(1)
        x_left = centers.sub_batch[:-1]
        x_right = centers.sub_batch[1:]
        xe_vec = edges.sub_batch[1:-1]  # interior edges, size N-1
        inv = 1.0 / (x_right - x_left)
        w_left = (x_right - xe_vec) * inv
        w_right = (xe_vec - x_left) * inv
        return w_left, w_right

    def forward(self, *nl_params, v: ChainRuleDict | None = None):  # type: ignore[override]
        cv_wrap = self._get_param("cell_values", nl_params, Scalar)
        w_left, w_right = self._weights()
        q_left = cv_wrap.sub_batch[:-1]
        q_right = cv_wrap.sub_batch[1:]
        out = w_left * q_left + w_right * q_right
        if v is None:
            return out

        cv_nl = self._nl_params.get("cell_values")
        if cv_nl is None:
            # cell_values is a static buffer -- no chain rule to propagate.
            return out, self.apply_chain_rule(v, self._out_name, {}, output=out)

        def cv_action(V_in: Scalar) -> Scalar:
            # Cross-cell stencil: fullify before slicing for the same
            # reason as ``FiniteVolumeUpwindedAdvectiveFlux.u_action``
            # (the K-paired broadcast eye assumption is broken when
            # adjacent cell tangents are linearly combined).
            V_full = fullify(V_in)
            V_left = V_full.sub_batch[:-1]
            V_right = V_full.sub_batch[1:]
            return w_left * V_left + w_right * V_right

        return out, self.apply_chain_rule(
            v, self._out_name, {cv_nl.input_name: cv_action}, output=out
        )


__all__ = ["LinearlyInterpolateToCellEdges"]
