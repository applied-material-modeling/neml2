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

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, output, parameter
from ...types import Scalar


@register_neml2_object("LinearlyInterpolateToCellEdges")
class LinearlyInterpolateToCellEdges(Model):
    """Linear interpolation from cell centers to interior cell edges.

    ``cell_values`` has size $N$; ``cell_centers`` has size $N$;
    ``cell_edges`` has size ``N+1`` (interior + boundary edges); output has
    size ``M = N-1`` (interior edges only).

    The KWN HIT wires ``cell_values`` from another model output (nl mode)
    and keeps ``cell_centers`` / ``cell_edges`` as static ``[Tensors]``
    references. Chain rule here is wired for ``cell_values`` only (other
    pairs would require the nl-mode parameter chain-rule, follow-on work).
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

    def __post_init__(self) -> None:
        # __post_init__ runs after _store_schema_values' deferred-parameter
        # pass, so _nl_params is populated. If cell_values resolved as an nl
        # input (mode 3/4), the output depends densely on that promoted input.
        cv_nl = self._nl_params.get("cell_values")
        if cv_nl is not None:
            self.list_deriv = {(self._out_name, cv_nl.input_name): "dense"}

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute ``w_left`` and ``w_right`` from the static positions."""
        x = self.cell_centers.data  # (*, N)
        xe = self.cell_edges.data  # (*, N+1)
        x_left = x[..., :-1]
        x_right = x[..., 1:]
        xe_vec = xe[..., 1:-1]  # interior edges, size N-1
        inv = 1.0 / (x_right - x_left)
        w_left = (x_right - xe_vec) * inv
        w_right = (xe_vec - x_left) * inv
        return w_left, w_right

    def forward(self, *nl_params, v: ChainRuleDict | None = None):  # type: ignore[override]
        cv_wrap = self._get_param("cell_values", nl_params, Scalar)
        q = cv_wrap.data  # (*B, N)
        w_left, w_right = self._weights()  # (*, M) each
        q_left = q[..., :-1]
        q_right = q[..., 1:]
        edge = w_left * q_left + w_right * q_right  # (*B, M)
        out = Scalar(edge, sub_batch_ndim=cv_wrap.sub_batch_ndim)
        if v is None:
            return out

        cv_nl = self._nl_params.get("cell_values")
        if cv_nl is None:
            # cell_values is a static buffer — no chain rule to propagate.
            return out, self.apply_chain_rule(v, self._out_name, {}, output=out)

        def cv_action(V: Scalar) -> Scalar:
            d = V.data  # (K, *dyn, N)
            return Scalar(
                w_left * d[..., :-1] + w_right * d[..., 1:], sub_batch_ndim=V.sub_batch_ndim
            )

        return out, self.apply_chain_rule(
            v, self._out_name, {cv_nl.input_name: cv_action}, output=out
        )


__all__ = ["LinearlyInterpolateToCellEdges"]
