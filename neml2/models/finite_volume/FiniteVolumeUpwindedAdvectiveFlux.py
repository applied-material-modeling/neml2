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

"""Python-native mirror of C++ ``finite_volume/FiniteVolumeUpwindedAdvectiveFlux.h``."""

from __future__ import annotations

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import Scalar


@register_neml2_object("FiniteVolumeUpwindedAdvectiveFlux")
class FiniteVolumeUpwindedAdvectiveFlux(Model):
    """Compute upwinded advective fluxes at cell edges."""

    hit = HitSchema(
        input("u", Scalar, "Cell-averaged field values.", attr="_u_name"),
        input("v_edge", Scalar, "Cell-edge advection velocity values.", attr="_v_edge_name"),
        output("flux", Scalar, "Cell-edge advective fluxes.", attr="_flux_name"),
    )

    # v_edge is diagonal (default — absent from list_deriv); u is dense.
    list_deriv = {("flux", "u"): "dense"}

    _u_name: str
    _v_edge_name: str
    _flux_name: str

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        u_wrap, v_wrap = inputs
        u = u_wrap.data  # (*B, N)
        ve = v_wrap.data  # (*B, M)
        v_abs = ve.abs()
        v_plus = 0.5 * (ve + v_abs)
        v_minus = 0.5 * (ve - v_abs)
        u_left = u[..., :-1]
        u_right = u[..., 1:]
        flux = v_plus * u_left + v_minus * u_right
        sb = max(u_wrap.sub_batch_ndim, v_wrap.sub_batch_ndim)
        out = Scalar(flux, sub_batch_ndim=sb)
        if v is None:
            return out

        s = torch.sign(ve)
        # d(flux[i])/d(v_edge[i]) = 0.5(1+s)·u_left + 0.5(1-s)·u_right = dJ_dv
        dv_coeff = 0.5 * (1.0 + s) * u_left + 0.5 * (1.0 - s) * u_right  # (*, M)

        def u_action(V: Scalar) -> Scalar:
            d = V.data  # (K, *dyn, N)
            flux = v_plus * d[..., :-1] + v_minus * d[..., 1:]
            return Scalar(flux, sub_batch_ndim=V.sub_batch_ndim)

        def v_action(V: Scalar) -> Scalar:
            # diagonal in the edge axis — element-wise weight passes through.
            return Scalar(dv_coeff * V.data, sub_batch_ndim=V.sub_batch_ndim)

        actions = {self._u_name: u_action, self._v_edge_name: v_action}
        return out, self.apply_chain_rule(v, self._flux_name, actions, output=out)


__all__ = ["FiniteVolumeUpwindedAdvectiveFlux"]
