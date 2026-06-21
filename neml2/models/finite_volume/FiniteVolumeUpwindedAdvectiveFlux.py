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

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar
from ...types.functions import fullify, heaviside
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("FiniteVolumeUpwindedAdvectiveFlux")
class FiniteVolumeUpwindedAdvectiveFlux(Model):
    """Compute upwinded advective fluxes at cell edges.

    $$
    J_e = v_e \\, \\bigl(H(v_e) u_{e^-} + (1 - H(v_e)) u_{e^+}\\bigr)
    $$

    where $v_e$ is the cell-edge advection velocity, $H$ is the Heaviside
    step ($1$ when $v_e > 0$, $0$ when $v_e < 0$), and $u_{e^\\pm}$ are the
    cell-averaged values on the left ($e^-$) and right ($e^+$) of edge $e$.
    The Heaviside factor selects the upwind cell -- the cell from which
    information propagates with the velocity sign.

    The chain rule uses the same upwind coefficient::

        dv_coeff = H u_left + (1 - H) u_right
                 = u_right + H (u_left - u_right)

    so ``J_e = v_e * dv_coeff`` and ``d J_e / d v_e = dv_coeff``.
    """

    hit = HitSchema(
        input("u", Scalar, "Cell-averaged field values.", attr="_u_name"),
        input("v_edge", Scalar, "Cell-edge advection velocity values.", attr="_v_edge_name"),
        output("flux", Scalar, "Cell-edge advective fluxes.", attr="_flux_name"),
    )

    _u_name: str
    _v_edge_name: str
    _flux_name: str

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        u_wrap, v_wrap = inputs
        # Typed upwind coefficient: ``u_left`` and ``u_right`` are
        # neighbouring cell values addressed by ``sub_batch`` slicing
        # (the cell axis is the first sub-batch axis on Scalar with
        # sub_batch_ndim=1). Going through typed ops preserves K
        # metadata when this forward is called as part of the chain
        # rule -- direct ``.data`` slicing would drop k_ndim and the
        # leading K axis would silently become a phantom batch axis on
        # the contribution, accumulating through the composed model.
        u_left = u_wrap.sub_batch[:-1]
        u_right = u_wrap.sub_batch[1:]
        H = heaviside(v_wrap)
        dv_coeff = u_right + H * (u_left - u_right)
        out = v_wrap * dv_coeff
        if v is None:
            return out

        def u_action(V_in: Scalar) -> Scalar:
            # The stencil mixes adjacent cells (``V_left`` and ``V_right``
            # are sliced views of the same sub-batch axis at offsets 0
            # and 1), which breaks the K-paired broadcast assumption
            # (``V[k, i] = delta(i==k)``). The combined contribution is
            # NON-diagonal in (K, sub) -- each output cell depends on
            # two K directions. Fullify expands the broadcast K to its
            # enumerated form so the stencil result is materialised
            # explicitly per perturbation direction. Without it the
            # off-diagonal sub-diagonal entries of the assembled
            # Jacobian are silently dropped.
            V_full = fullify(V_in)
            V_left = V_full.sub_batch[:-1]
            V_right = V_full.sub_batch[1:]
            return v_wrap * (V_right + H * (V_left - V_right))

        def v_action(V_in: Scalar) -> Scalar:
            return dv_coeff * V_in

        actions = {self._u_name: u_action, self._v_edge_name: v_action}
        return out, self.apply_chain_rule(v, self._flux_name, actions, output=out)


__all__ = ["FiniteVolumeUpwindedAdvectiveFlux"]
