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

"""Python-native mirror of C++ ``finite_volume/FiniteVolumeGradient.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar
from ...types.functions import fullify
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("FiniteVolumeGradient")
class FiniteVolumeGradient(Model):
    """Compute prefactor-weighted gradients at cell edges using first-order reconstruction.

    $$
    \\nabla u\\bigl|_e = -\\frac{\\text{prefactor}}{\\Delta x_e}\\,(u_{e^+} - u_{e^-})
    $$

    The minus sign matches the C++ implementation and the surrounding
    finite-volume conventions.
    """

    hit = HitSchema(
        input("u", Scalar, "Cell-averaged field values.", attr="_u_name"),
        output(
            "grad_u",
            Scalar,
            "Cell-edge prefactor-weighted gradients.",
            attr="_out_name",
        ),
        parameter("dx", Scalar, "Cell center spacing between adjacent cells."),
        parameter(
            "prefactor",
            Scalar,
            "Cell-edge prefactor values (defaults to 1).",
            default="1.0",
            allow_promotion=True,
        ),
    )

    dx: Scalar
    prefactor: Scalar
    _u_name: str
    _out_name: str

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        u_wrap = inputs[0]
        promoted_params = inputs[1:]
        pf_wrap = self._get_param("prefactor", promoted_params, Scalar)
        # All typed: ``inv_dx`` via Scalar.__rtruediv__, neighbour
        # differences via sub_batch slicing on the cell axis. The chain
        # rule path reuses ``coeff`` so the value and pushforward share
        # the same materialised stencil weights.
        inv_dx = 1.0 / self._get_param("dx", promoted_params, Scalar)
        coeff = pf_wrap * inv_dx
        u_left = u_wrap.sub_batch[:-1]
        u_right = u_wrap.sub_batch[1:]
        du = u_right - u_left
        out = -(coeff * du)
        if v is None:
            return out

        def u_action(V_in: Scalar) -> Scalar:
            # Fullify before the stencil: ``V_left - V_right`` mixes
            # adjacent cells, breaking the K-paired broadcast eye
            # assumption (each output cell depends on two K directions).
            # See ``FiniteVolumeUpwindedAdvectiveFlux.u_action`` for the
            # full explanation; the same applies here.
            V_full = fullify(V_in)
            V_left = V_full.sub_batch[:-1]
            V_right = V_full.sub_batch[1:]
            return coeff * (V_left - V_right)

        actions: dict[str, ChainRuleAction] = {self._u_name: u_action}

        pf_nl = self._promoted_params.get("prefactor")
        if pf_nl is not None:
            # d(grad)/d(prefactor) = -du / dx (the partial w.r.t. the
            # prefactor scaling factor); the action multiplies the
            # prefactor's tangent by that scalar weight.
            scale = -du * inv_dx

            def pf_action(V_in: Scalar, c: Scalar = scale) -> Scalar:
                return c * V_in

            actions[pf_nl.input_name] = pf_action

        return out, self.apply_chain_rule(v, self._out_name, actions, output=out)


__all__ = ["FiniteVolumeGradient"]
