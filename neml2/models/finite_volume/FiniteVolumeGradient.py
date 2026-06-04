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

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar


@register_neml2_object("FiniteVolumeGradient")
class FiniteVolumeGradient(Model):
    """Compute prefactor-weighted gradients at cell edges using first-order reconstruction."""

    hit = HitSchema(
        input("u", Scalar, "Cell-averaged field values.", attr="_u_name"),
        output(
            "grad_u",
            Scalar,
            "Cell-edge prefactor-weighted gradients.",
            attr="_out_name",
        ),
        parameter("dx", Scalar, "Cell center spacing between adjacent cells."),
        # Default prefactor literal "1.0" makes declare_typed_parameter wrap it
        # as Scalar(1.0); the scalar broadcasts across the M-sized output.
        # ``allow_nonlinear=True`` lets HIT promote ``prefactor`` to a runtime
        # input when wired to another model output (e.g. an edge diffusivity
        # produced by ``LinearlyInterpolateToCellEdges``); the chain rule below
        # picks up the corresponding pushforward only when the promotion fires.
        parameter(
            "prefactor",
            Scalar,
            "Cell-edge prefactor values (defaults to 1).",
            default="1.0",
            allow_nonlinear=True,
        ),
    )

    list_deriv = {("grad_u", "u"): "dense"}

    dx: Scalar
    prefactor: Scalar
    _u_name: str
    _out_name: str

    def __post_init__(self) -> None:
        # If ``prefactor`` was promoted to a runtime input (mode 3/4), the
        # output also depends densely on that input — mirror the C++
        # ``setup_input_derivative_storage`` pattern by extending list_deriv.
        pf_nl = self._nl_params.get("prefactor")
        if pf_nl is not None:
            self.list_deriv = {
                **self.list_deriv,
                (self._out_name, pf_nl.input_name): "dense",
            }

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        # Structural inputs come first, then any promoted parameters in the
        # order ``_nl_params`` was registered (``prefactor`` is the only one
        # here when it gets promoted).
        u_wrap = inputs[0]
        nl_params = inputs[1:]
        u = u_wrap.data  # (*B, N)
        dx = self.dx.data
        pf_wrap = self._get_param("prefactor", nl_params, Scalar)
        prefactor = pf_wrap.data
        inv_dx = 1.0 / dx
        du = u[..., 1:] - u[..., :-1]  # (*B, M)
        grad = -prefactor * du * inv_dx  # (*B, M)
        out = Scalar(grad, sub_batch_ndim=u_wrap.sub_batch_ndim)
        if v is None:
            return out

        # Bidiagonal pushforward in u: d(grad[i]) = (prefactor/dx)·(dV[i] − dV[i+1]).
        # The cell axis is the Scalar tangent's trailing axis (K, *dyn, N).
        coeff = prefactor * inv_dx  # (*, M)

        def u_action(V: Scalar) -> Scalar:
            d = V.data
            return Scalar(coeff * (d[..., :-1] - d[..., 1:]), sub_batch_ndim=V.sub_batch_ndim)

        actions: dict[str, ChainRuleAction] = {self._u_name: u_action}

        # When prefactor was promoted (mode 3/4), its tangent multiplies the
        # raw gradient ``-du/dx`` (the partial d(grad)/d(prefactor)).
        pf_nl = self._nl_params.get("prefactor")
        if pf_nl is not None:
            scale = -du * inv_dx  # (*, M)

            def pf_action(V: Scalar, c=scale) -> Scalar:
                return Scalar(c * V.data, sub_batch_ndim=V.sub_batch_ndim)

            actions[pf_nl.input_name] = pf_action

        return out, self.apply_chain_rule(v, self._out_name, actions, output=out)


__all__ = ["FiniteVolumeGradient"]
