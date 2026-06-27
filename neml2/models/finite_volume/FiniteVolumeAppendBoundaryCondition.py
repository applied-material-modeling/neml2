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

"""Python-native mirror of C++ ``finite_volume/FiniteVolumeAppendBoundaryCondition.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar
from ...types.functions import cat
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("FiniteVolumeAppendBoundaryCondition")
class FiniteVolumeAppendBoundaryCondition(Model):
    """Concatenate a single BC value onto the trailing sub-batch axis.

    With ``input`` shape ``(*B, N)`` the output has shape ``(*B, N+1)``.
    Left-side: $output = [bc, *input]$; right-side: ``[*input, bc]``.

    The bc value is a static buffer (per the KWN HIT, ``bc_value = 0.0``);
    chain rule is wired only for ``input``, where the derivative is a
    bidiagonal selector ((N+1, N) identity block with one zero row).
    The output variable name defaults to ``<input>_with_bc_<side>``.
    """

    hit = HitSchema(
        option(
            "side",
            str,
            "Which side to append the boundary condition value to. Options are: right, left",
            default="left",
            attr="_side",
        ),
        input(
            "input",
            Scalar,
            "Input tensor to append the boundary condition to.",
            attr="_input_name",
        ),
        # default=None lets __post_init__ derive the output name from input + side.
        output(
            "output",
            Scalar,
            "Output tensor name. Defaults to input + '_with_bc_left' or '_with_bc_right'.",
            default=None,
            attr="_output_name",
        ),
        parameter("bc_value", Scalar, "Boundary condition value to append."),
    )

    bc_value: Scalar
    _side: str
    _input_name: str
    _output_name: str  # may arrive as None from HIT; __post_init__ derives a string then

    def __post_init__(self) -> None:
        if self._side not in ("left", "right"):
            raise ValueError(
                f"FiniteVolumeAppendBoundaryCondition side={self._side!r} must be 'left' or 'right'"
            )
        if self._output_name is None:
            suffix = "_with_bc_left" if self._side == "left" else "_with_bc_right"
            self._output_name = f"{self._input_name}{suffix}"
        # Schema-side rename couldn't add the derived name to output_spec
        # because we only just computed it; finish that here.
        self.input_spec = {self._input_name: Scalar}
        self.output_spec = {self._output_name: Scalar}

    def _bc_tail(self, template: Scalar, promoted_params) -> Scalar:
        """Typed (*template_dyn, 1) Scalar carrying the parameter ``bc_value``.

        The tail mirrors ``template``'s dynamic batch (broadcast) plus a
        single sub-batch slot at the boundary. ``sub_batch_zeros_like``
        gives the zero baseline with the correct K layout; adding the
        ``bc_value`` parameter (read through ``_get_param``, not ``self.<attr>``,
        to stay promotion-compatible) lifts it to the boundary value.
        """
        zero = Scalar.zeros_like(template, sub_batch_shape=(1,))
        return zero + self._get_param("bc_value", promoted_params, Scalar)

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        (u_wrap,) = inputs  # bc_value is static (not promotable) -> no promoted_params
        # Value path: bc_tail carries the parameter value, then cat
        # along the cell axis (sub-batch dim 0 for a Scalar with
        # sub_batch_ndim=1).
        bc_tail = self._bc_tail(u_wrap, ())
        if self._side == "left":
            out = cat([bc_tail.sub_batch, u_wrap.sub_batch], dim=0)
        else:
            out = cat([u_wrap.sub_batch, bc_tail.sub_batch], dim=0)
        if v is None:
            return out

        side = self._side

        def input_action(V_in: Scalar) -> Scalar:
            # The chain rule contribution appends a zero tail (the BC
            # value is a parameter, not an unknown -- it doesn't
            # contribute to ``d output / d input``).
            zero_tail = Scalar.zeros_like(V_in, sub_batch_shape=(1,))
            if side == "left":
                return cat([zero_tail.sub_batch, V_in.sub_batch], dim=0)
            return cat([V_in.sub_batch, zero_tail.sub_batch], dim=0)

        return out, self.apply_chain_rule(
            v, self._output_name, {self._input_name: input_action}, output=out
        )


__all__ = ["FiniteVolumeAppendBoundaryCondition"]
