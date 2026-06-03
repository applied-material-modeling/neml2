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

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar


@register_native("FiniteVolumeAppendBoundaryCondition")
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
        # Schema-side rename couldn't add the derived name to output_spec /
        # list_deriv because we only just computed it; finish that here.
        self.input_spec = {self._input_name: Scalar}
        self.output_spec = {self._output_name: Scalar}
        self.list_deriv = {(self._output_name, self._input_name): "dense"}

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        (u_wrap,) = inputs
        u = u_wrap.data  # (*B, N)
        bc = self.bc_value.data  # scalar, broadcasts
        # Expand bc to match input's shape minus the trailing sub-batch axis,
        # with a length-1 sub-batch axis for the concat.
        bc_exp = bc.expand(*u.shape[:-1], 1) if bc.ndim < u.ndim else bc
        if self._side == "left":
            y = torch.cat([bc_exp, u], dim=-1)
        else:
            y = torch.cat([u, bc_exp], dim=-1)
        out = Scalar(y, sub_batch_ndim=u_wrap.sub_batch_ndim)
        if v is None:
            return out

        # Leading-K: the cell axis is the trailing sub-batch axis of the
        # Scalar tangent ``(K, *dyn, L_in)``. Pushforward = same concat with a
        # zero row at the BC side, taking ``L_in → L_in + 1``.
        side = self._side

        def input_action(V: Scalar) -> Scalar:
            d = V.data
            zero = d.new_zeros(*d.shape[:-1], 1)
            new_d = torch.cat([zero, d], dim=-1) if side == "left" else torch.cat([d, zero], dim=-1)
            return Scalar(new_d, sub_batch_ndim=V.sub_batch_ndim)

        return out, self.apply_chain_rule(
            v, self._output_name, {self._input_name: input_action}, output=out
        )


__all__ = ["FiniteVolumeAppendBoundaryCondition"]
