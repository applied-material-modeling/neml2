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

"""Python-native mirror of C++ ``common/MixedControlSetup.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import (
    SR2,
    gt,
    where,
)


@register_neml2_object("MixedControlSetup")
class MixedControlSetup(Model):
    """Per-component selection between strain- and stress-controlled inputs.

    For each Mandel component $i$: ``y_i = x_above_i`` when ``control_i >
    threshold``, else ``x_below_i``; $z_i$ is the complementary pick.
    Mirrors C++ ``MixedControlSetup`` in ``src/neml2/models/common/MixedControlSetup.cxx``.
    """

    hit = HitSchema(
        input("control", SR2, "The control signal.", attr="_control"),
        input(
            "x_above",
            SR2,
            "The variable whose values are selected when the control signal is greater "
            "than the threshold.",
            attr="_x_above",
        ),
        input(
            "x_below",
            SR2,
            "The variable whose values are selected when the control signal is less than "
            "or equal to the threshold.",
            attr="_x_below",
        ),
        output(
            "y",
            SR2,
            "The output variable holding the selected values based on the control signal.",
            attr="_y",
        ),
        output(
            "z",
            SR2,
            "The output variable holding the non-selected values based on the control signal.",
            attr="_z",
        ),
        option(
            "threshold",
            float,
            "The threshold to switch between the two controls",
            default=0.5,
            attr="_threshold",
        ),
    )

    _control: str
    _x_above: str
    _x_below: str
    _y: str
    _z: str
    _threshold: float

    def forward(  # type: ignore[override]
        self,
        control: SR2,
        x_above: SR2,
        x_below: SR2,
        v: ChainRuleDict | None = None,
    ):
        above = gt(control, self._threshold)
        y = where(above, x_above, x_below)
        z = where(above, x_below, x_above)
        if v is None:
            return y, z

        # Per-component selectors are diagonal in the Mandel axis: the
        # pushforward weights each tangent component by the 0/1 pick mask
        # (broadcasts over the leading K axis of the SR2 tangent).
        above_mask = above.data.to(control.dtype)  # (*B, 6)
        below_mask = (~above.data).to(control.dtype)  # (*B, 6)

        def y_above(V: SR2) -> SR2:
            return SR2(above_mask * V.data, sub_batch_ndim=V.sub_batch_ndim)

        def y_below(V: SR2) -> SR2:
            return SR2(below_mask * V.data, sub_batch_ndim=V.sub_batch_ndim)

        def z_above(V: SR2) -> SR2:
            return SR2(below_mask * V.data, sub_batch_ndim=V.sub_batch_ndim)

        def z_below(V: SR2) -> SR2:
            return SR2(above_mask * V.data, sub_batch_ndim=V.sub_batch_ndim)

        v_y = self.apply_chain_rule(
            v, self._y, {self._x_above: y_above, self._x_below: y_below}, output=y
        )
        v_z = self.apply_chain_rule(
            v, self._z, {self._x_above: z_above, self._x_below: z_below}, output=z
        )
        # Merge dicts (different outer keys).
        return y, z, {**v_y, **v_z}


__all__ = ["MixedControlSetup"]
