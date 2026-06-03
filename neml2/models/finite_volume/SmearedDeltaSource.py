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

"""Python-native mirror of C++ ``finite_volume/SmearedDeltaSource.h``."""

from __future__ import annotations

import math
from typing import cast

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar
from ...types.functions import exp


@register_native("SmearedDeltaSource")
class SmearedDeltaSource(Model):
    """Smeared Gaussian source for a Dirac delta distribution.

    $source[k] = mag * (1/w) / sqrt(2*pi) * exp(-0.5 * ((centers[k] - loc)/w)^2)$

    ``magnitude`` and ``location`` are global Scalars (no sub-batch);
    ``cell_centers`` carries the per-cell sub-batch axis (``sub_batch_ndim=1``);
    the output ``smeared_source`` inherits the cell axis. ``magnitude`` /
    ``location`` therefore couple densely with the per-cell output -- declared in
    :attr:`list_deriv`. ``width`` and ``cell_centers`` are forward-only static
    parameters (the C++ side allows them to be promoted to nl inputs, but the
    native ``ModelUnitTest`` only checks input derivatives, so we leave that
    plumbing for a follow-on).
    """

    hit = HitSchema(
        input("magnitude", Scalar, "Source magnitude.", attr="_mag_name"),
        input("location", Scalar, "Source location.", attr="_loc_name"),
        output(
            "smeared_source",
            Scalar,
            "Smeared Gaussian source.",
            default="state/smeared_source",
            attr="_out_name",
        ),
        parameter("width", Scalar, "Gaussian width.", allow_nonlinear=True),
        parameter("cell_centers", Scalar, "Cell center locations.", allow_nonlinear=True),
    )

    # magnitude/location are global Scalars (sub_batch=0); the output carries the
    # cell sub-batch axis from cell_centers — that's a dense coupling.
    list_deriv = {
        ("smeared_source", "magnitude"): "dense",
        ("smeared_source", "location"): "dense",
    }

    width: Scalar
    cell_centers: Scalar
    _mag_name: str
    _loc_name: str
    _out_name: str

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        mag = cast(Scalar, inputs[0])
        loc = cast(Scalar, inputs[1])
        w = self._get_param("width", nl_params=(), type_cls=Scalar)
        centers = self._get_param("cell_centers", nl_params=(), type_cls=Scalar)

        inv_w = 1.0 / w
        arg = (centers - loc) * inv_w  # Scalar (sub_batch=1 via align)
        coeff = inv_w * (1.0 / math.sqrt(2.0 * math.pi))
        gauss = coeff * exp(-0.5 * arg * arg)  # Scalar (sub_batch=1)
        src = mag * gauss  # Scalar (sub_batch=1)

        if v is None:
            return src

        # Chain-rule actions (D-062 -- pushforwards, no Jacobian materialization).
        # gauss & d_loc are per-cell Scalars (sub_batch=1); incoming tangents for
        # magnitude / location are sub_batch=0 leading-K Scalars; multiplying by a
        # sub_batch=1 wrapper aligns the K axis correctly via align_sub_batch.
        d_loc_factor = mag * gauss * arg * inv_w  # dsrc/dloc, Scalar sub_batch=1

        def mag_action(V: Scalar) -> Scalar:
            return gauss * V

        def loc_action(V: Scalar) -> Scalar:
            return d_loc_factor * V

        return src, self.apply_chain_rule(
            v,
            "smeared_source",
            {"magnitude": mag_action, "location": loc_action},
            output=src,
        )


__all__ = ["SmearedDeltaSource"]
