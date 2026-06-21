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

"""Python-native mirror of the C++ ``PrecipitateVolumeFraction`` model."""

from __future__ import annotations

import math

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, pow, sum
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("PrecipitateVolumeFraction")
class PrecipitateVolumeFraction(Model):
    r"""Compute the precipitate volume fraction from a discrete size distribution.

    For a discrete distribution with per-bin radius $R_i$ and per-bin
    number density $n_i$, the precipitate volume fraction is

    $$
    f = \sum_i \tfrac{4}{3} \pi R_i^3 \, n_i
    $$

    where the sum runs over all precipitate size bins.
    """

    hit = HitSchema(
        input("number_density", Scalar, "Number density per size bin"),
        output("volume_fraction", Scalar, "Precipitate volume fraction"),
        parameter(
            "radius",
            Scalar,
            "Precipitate radius per size bin",
            attr="R",
            allow_nonlinear=True,
        ),
    )
    # The reduction sums across the trailing sub-batch (size-bin) axis, so
    # the ``volume_fraction``-vs-``number_density`` edge reduces the ``"bin"``
    # label rather than passing it through diagonally.

    # ``from_hit`` auto-declares the ``radius`` parameter (stored as ``R``).
    # Annotate so pyright sees the typed wrapper that ``Model.__getattr__``
    # returns; runtime resolution is via ``_get_param`` so the same code path
    # handles both static and nl-promoted radii.
    R: Scalar

    def forward(  # type: ignore[override]
        self,
        n: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        R = self._get_param("R", nl_params, Scalar)
        # ``register_typed_parameter`` only stores the raw tensor data, so a
        # per-bin radius round-trips through ``self.R`` with ``sub_batch_ndim=0``
        # even when the user declared it as a per-bin Scalar. Re-tag to match
        # the per-bin layout of ``n`` so the bin-wise product and the trailing
        # sub-batch sum line up. A true scalar radius (no per-bin axis) has
        # data shape ``()`` and falls through unchanged — broadcast handles it.
        if n.sub_batch_ndim > R.sub_batch_ndim and R.ndim >= n.sub_batch_ndim:
            R = R.sub_batch.retag(n.sub_batch_ndim)

        # Forward: f = sum_i (4/3) π R_i^3 n_i. Typed Scalar algebra; both R
        # and n carry sub_batch_ndim>=1 over the size-bin axis, then we reduce
        # along the trailing sub-batch axis to a sub_batch_ndim=0 scalar.
        coef_n = (4.0 / 3.0) * math.pi * pow(R, 3.0)
        volume = coef_n * n
        f = sum(volume.sub_batch, -1)

        if v is None:
            return f

        # Differential pushforward. The forward is linear in n and in
        # any nl-promoted R, so each action is the same reduction applied to
        # the per-bin coefficient times the input tangent:
        #   df/dn_i = (4/3) π R_i^3
        #   df/dR_i = 4 π R_i^2 n_i
        # No Jacobian materialization; the sub_batch_sum collapses the size-bin
        # axis on the tangent the same way it does on the forward.
        actions: dict[str, ChainRuleAction] = {}

        def n_action(V: Scalar) -> Scalar:
            return sum((coef_n * V).sub_batch, -1)

        actions["number_density"] = n_action

        R_nlp = self._nl_params.get("R")
        if R_nlp is not None:
            coef_R = 4.0 * math.pi * pow(R, 2.0) * n

            def R_action(V: Scalar) -> Scalar:
                return sum((coef_R * V).sub_batch, -1)

            actions[R_nlp.input_name] = R_action

        return f, self.apply_chain_rule(v, "volume_fraction", actions, output=f)


__all__ = ["PrecipitateVolumeFraction"]
