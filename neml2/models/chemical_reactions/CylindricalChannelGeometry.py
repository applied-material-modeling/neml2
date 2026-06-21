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

"""Python-native mirror of the C++ ``CylindricalChannelGeometry`` model."""

from __future__ import annotations

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar, clamp, gt, sqrt, where
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("CylindricalChannelGeometry")
class CylindricalChannelGeometry(Model):
    r"""Dimensionless inner/outer radii of a cylindrical reaction product.

    Given the volume fractions $\phi_s$ (solid) and $\phi_p$
    (product), define $\mathrm{cap} = 1 - \phi_s - \phi_p$. Then

    $$
    r_i = \sqrt{\operatorname{clamp}(\mathrm{cap}, \varepsilon, 1 - \varepsilon)},
    \qquad
    r_o = \sqrt{1 - \phi_s}.
    $$

    The clamp guards :func:`sqrt` against negative arguments and the flat
    saturation tails kill the derivative of $r_i$ exactly as in the
    C++ source.
    """

    hit = HitSchema(
        input("solid_fraction", Scalar, "Volume fraction of the solid phase"),
        input("product_fraction", Scalar, "Volume fraction of the product phase"),
        output("inner_radius", Scalar, "Dimensionless inner radius of the product phase"),
        output("outer_radius", Scalar, "Dimensionless outer radius of the product phase"),
    )

    def forward(  # type: ignore[override]
        self,
        solid_fraction: Scalar,
        product_fraction: Scalar,
        v: ChainRuleDict | None = None,
    ):
        phi_s = solid_fraction
        phi_p = product_fraction

        eps = torch.finfo(phi_s.dtype).eps
        cap = 1.0 - phi_s - phi_p
        cap_clamped = clamp(cap, eps, 1.0 - eps)
        ri = sqrt(cap_clamped)
        ro = sqrt(1.0 - phi_s)
        if v is None:
            return ri, ro

        # Differential pushforward.
        # Inside the saturation tails (cap <= eps or cap >= 1 - eps) the clamp
        # is flat, so dri/d(phi_s) = dri/d(phi_p) = 0. The upper saturation is
        # unreachable from physical fractions (would need phi_s + phi_p < eps);
        # the C++ source only guards the lower tail and we mirror that exactly:
        # use ``gt(cap, eps)`` as the in-active-region mask.
        zero = Scalar.from_value(0.0, like=ri)
        dri_dphi = where(gt(cap, eps), -0.5 / ri, zero)
        dro_dphi_s = -0.5 / ro

        v_ri = self.apply_chain_rule(
            v,
            "inner_radius",
            {
                "solid_fraction": lambda V, c=dri_dphi: c * V,
                "product_fraction": lambda V, c=dri_dphi: c * V,
            },
            output=ri,
        )
        v_ro = self.apply_chain_rule(
            v,
            "outer_radius",
            {"solid_fraction": lambda V, c=dro_dphi_s: c * V},
            output=ro,
        )
        return ri, ro, {**v_ri, **v_ro}


__all__ = ["CylindricalChannelGeometry"]
