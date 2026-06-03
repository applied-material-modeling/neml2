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

"""Python-native mirror of the C++ ``OrientationRate`` crystal-plasticity leaf."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2, WR2, r2_from_sr2, skew


@register_native("OrientationRate")
class OrientationRate(Model):
    r"""Defines the rate of the crystal orientations as a spin given by
    $\Omega^e = w - w^p - \varepsilon d^p + d^p \varepsilon$ where
    $\Omega^e = \dot{Q} Q^T$, $Q$ is the orientation, $w$ is
    the vorticity, $w^p$ is the plastic vorticity, $d^p$ is the
    plastic deformation rate, and $\varepsilon$ is the elastic stretch.
    """

    hit = HitSchema(
        input("elastic_strain", SR2, "The name of the elastic strain tensor"),
        input("vorticity", WR2, "The name of the vorticity tensor"),
        input("plastic_deformation_rate", SR2, "The name of the plastic deformation rate"),
        input("plastic_vorticity", WR2, "The name of the plastic vorticity"),
        output("orientation_rate", WR2, "The name of the orientation rate"),
    )

    def forward(  # type: ignore[override]
        self,
        e: SR2,
        w: WR2,
        dp: SR2,
        wp: WR2,
        v: ChainRuleDict | None = None,
    ):
        # wrapper ops carry sub_batch_ndim through; the
        # ``R2 @ R2`` / ``WR2 - WR2`` chain aligns global ``w`` against
        # per-crystal ``wp``/``dp``/``e`` automatically.
        E = r2_from_sr2(e)
        D = r2_from_sr2(dp)
        twist = skew(D @ E - E @ D)  # WR2 with per-crystal sub_batch
        out = w - wp + twist
        if v is None:
            return out

        # Differential pushforwards. Ω = w − wp + skew(DE − ED), linear
        # in each input; the commutator product rule is pure typed 3×3 algebra:
        #   w:  V        wp: −V
        #   e:  dΩ = skew(D·dE − dE·D)   with dE = r2_from_sr2(V)
        #   dp: dΩ = skew(dD·E − E·dD)   with dD = r2_from_sr2(V)
        def w_action(V: WR2) -> WR2:
            return V

        def wp_action(V: WR2) -> WR2:
            return -V

        def e_action(V: SR2) -> WR2:
            dE = r2_from_sr2(V)
            return skew(D @ dE - dE @ D)

        def dp_action(V: SR2) -> WR2:
            dD = r2_from_sr2(V)
            return skew(dD @ E - E @ dD)

        return out, self.apply_chain_rule(
            v,
            "orientation_rate",
            {
                "vorticity": w_action,
                "plastic_vorticity": wp_action,
                "elastic_strain": e_action,
                "plastic_deformation_rate": dp_action,
            },
            output=out,
        )
