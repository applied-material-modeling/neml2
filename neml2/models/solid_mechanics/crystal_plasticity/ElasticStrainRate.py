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

"""Python-native mirror of the C++ ``ElasticStrainRate`` crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import SR2, WR2, r2_from_sr2, r2_from_wr2, sym
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("ElasticStrainRate")
class ElasticStrainRate(Model):
    r"""Calculates the elastic strain rate as
    $\dot{\varepsilon} = d - d^p - \varepsilon w + w \varepsilon$ where
    $d$ is the deformation rate, $d^p$ is the plastic deformation
    rate, $w$ is the vorticity, and $\varepsilon$ is the elastic
    strain.
    """

    hit = HitSchema(
        input("elastic_strain", SR2, "Name of the elastic strain"),
        input("deformation_rate", SR2, "Name of the deformation rate"),
        input("vorticity", WR2, "Name of the vorticity"),
        input("plastic_deformation_rate", SR2, "Name of the plastic deformation rate"),
        output("elastic_strain_rate", SR2, "Name of the elastic strain rate"),
    )

    def forward(  # type: ignore[override]
        self,
        e: SR2,
        d: SR2,
        w: WR2,
        dp: SR2,
        v: ChainRuleDict | None = None,
    ):
        # ComposedModel preserves sub_batch_ndim across leaf boundaries, so the
        # typed 3×3 ops below align global ``d``/``w`` against per-crystal
        # ``e``/``dp`` automatically. The whole forward (and its D-062
        # pushforward) stays in typed wrapper algebra — no raw-tensor Jacobian
        # kernels.
        E = r2_from_sr2(e)  # per-crystal R2
        W = r2_from_wr2(w)  # global R2
        twist = sym(W @ E - E @ W)  # SR2 with sub_batch from E
        out = d - dp + twist
        if v is None:
            return out

        # Differential pushforwards. ε̇ = d − dp + sym(WE − EW) is linear
        # in each input; the product rule on the commutator is pure typed 3×3
        # algebra (no Jacobian kernels):
        #   d:  dε̇ = V        dp: dε̇ = −V
        #   e:  dε̇ = sym(W·dE − dE·W)   with dE = r2_from_sr2(V)
        #   w:  dε̇ = sym(dW·E − E·dW)   with dW = r2_from_wr2(V)
        def d_action(V: SR2) -> SR2:
            return V

        def dp_action(V: SR2) -> SR2:
            return -V

        def w_action(V: WR2) -> SR2:
            dW = r2_from_wr2(V)
            return sym(dW @ E - E @ dW)

        def e_action(V: SR2) -> SR2:
            dE = r2_from_sr2(V)
            return sym(W @ dE - dE @ W)

        return out, self.apply_chain_rule(
            v,
            "elastic_strain_rate",
            {
                "deformation_rate": d_action,
                "plastic_deformation_rate": dp_action,
                "vorticity": w_action,
                "elastic_strain": e_action,
            },
            output=out,
        )
