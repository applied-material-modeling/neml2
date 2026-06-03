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

"""Python-native mirror of the C++ ``ModeMixity`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import Scalar, gt, where


@register_native("ModeMixity")
class ModeMixity(Model):
    r"""Mode-mixity ratio $\beta = \delta_s / \delta_n^+$ in the opening
    branch; $\beta = 0$ in compression. Uses a safe-divisor
    `where`-and-detach pattern so the masked-off compression branch does not
    trip on a division by zero.
    """

    hit = HitSchema(
        input(
            "normal_separation",
            Scalar,
            "Normal separation (typically the Macaulay-positive part of the normal jump)",
        ),
        input("tangential_separation", Scalar, "Tangential separation magnitude"),
        output("mode_mixity", Scalar, "Mode-mixity ratio"),
    )

    def forward(  # type: ignore[override]
        self,
        dn: Scalar,
        ds: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Detached opening-branch mask (matches C++ ``(_dn() > 0.0).detach()``):
        # the masked compression branch must not contribute through ``dn``.
        pos_mask = gt(dn, 0.0)
        one = Scalar.from_value(1.0, like=dn)
        zero = Scalar.from_value(0.0, like=dn)
        # Safe denominator: replace dn with 1 on the masked-off branch so the
        # opening-branch division stays finite when dn <= 0.
        safe_dn = where(pos_mask, dn, one)
        inv_dn = 1.0 / safe_dn

        beta_open = ds * inv_dn
        beta = where(pos_mask, beta_open, zero)

        if v is None:
            return beta

        # -------- D-062 pushforward.
        #
        # In the opening branch (delta_n > 0):
        #   d(beta)/d(delta_s)     =  1 / delta_n
        #   d(beta)/d(delta_n_pos) = -delta_s / delta_n^2
        # In the compression branch beta is identically zero, so both partials
        # are zero. The branch mask itself is detached, so its variation does
        # not contribute.
        dbeta_dds = where(pos_mask, inv_dn, zero)
        dbeta_ddn = where(pos_mask, -ds * inv_dn * inv_dn, zero)

        actions: dict[str, ChainRuleAction] = {
            "tangential_separation": lambda V, c=dbeta_dds: c * V,
            "normal_separation": lambda V, c=dbeta_ddn: c * V,
        }
        return beta, self.apply_chain_rule(v, "mode_mixity", actions, output=beta)


__all__ = ["ModeMixity"]
