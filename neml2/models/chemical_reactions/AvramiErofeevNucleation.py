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

"""Python-native mirror of the C++ ``AvramiErofeevNucleation`` model."""

from __future__ import annotations

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, pow
from ..chain_rule import ChainRuleDict
from ..model import Model


def _log(s: Scalar) -> Scalar:
    """Element-wise natural log on a Scalar wrapper.

    Mirrors :func:`neml2.types.sqrt` / :func:`exp` — a trivial typed
    wrapper around ``torch.log`` that preserves ``sub_batch_ndim``.
    """
    return Scalar(torch.log(s.data), sub_batch_ndim=s.sub_batch_ndim)


@register_neml2_object("AvramiErofeevNucleation")
class AvramiErofeevNucleation(Model):
    r"""Avrami--Erofeev nucleation reaction mechanism.

    Maps the conversion degree :math:`a` to the reaction rate

    .. math:: f = k (1 - a) \bigl(-\ln(1 - a)\bigr)^n

    where :math:`k` is the reaction coefficient and :math:`n` is the reaction
    order.
    """

    hit = HitSchema(
        input("conversion_degree", Scalar, "Degree of conversion"),
        output("reaction_rate", Scalar, "Reaction rate"),
        parameter("coef", Scalar, "Reaction coefficient", attr="k", allow_nonlinear=True),
        parameter("order", Scalar, "Reaction order", attr="n", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the ``coef`` / ``order`` parameters (stored as
    # ``k`` / ``n``) — no __init__ needed. The annotations let pyright see the
    # typed wrappers that ``Model.__getattr__`` returns.
    k: Scalar
    n: Scalar

    def forward(  # type: ignore[override]
        self,
        conversion_degree: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        a = conversion_degree
        k = self._get_param("k", nl_params, Scalar)
        n = self._get_param("n", nl_params, Scalar)

        u = 1.0 - a  # 1 - a
        mlog_u = -_log(u)  # -log(1 - a)
        u_pow_n = pow(mlog_u, n)  # (-log(1-a))^n
        f = k * u * u_pow_n
        if v is None:
            return f

        # Differential pushforward.
        # ∂f/∂a = k * [ n * (-log(1-a))^(n-1) - (-log(1-a))^n ]
        u_pow_nm1 = pow(mlog_u, n - 1.0)
        df_da = k * (n * u_pow_nm1 - u_pow_n)
        # ∂f/∂k = (1 - a) * (-log(1-a))^n
        df_dk = u * u_pow_n
        # ∂f/∂n = f * log(-log(1-a))   (valid for a > 0)
        df_dn = f * _log(mlog_u)

        actions = {"conversion_degree": lambda V, c=df_da: c * V}
        if "k" in self._nl_params:
            actions[self._nl_params["k"].input_name] = lambda V, c=df_dk: c * V
        if "n" in self._nl_params:
            actions[self._nl_params["n"].input_name] = lambda V, c=df_dn: c * V
        return f, self.apply_chain_rule(v, "reaction_rate", actions, output=f)
