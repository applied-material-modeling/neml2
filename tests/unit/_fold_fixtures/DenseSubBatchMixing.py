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

r"""Test-only model producing a DENSE within-sub-batch Jacobian: ``y_i = sum_r M_ir x_r``.

This is the minimal, physics-free trigger for the dense sub-batch fold added to the
equation-system assembler. ``x`` carries a per-site axis as its trailing sub-batch
axis; the mixing matrix ``M`` carries the destination/source pair ``(i, r)`` on two
sub-batch axes ``(n, n)``, so the output on site ``i`` mixes EVERY source site ``r``:

.. math::
    y_i = \sum_r M_{ir}\, x_r, \qquad \frac{\partial y_i}{\partial x_r} = M_{ir}.

The Jacobian ``dy/dx`` is therefore dense over the sub-batch axis (not diagonal),
which is precisely the case the assembler's ``_convert_tangent_to_paired_block``
dense path + ``fullify(sub_axis=-1)`` must fold into per-site base. With ``M = I`` it
collapses to the identity map ``y = x``, whose assembled result must match the
pre-existing diagonal (broadcast) fold -- the dense-vs-diagonal equivalence check.

Test-fixture only; self-registers via ``@register_neml2_object``. It carries no
physics, so it pins the *assembler* mechanism in isolation.
"""

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleAction, ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output, parameter
from neml2.types import Scalar, sum
from neml2.types.functions import fullify


@register_neml2_object("DenseSubBatchMixing")
class DenseSubBatchMixing(Model):
    r"""Dense within-sub-batch linear map ``y_i = sum_r M_ir x_r`` (test fixture).

    The mixing matrix ``mixing_matrix`` is a :class:`~neml2.types.Scalar` carried on
    two trailing sub-batch axes ``(n, n)`` (the ``(i, r)`` pair). It is a promotable
    parameter so the dense pushforward can be checked against finite differences. The
    model exists only to exercise the dense sub-batch Jacobian fold; with ``M = I`` it
    is the identity.
    """

    hit = HitSchema(
        input("x", Scalar, "Per-site input, carrying the site axis as trailing sub-batch"),
        output("y", Scalar, "Per-site output ``y_i = sum_r M_ir x_r``"),
        parameter(
            "mixing_matrix",
            Scalar,
            "Dense mixing matrix, sub-batched over (n, n) = (destination i, source r)",
            allow_promotion=True,
        ),
    )

    mixing_matrix: Scalar

    @staticmethod
    def _mix(x: Scalar, m: Scalar) -> Scalar:
        """Contract the source site axis: ``sum_r M_ir x_r``.

        ``x`` carries the site axis as its trailing sub-batch axis; ``m`` carries
        ``(i, r)`` as its two trailing sub-batch axes. The axes are used positionally
        (the ``sub_batch_ndim`` hint resets across ``ComposedModel`` boundaries).
        """
        x_r = x.sub_batch.unsqueeze(-2)  # (..., 1, n_r)
        return sum((m * x_r).sub_batch, -1)  # contract r -> (..., n_i)

    def forward(  # type: ignore[override]
        self,
        x: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        m = self._get_param("mixing_matrix", promoted_params, Scalar)
        y = self._mix(x, m)

        if v is None:
            return y

        def x_action(V: Scalar, mm: Scalar = m) -> Scalar:
            # dy_i/dx_r = M_ir is dense over r; fullify only the trailing (site) axis.
            V = fullify(V, sub_axis=-1)
            return sum((mm * V.sub_batch.unsqueeze(-2)).sub_batch, -1)

        actions: dict[str, ChainRuleAction] = {"x": x_action}

        if "mixing_matrix" in self._promoted_params:
            # dy_i/dM_ir = x_r (per destination i); cross-site mix over r.
            def m_action(V: Scalar, xx: Scalar = x) -> Scalar:
                V = fullify(V, sub_axis=-1)
                return sum((V * xx.sub_batch.unsqueeze(-2)).sub_batch, -1)

            actions[self._promoted_params["mixing_matrix"].input_name] = m_action

        return y, self.apply_chain_rule(v, "y", actions, output=y)
