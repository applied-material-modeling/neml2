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

"""Linear slip-system interaction hardening rate."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, abs, sign, sum
from ....types.functions import fullify
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("LinearInteractionHardeningRule")
class LinearInteractionHardeningRule(Model):
    r"""Per-slip hardening rate from a linear slip-system interaction matrix,

    $\dot{h}_i = \sum_j M_{ij}\, \lvert \dot{\gamma}_j \rvert,$

    putting the slip-system interaction in the hardening *rate* (evolution); the strength is
    then the trivial per-slip sum $\tau_i = \tau_{0,i} + h_i$, composed from existing
    primitives. The interaction matrix $M$ couples the per-system slip rates into each
    system's hardening rate -- latent hardening with a constant Jacobian $\partial \dot{h}_i /
    \partial \lvert\dot{\gamma}_j\rvert = M_{ij}$. With $M = I$ the systems decouple into
    independent linear hardening.

    The absolute value $\lvert\dot{\gamma}\rvert$ makes hardening accumulate regardless of
    slip direction. ``interaction_matrix`` is an $n_\text{slip}\times n_\text{slip}$
    :class:`~neml2.types.Scalar` carried on two sub-batch axes (build it with the
    ``SquareMatrix`` ``[Tensors]`` type); it is promotable and carries a chain-rule action
    only when promoted.
    """

    hit = HitSchema(
        input(
            "slip_hardening",
            Scalar,
            "Per-slip hardening state. Unused in the rate (which depends only on the slip "
            "rates), present so the time integrator wires the state being evolved.",
        ),
        input("slip_rates", Scalar, "Per-slip system slip rates"),
        output("slip_hardening_rate", Scalar, "Per-slip hardening rate"),
        parameter(
            "interaction_matrix",
            Scalar,
            "Slip-system interaction matrix, sub-batched over (n_slip, n_slip)",
            allow_promotion=True,
        ),
    )

    interaction_matrix: Scalar

    @staticmethod
    def _combine(g: Scalar, m: Scalar) -> Scalar:
        """Linear coupling ``sum_j M_ij g_j``, contracting the source slip axis."""
        g_j = g.sub_batch.unsqueeze(-2)  # (..., 1, n_slip_j)
        return sum((m * g_j).sub_batch, -1)  # contract j -> (..., n_slip_i)

    @staticmethod
    def _check_consistency(gamma_dot: Scalar, m: Scalar) -> None:
        """The interaction matrix must be square and match the slip-rate slip count."""
        import torch  # local: only for the tracing guard

        if torch.compiler.is_compiling():
            return
        ms = tuple(int(s) for s in m.shape)
        if len(ms) < 2 or ms[-1] != ms[-2]:
            raise ValueError(
                "LinearInteractionHardeningRule: interaction_matrix must be square "
                f"(n_slip x n_slip); got trailing shape {ms[-2:] if len(ms) >= 2 else ms}."
            )
        n_g = int(gamma_dot.shape[-1]) if gamma_dot.ndim >= 1 else 1
        if ms[-1] != n_g:
            raise ValueError(
                f"LinearInteractionHardeningRule: interaction_matrix slip count {ms[-1]} does "
                f"not match slip_rates slip count {n_g}."
            )

    def forward(  # type: ignore[override]
        self,
        h: Scalar,
        gamma_dot: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        del h  # the rate depends only on the slip rates (structural zero wrt slip_hardening)
        m = self._get_param("interaction_matrix", promoted_params, Scalar)

        self._check_consistency(gamma_dot, m)
        abs_gamma = abs(gamma_dot)
        rate = self._combine(abs_gamma, m)

        if v is None:
            return rate

        # d rate_i / d gamma_dot_j = M_ij sign(gamma_dot_j) (dense over j).
        sg = sign(gamma_dot)

        def gamma_action(V: Scalar, mm: Scalar = m, s: Scalar = sg) -> Scalar:
            V = fullify(V, sub_axis=-1)
            return sum((mm * (s * V).sub_batch.unsqueeze(-2)).sub_batch, -1)

        actions: dict[str, ChainRuleAction] = {"slip_rates": gamma_action}

        if "interaction_matrix" in self._promoted_params:
            # d rate_i / d M_ij = |gamma_dot_j| (per destination i).
            def m_action(V: Scalar, ag: Scalar = abs_gamma) -> Scalar:
                V = fullify(V, sub_axis=-1)
                return sum((V * ag.sub_batch.unsqueeze(-2)).sub_batch, -1)

            actions[self._promoted_params["interaction_matrix"].input_name] = m_action

        return rate, self.apply_chain_rule(v, "slip_hardening_rate", actions, output=rate)
