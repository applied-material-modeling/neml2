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

"""Linear slip-system interaction strength map."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sum
from ....types.functions import fullify
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("LinearInteractionStrengthMap")
class LinearInteractionStrengthMap(Model):
    r"""Per-slip hardening state to strength through a linear slip-system interaction matrix,

    $\tau_i = \tau_{0,i} + \sum_j h_{ij}\,\phi_j,$

    where the per-slip hardening state $\phi_j$ is commonly the accumulated slip $|\gamma_j|$.
    The interaction matrix $h$ couples slip systems linearly -- latent hardening with a
    constant (state-independent) Jacobian $h_{ij}$ -- in contrast to
    :class:`DislocationInteractionStrengthMap`, whose forest form adds the $\alpha\mu b
    \sqrt{\cdot}$ nonlinearity. With $h = I$ the systems decouple into independent linear
    hardening.

    ``interaction_matrix`` is an $n_\text{slip}\times n_\text{slip}$ :class:`~neml2.types.Scalar`
    carried on two sub-batch axes (build it with the ``SquareMatrix`` ``[Tensors]`` type --
    fully, or from a handful of coefficients via its ``block`` / ``diagonal_blocks`` fills).
    ``constant_strength`` is the offset $\tau_0$ (shared, or per-slip if sub-batched). Both
    are promotable/calibratable; the interaction matrix carries a chain-rule action only when
    promoted.
    """

    hit = HitSchema(
        input("slip_hardening", Scalar, "Per-slip hardening state (e.g. accumulated slip)"),
        output("slip_strengths", Scalar, "Per-slip slip system strengths"),
        parameter("constant_strength", Scalar, "Constant strength offset", allow_promotion=True),
        parameter(
            "interaction_matrix",
            Scalar,
            "Slip-system interaction matrix, sub-batched over (n_slip, n_slip)",
            allow_promotion=True,
        ),
    )

    constant_strength: Scalar
    interaction_matrix: Scalar

    @staticmethod
    def _combine(phi: Scalar, h: Scalar) -> Scalar:
        """Linear coupling ``sum_j h_ij phi_j``, contracting the source slip axis."""
        phi_j = phi.sub_batch.unsqueeze(-2)  # (..., 1, n_slip_j)
        return sum((h * phi_j).sub_batch, -1)  # contract j -> (..., n_slip_i)

    @staticmethod
    def _check_consistency(phi: Scalar, h: Scalar) -> None:
        """The interaction matrix must be square and match the per-slip state's slip count."""
        import torch  # local: only for the tracing guard

        if torch.compiler.is_compiling():
            return
        hs = tuple(int(s) for s in h.shape)
        if len(hs) < 2 or hs[-1] != hs[-2]:
            raise ValueError(
                "LinearInteractionStrengthMap: interaction_matrix must be square "
                f"(n_slip x n_slip); got trailing shape {hs[-2:] if len(hs) >= 2 else hs}."
            )
        n_phi = int(phi.shape[-1]) if phi.ndim >= 1 else 1
        if hs[-1] != n_phi:
            raise ValueError(
                f"LinearInteractionStrengthMap: interaction_matrix slip count {hs[-1]} does "
                f"not match slip_hardening slip count {n_phi}."
            )

    def forward(  # type: ignore[override]
        self,
        phi: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        tau_const = self._get_param("constant_strength", promoted_params, Scalar)
        h = self._get_param("interaction_matrix", promoted_params, Scalar)

        self._check_consistency(phi, h)
        tau = tau_const + self._combine(phi, h)

        if v is None:
            return tau

        # d tau_i / d phi_j = h_ij (dense over j, constant) -- fullify only the source axis.
        def phi_action(V: Scalar, hh: Scalar = h) -> Scalar:
            V = fullify(V, sub_axis=-1)
            return sum((hh * V.sub_batch.unsqueeze(-2)).sub_batch, -1)

        actions: dict[str, ChainRuleAction] = {"slip_hardening": phi_action}

        if "constant_strength" in self._promoted_params:
            actions[self._promoted_params["constant_strength"].input_name] = lambda V: V
        if "interaction_matrix" in self._promoted_params:
            # d tau_i / d h_ij = phi_j (per destination i).
            def h_action(V: Scalar, pp: Scalar = phi) -> Scalar:
                V = fullify(V, sub_axis=-1)
                return sum((V * pp.sub_batch.unsqueeze(-2)).sub_batch, -1)

            actions[self._promoted_params["interaction_matrix"].input_name] = h_action

        return tau, self.apply_chain_rule(v, "slip_strengths", actions, output=tau)
