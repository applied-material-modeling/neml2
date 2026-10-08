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

"""Forest-dislocation strength with a slip-system interaction (latent-hardening) matrix."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sqrt, sum
from ....types.functions import fullify
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("DislocationInteractionStrengthMap")
class DislocationInteractionStrengthMap(Model):
    r"""Dislocation density to strength with a slip-system interaction matrix,

    $\tau_i = \tau_{const} + \alpha \mu b \sqrt{\sum_r h_{ir} \rho_r}$.

    This generalizes :class:`DislocationObstacleStrengthMap` (which is the
    diagonal, self-hardening special case $h = I$) by letting the forest density
    seen by slip system $i$ be a weighted sum over all systems -- classical
    latent hardening, where dislocations on system $r$ obstruct system $i$
    through the interaction matrix $h_{ir}$.

    The interaction matrix ``interaction_matrix`` is a **parameter**: a
    :class:`~neml2.types.Scalar` carried on two sub-batch axes
    ``(n_slip, n_slip)`` -- the same representation
    :class:`SlipSystemElasticInteraction` uses for a slip-system matrix. As a
    parameter it is model-agnostic in how it is supplied (a ``[Tensors]``
    constant, a promoted/calibratable value, or another ``[Models]`` output such
    as a junction-category classifier) and general in ``n_slip``. With $h = I$
    the map reduces exactly to :class:`DislocationObstacleStrengthMap`.

    The scalar coefficients ``constant_strength``, ``alpha``, ``mu`` and ``b``
    are shared across slip systems. All parameters are promotable/calibratable;
    for the interaction matrix, a chain-rule action is registered only when it is
    promoted (calibrated) -- when it is a static material property it carries no
    sensitivity, so no derivative is assembled for it.

    Non-negativity: the forest sum $\sum_r h_{ir}\rho_r$ must be $\ge 0$ for every
    system (the sqrt and its derivative $\propto 1/\sqrt{\cdot}$ are real only then).
    This holds automatically for $\rho \ge 0$ and a non-negative $h$ (e.g.
    $h = I + q(1-I)$ with $q \ge 0$); a general/calibrated $h$ must be constrained to
    keep it non-negative, else the value and gradient are NaN.
    """

    # Variable / parameter names match the C++ / DislocationObstacleStrengthMap
    # convention so the same .i scenarios drive both.
    hit = HitSchema(
        input("dislocation_density", Scalar, "Per-slip dislocation density"),
        output("slip_strengths", Scalar, "Name of the slip system strengths"),
        parameter("constant_strength", Scalar, "Constant strength offset", allow_promotion=True),
        parameter("alpha", Scalar, "Interaction coefficient", allow_promotion=True),
        parameter("mu", Scalar, "Shear modulus", allow_promotion=True),
        parameter("b", Scalar, "Burgers vector", allow_promotion=True),
        parameter(
            "interaction_matrix",
            Scalar,
            "Slip-system interaction (latent-hardening) matrix, sub-batched over (n_slip, n_slip)",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the parameters via the schema.
    constant_strength: Scalar
    alpha: Scalar
    mu: Scalar
    b: Scalar
    interaction_matrix: Scalar

    @staticmethod
    def _forest_density(rho: Scalar, h: Scalar) -> Scalar:
        r"""Forest density $\sum_r h_{ir}\rho_r$, contracted over the source slip axis.

        ``rho`` carries the slip axis as its trailing sub-batch axis; ``h`` carries
        the destination/source slip pair ``(i, r)`` as its two trailing sub-batch
        axes. The ``sub_batch_ndim`` hint resets across ``ComposedModel``
        boundaries, so the trailing sub-batch axes are used positionally (as in
        :class:`SumSlipRates`), not read from the hint.
        """
        rho_r = rho.sub_batch.unsqueeze(-2)  # (..., 1, n_slip_r)
        return sum((h * rho_r).sub_batch, -1)  # contract r -> (..., n_slip_i)

    @staticmethod
    def _check_consistency(rho: Scalar, h: Scalar) -> None:
        """Dimension check: the interaction matrix must be square and its slip count
        must match the dislocation density, giving a clear error instead of a raw
        broadcast failure. Skipped under tracing, where the shapes are already
        fixed/validated eagerly."""
        import torch  # local: only for the tracing guard

        if torch.compiler.is_compiling():
            return
        hs = tuple(int(s) for s in h.shape)
        if len(hs) < 2 or hs[-1] != hs[-2]:
            raise ValueError(
                "DislocationInteractionStrengthMap: interaction_matrix must be square "
                f"(n_slip x n_slip); got trailing shape {hs[-2:] if len(hs) >= 2 else hs}."
            )
        n_rho = int(rho.shape[-1]) if rho.ndim >= 1 else 1
        if hs[-1] != n_rho:
            raise ValueError(
                f"DislocationInteractionStrengthMap: interaction_matrix slip count {hs[-1]} "
                f"does not match dislocation_density slip count {n_rho}."
            )

    def forward(  # type: ignore[override]
        self,
        rho: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        tau_const = self._get_param("constant_strength", promoted_params, Scalar)
        alpha = self._get_param("alpha", promoted_params, Scalar)
        mu = self._get_param("mu", promoted_params, Scalar)
        b = self._get_param("b", promoted_params, Scalar)
        h = self._get_param("interaction_matrix", promoted_params, Scalar)

        self._check_consistency(rho, h)
        forest = self._forest_density(rho, h)
        sqrt_forest = sqrt(forest)
        coeff = alpha * mu * b
        tau = tau_const + coeff * sqrt_forest

        if v is None:
            return tau

        # Differential pushforward (closed form, no Jacobian materialised):
        #   d tau_i / d rho_r           = coeff * 0.5 / sqrt(forest_i) * h_ir   (dense over r)
        #   d tau_i / d h_ir            = coeff * 0.5 / sqrt(forest_i) * rho_r  (promoted only)
        #   d tau_i / d constant_strength = 1
        #   d tau_i / d alpha           = mu * b * sqrt(forest_i)
        #   d tau_i / d mu              = alpha * b * sqrt(forest_i)
        #   d tau_i / d b               = alpha * mu * sqrt(forest_i)
        d_coeff = coeff * 0.5 / sqrt_forest  # (..., n_slip_i)

        def rho_action(V: Scalar, c=d_coeff, hh=h) -> Scalar:
            # Cross-slip contraction sum_r h_ir V_r: this mixes the source slip
            # axis, so the K-paired-broadcast (diagonal) assumption on V is
            # broken -- materialise ONLY the trailing (per-slip) paired K axis
            # with ``fullify(sub_axis=-1)`` before contracting. A broader
            # fullify would also densify an outer paired axis (e.g. a per-grain
            # block axis in a coupled Taylor solve) that must stay diagonal.
            V = fullify(V, sub_axis=-1)
            return c * sum((hh * V.sub_batch.unsqueeze(-2)).sub_batch, -1)

        actions: dict[str, ChainRuleAction] = {
            "dislocation_density": rho_action,
        }
        if "constant_strength" in self._promoted_params:
            actions[self._promoted_params["constant_strength"].input_name] = lambda V: V
        if "alpha" in self._promoted_params:
            actions[self._promoted_params["alpha"].input_name] = lambda V, c=mu * b * sqrt_forest: (
                c * V
            )
        if "mu" in self._promoted_params:
            actions[self._promoted_params["mu"].input_name] = lambda V, c=alpha * b * sqrt_forest: (
                c * V
            )
        if "b" in self._promoted_params:
            actions[self._promoted_params["b"].input_name] = lambda V, c=alpha * mu * sqrt_forest: (
                c * V
            )
        if "interaction_matrix" in self._promoted_params:
            # Only when the matrix is promoted (calibrated). sum_r V_ir rho_r,
            # weighted per destination system i (cross-slip mix over r -> fullify
            # only the trailing axis, as in rho_action).
            def h_action(V: Scalar, c=d_coeff, rr=rho) -> Scalar:
                V = fullify(V, sub_axis=-1)
                return c * sum((V * rr.sub_batch.unsqueeze(-2)).sub_batch, -1)

            actions[self._promoted_params["interaction_matrix"].input_name] = h_action

        return tau, self.apply_chain_rule(v, "slip_strengths", actions, output=tau)
