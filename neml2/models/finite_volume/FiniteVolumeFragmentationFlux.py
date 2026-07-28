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

"""Assemble the finite-volume fragmentation flux operator for a population balance."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar, cumsum, triu
from ...types.functions import fullify
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("FiniteVolumeFragmentationFlux")
class FiniteVolumeFragmentationFlux(Model):
    """Assemble the interior-edge fragment-flux operator of a population balance.

    For the fragmentation-only population balance in particle-volume
    coordinate, with mass density $u = \\rho v n$ on $N$ cells, the
    conservative form $\\partial_t u + \\partial_v J = 0$ has interior-edge
    flux $J = M u$. This model builds the $(N-1) \\times N$ operator $M$ from
    the per-cell density, volume, width, fragmentation rate, and the breakage
    matrix $p$.

    $$
    K_{kj} = \\Delta v_j\\, \\Delta v_k\\, \\gamma_j\\,
             \\frac{\\rho_k v_k}{\\rho_j v_j}\\, p_{kj},
    \\qquad
    M_{ij} = -\\sum_{k \\le i} K_{kj}\\ \\text{for } j > i,\\ 0\\ \\text{else}.
    $$

    The row axis $k$ is the child class, the column axis $j$ is the parent.
    $M$ is a cumulative sum of $K$ over $k$, masked to the strict upper
    triangle, negated, with the last (all-zero) row dropped. $M$ is
    independent of $u$; the contraction $M u$ is applied downstream by
    ``IntermediateLinearContraction``.
    """

    hit = HitSchema(
        input("cell_density", Scalar, "Per-cell material density.", attr="_cell_density_name"),
        input(
            "cell_volume",
            Scalar,
            "Per-cell particle volume (size coordinate at cell centers).",
            attr="_cell_volume_name",
        ),
        input(
            "cell_width",
            Scalar,
            "Per-cell width in the volume (size) coordinate.",
            attr="_cell_width_name",
        ),
        input(
            "fragmentation_rate",
            Scalar,
            "Per-cell fragmentation rate.",
            attr="_fragmentation_rate_name",
        ),
        input(
            "breakage_matrix",
            Scalar,
            "Breakage matrix p[k,j] (child k from parent j).",
            attr="_breakage_matrix_name",
        ),
        output(
            "flux_operator",
            Scalar,
            "Interior-edge fragment-flux operator (N-1, N).",
            attr="_flux_operator_name",
        ),
    )

    _cell_density_name: str
    _cell_volume_name: str
    _cell_width_name: str
    _fragmentation_rate_name: str
    _breakage_matrix_name: str
    _flux_operator_name: str

    @staticmethod
    def _assemble(K: Scalar) -> Scalar:
        # M = -cumsum_k(K), strict upper triangle (j > i), last row dropped;
        # linear in K, so it serves both the value and every pushforward.
        C = cumsum(K.sub_batch, dim=0)
        U = triu(C.sub_batch, diagonal=1)
        return (-U).sub_batch[:-1]

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        rho, vol, dv, gamma, p = inputs
        # Broadcast per-cell fields onto the (k, j) grid (k = child/row, j = parent/col).
        rho_k = rho.sub_batch.unsqueeze(1)
        rho_j = rho.sub_batch.unsqueeze(0)
        v_k = vol.sub_batch.unsqueeze(1)
        v_j = vol.sub_batch.unsqueeze(0)
        dv_k = dv.sub_batch.unsqueeze(1)
        dv_j = dv.sub_batch.unsqueeze(0)
        gam_j = gamma.sub_batch.unsqueeze(0)

        inv_rv_j = 1.0 / (rho_j * v_j)
        ratio = (rho_k * v_k) * inv_rv_j
        pref = dv_k * dv_j * gam_j
        K = pref * ratio * p
        out = self._assemble(K)
        if v is None:
            return out

        # Elementwise dK/d(input) coefficients. gamma and p may be zero, so their
        # coefficients omit that factor instead of dividing; rho, v > 0, so the
        # j-slot quotients are safe.
        coeff_p = pref * ratio
        coeff_g = dv_k * dv_j * ratio * p
        coeff_dvk = dv_j * gam_j * ratio * p
        coeff_dvj = dv_k * gam_j * ratio * p
        coeff_rhok = pref * p * v_k * inv_rv_j
        coeff_vk = pref * p * rho_k * inv_rv_j
        coeff_rhoj = -K / rho_j
        coeff_vj = -K / v_j

        # Each action fullifies its tangent (cumsum/triu in _assemble mix the row
        # axis, so the K-paired grid must be materialised first). dv couples both
        # k and j (product rule); rho and v are numerator at k, denominator at j
        # (quotient rule); gamma and p enter only at one slot.
        def p_action(V_in: Scalar) -> Scalar:
            return self._assemble(coeff_p * fullify(V_in))

        def gamma_action(V_in: Scalar) -> Scalar:
            return self._assemble(coeff_g * fullify(V_in).sub_batch.unsqueeze(0))

        def dv_action(V_in: Scalar) -> Scalar:
            Vf = fullify(V_in)
            dK = coeff_dvk * Vf.sub_batch.unsqueeze(1) + coeff_dvj * Vf.sub_batch.unsqueeze(0)
            return self._assemble(dK)

        def rho_action(V_in: Scalar) -> Scalar:
            Vf = fullify(V_in)
            dK = coeff_rhok * Vf.sub_batch.unsqueeze(1) + coeff_rhoj * Vf.sub_batch.unsqueeze(0)
            return self._assemble(dK)

        def v_action(V_in: Scalar) -> Scalar:
            Vf = fullify(V_in)
            dK = coeff_vk * Vf.sub_batch.unsqueeze(1) + coeff_vj * Vf.sub_batch.unsqueeze(0)
            return self._assemble(dK)

        actions = {
            self._cell_density_name: rho_action,
            self._cell_volume_name: v_action,
            self._cell_width_name: dv_action,
            self._fragmentation_rate_name: gamma_action,
            self._breakage_matrix_name: p_action,
        }
        return out, self.apply_chain_rule(v, self._flux_operator_name, actions, output=out)


__all__ = ["FiniteVolumeFragmentationFlux"]
