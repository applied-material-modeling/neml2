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

"""Viscous relaxation of a damage variable toward a target (Brandyberry eq 24)."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import Scalar, gt, where
from ...model import Model


@register_neml2_object("ViscousDamageRelaxation")
class ViscousDamageRelaxation(Model):
    r"""Backward-Euler viscous relaxation of a damage variable toward a
    target -- Brandyberry, Zhang & Geubelle (2022) eq (24).

    Under loading (:math:`g = \omega^{\mathrm{target}} - \omega^{n-1} \ge 0`):

    .. math::
        \omega^{n} = \frac{\omega^{n-1} + \mu_{\mathrm{visc}}\,\Delta t\,\omega^{\mathrm{target}}}
                          {1 + \mu_{\mathrm{visc}}\,\Delta t}

    Under unloading (:math:`g < 0`): :math:`\omega^{n} = \omega^{n-1}` (damage
    is monotone -- never heals).

    Parameter roles:

    * :math:`\mu_{\mathrm{visc}}` -- damage fluidity coefficient (units: 1/time).
      Small values retard damage growth (curves toward the elastic limit);
      large values approach the rate-independent target
      (:math:`\omega^{n} \to \omega^{\mathrm{target}}`).

    Typical values: :math:`\mu_{\mathrm{visc}} \approx 20\ \mathrm{s}^{-1}`
    for the concrete/composite regime Brandyberry study; effectively
    rate-independent for slow quasi-static loading.

    Intended composition:

    * Upstream leaf computes ``target = G(psi_0)`` (any G function -- Weibull,
      exponential softening, linear, etc.). This leaf is G-agnostic.
    * This leaf relaxes toward ``target`` with fluidity ``mu_visc``.
    * Downstream :class:`DamagedStress` applies ``sigma = (1 - omega) * sigma_tilde``.

    Reference: Brandyberry, D. R., Zhang, X., & Geubelle, P. H. (2022),
    "Multiscale design of nonlinear materials using reduced-order modeling",
    *Comput. Methods Appl. Mech. Engrg.* 399, 115388, eq (24).
    Framework: Simo, J. C. & Ju, J. W. (1987) *Int. J. Solids Struct.*
    23(7), 841-869 sec. 3.4 (viscous regularization structure).
    """

    hit = HitSchema(
        input("target", Scalar, "Rate-independent target damage (typically G(psi_0))"),
        input("time", Scalar, "Current time", default="t", attr="_t"),
        output("omega", Scalar, "Viscously-relaxed damage variable in [0, 1)"),
        derived_input("omega", Scalar, attr="_omega_prev", suffix="~1"),
        derived_input("time", Scalar, attr="_t_prev", suffix="~1"),
        parameter(
            "mu_visc",
            Scalar,
            "Damage fluidity coefficient (1/time). "
            "mu_visc*dt >> 1 recovers rate-independent target; "
            "mu_visc*dt << 1 gives near-elastic response.",
            attr="mu_visc",
            allow_promotion=True,
        ),
    )

    mu_visc: Scalar
    _t: str
    _omega_prev: str
    _t_prev: str

    def __post_init__(self) -> None:
        # First-order chain rule via reverse-mode autograd; primal composes
        # smooth typed primitives (arithmetic + where) that autograd handles
        # cleanly. The `where` non-smoothness at the loading/unloading
        # boundary is measure-zero and both branches are correctly
        # sub-differentiable.
        self.request_AD()

    def forward(  # type: ignore[override]
        self,
        target: Scalar,
        t: Scalar,
        omega_prev: Scalar,
        t_prev: Scalar,
        *promoted_params,
    ) -> Scalar:
        mu = self._get_param("mu_visc", promoted_params, Scalar)
        dt = t - t_prev
        mu_dt = mu * dt

        # Loading branch: viscous relaxation toward target.
        omega_load = (omega_prev + mu_dt * target) / (Scalar.from_value(1.0, like=mu_dt) + mu_dt)

        # Unloading branch: freeze at previous value.
        # Kuhn-Tucker gate: g = target - omega_prev >= 0 <=> target > omega_prev
        # (strictly >; at equality the two branches agree so either is fine).
        loading = gt(target, omega_prev)
        return where(loading, omega_load, omega_prev)


__all__ = ["ViscousDamageRelaxation"]
