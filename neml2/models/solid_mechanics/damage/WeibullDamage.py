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

"""Three-parameter Weibull damage law (Simo-Ju family, Brandyberry variant)."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, exp, macaulay, pow
from ...model import Model


@register_neml2_object("WeibullDamage")
class WeibullDamage(Model):
    r"""Three-parameter Weibull damage law -- Brandyberry et al. (2022)
    variant of Simo & Ju (1987) isotropic elastic damage.

    .. math::
        D(r) \;=\; 1 \;-\; \exp\!\left[\;-\left(\frac{\langle r-Y_\mathrm{in}\rangle_+}
                                              {p_1\,Y_\mathrm{in}}\right)^{p_2}\;\right]

    Reads the damage-threshold history variable :math:`r` (typically the output
    of :class:`IrreversibleScalar` acting on the elastic strain energy density
    :math:`\psi_0`), returns the isotropic scalar damage :math:`D \in [0, 1)`.

    Parameter roles:

    * ``Y_in`` -- damage-onset energy threshold. Below :math:`r = Y_\mathrm{in}`
      the Macaulay bracket zeros the argument, so :math:`D = 0` exactly.
    * ``p_1`` -- Weibull scale, "how quickly damage grows once initiated".
      Small ``p_1`` (~0.01) gives cliff-like brittle collapse; large ``p_1``
      (~100) gives very gradual softening.
    * ``p_2`` -- Weibull shape, controls the softening curve shape.
      ``p_2 = 1`` recovers a pure exponential; ``p_2 > 1`` sharpens the
      knee into a sigmoidal shape.

    Wired downstream to :class:`DamagedStress` to obtain the nominal stress
    :math:`\sigma = (1 - D)\tilde{\sigma}`. All three parameters may be
    promoted to runtime inputs for gradient-based calibration.

    Reference: Brandyberry, D. R., Zhang, X., & Geubelle, P. H. (2022),
    "Multiscale design of nonlinear materials using reduced-order modeling",
    *Comput. Methods Appl. Mech. Engrg.* 399, 115388, Appendix A.2 eq (23).
    Underlying framework: Simo, J. C. & Ju, J. W. (1987),
    *Int. J. Solids Struct.* 23(7), 821-869 (Parts I & II).
    """

    hit = HitSchema(
        input("r", Scalar, "Damage threshold history variable (max-history of psi_0)"),
        output("D", Scalar, "Isotropic scalar damage in [0, 1)"),
        parameter(
            "Y_in",
            Scalar,
            "Damage-onset energy threshold (same units as r)",
            attr="Y_in",
            allow_promotion=True,
        ),
        parameter(
            "p1",
            Scalar,
            "Weibull scale parameter (dimensionless)",
            attr="p1",
            allow_promotion=True,
        ),
        parameter(
            "p2",
            Scalar,
            "Weibull shape parameter (dimensionless)",
            attr="p2",
            allow_promotion=True,
        ),
    )

    Y_in: Scalar
    p1: Scalar
    p2: Scalar

    def __post_init__(self) -> None:
        # First-order chain rule (d D / d r, plus promoted-param derivatives)
        # is auto-derived by reverse-mode autograd rather than hand-written.
        # The primal forward composes typed primitives (macaulay, exp, pow,
        # arithmetic) — each already differentiable — so an explicit action
        # would just duplicate what autograd computes anyway.
        self.request_AD()

    def forward(  # type: ignore[override]
        self,
        r: Scalar,
        *promoted_params,
    ) -> Scalar:
        Y_in = self._get_param("Y_in", promoted_params, Scalar)
        p1 = self._get_param("p1", promoted_params, Scalar)
        p2 = self._get_param("p2", promoted_params, Scalar)

        # arg = <r - Y_in>_+ / (p1 * Y_in)
        # Macaulay ensures D = 0 for r < Y_in (damage-onset gate).
        arg = macaulay(r - Y_in) / (p1 * Y_in)
        return Scalar.from_value(1.0, like=r) - exp(-pow(arg, p2))


__all__ = ["WeibullDamage"]
