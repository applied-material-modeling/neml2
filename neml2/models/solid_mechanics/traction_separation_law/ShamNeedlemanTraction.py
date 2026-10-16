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

"""Series-spring traction assembly for the simplified Sham-Needleman model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, lt, macaulay, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("ShamNeedlemanTraction")
class ShamNeedlemanTraction(Model):
    r"""Tractions for the simplified Sham-Needleman cohesive model, assembled
    from linear springs placed **in series** with the cavitation and sliding
    dashpots.

    Each jump splits additively, $[\![u]\!] = [\![u]\!]^e + [\![u]\!]^v$, and
    the spring acts on the elastic part only. Tension branch,
    $[\![u]\!]_n \ge 0$:

    $$
    T_n = K_n \left( [\![u]\!]_n - [\![u]\!]_n^v \right), \qquad
    T_t = K_t \left( [\![u]\!]_t - [\![u]\!]_t^v \right),
    $$

    with $[\![u]\!]_n^v = NV - N_0V_0$ supplied by
    :class:`~.CavityVolumeOpening.CavityVolumeOpening`. Compression,
    $[\![u]\!]_n < 0$, switches to a contact penalty $T_n = K_p [\![u]\!]_n$.

    The springs must be in series. A parallel spring would carry load forever
    and the interface would never separate completely.

    Once $f \ge f_F$ the interface has separated and carries

    $$
    T_n = K_p \min\left([\![u]\!]_n, 0\right), \qquad T_t = 0,
    $$

    so the compressive penalty survives separation while the tensile and
    shear load paths do not. Frictional sliding could replace $T_t = 0$ if a
    model needed it.

    Freezing the state after separation needs no extra machinery: every rate
    in this model is driven by $T_n$ or $T_t$, and on the separated branch the
    shear traction is zero while the normal traction is non-positive, so the
    sliding rate, the nucleation rate (through both of its factors) and the
    diffusive growth rate (through its Macaulay bracket) all vanish
    identically. $f$, $N$ and $[\![u]\!]_t^v$ therefore hold their values
    without being clamped.
    """

    hit = HitSchema(
        input("normal_jump", Scalar, "Signed normal jump across the interface"),
        input("tangential_jump", Scalar, "Signed tangential (sliding) jump"),
        input(
            "viscous_normal_opening",
            Scalar,
            "Viscous part of the normal jump, i.e. the smeared cavity volume",
        ),
        input("viscous_sliding", Scalar, "Viscous part of the sliding jump"),
        input(
            "cavity_fraction",
            Scalar,
            "Physical cavitated area fraction, tested against the failure fraction "
            "to detect separation",
        ),
        output("normal_traction", Scalar, "Normal traction"),
        output("shear_traction", Scalar, "Shear traction"),
        parameter(
            "normal_stiffness",
            Scalar,
            "Normal spring stiffness K_n",
            attr="K_n",
            allow_promotion=True,
        ),
        parameter(
            "shear_stiffness",
            Scalar,
            "Shear spring stiffness K_t",
            attr="K_t",
            allow_promotion=True,
        ),
        parameter(
            "penalty_stiffness",
            Scalar,
            "Compressive contact penalty stiffness K_p",
            attr="K_p",
            allow_promotion=True,
        ),
        parameter(
            "failure_fraction",
            Scalar,
            "Physical area fraction at final separation",
            attr="f_F",
            allow_promotion=True,
        ),
    )

    K_n: Scalar
    K_t: Scalar
    K_p: Scalar
    f_F: Scalar

    def forward(  # type: ignore[override]
        self,
        un: Scalar,
        ut: Scalar,
        unv: Scalar,
        utv: Scalar,
        f: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        K_n = self._get_param("K_n", promoted_params, Scalar)
        K_t = self._get_param("K_t", promoted_params, Scalar)
        K_p = self._get_param("K_p", promoted_params, Scalar)
        f_F = self._get_param("f_F", promoted_params, Scalar)

        zero = Scalar.from_value(0.0, like=un)

        # Detached branch masks. Both the separation test and the
        # tension/compression switch are kinks, so neither may contribute
        # through its own argument; `f` consequently has no action at all.
        separated = lt(f, f_F)  # True where the interface is still intact
        compress = lt(un, 0.0)

        # min(un, 0) written through the Macaulay bracket so the compressive
        # penalty is shared by the intact and separated branches: on the
        # intact compressive branch min(un, 0) == un, and on the separated
        # tensile branch it is zero.
        un_neg = -macaulay(-un)
        penalty = K_p * un_neg

        Tn_intact = where(compress, penalty, K_n * (un - unv))
        Tn = where(separated, Tn_intact, penalty)

        Tt_intact = K_t * (ut - utv)
        Tt = where(separated, Tt_intact, zero)

        if v is None:
            return Tn, Tt

        # ----- Normal traction partials.
        #   intact + tension:    dTn/dun = K_n, dTn/dunv = -K_n
        #   intact + compression and separated: dTn/dun = K_p where un < 0,
        #   zero elsewhere (the Macaulay kink), and no dependence on unv.
        dTn_dun_intact = where(compress, K_p, K_n)
        dTn_dun = where(separated, dTn_dun_intact, where(compress, K_p, zero))
        dTn_dunv_intact = where(compress, zero, -K_n)
        dTn_dunv = where(separated, dTn_dunv_intact, zero)

        # ----- Shear traction partials.
        dTt_dut = where(separated, K_t, zero)
        dTt_dutv = where(separated, -K_t, zero)

        normal_actions: dict[str, ChainRuleAction] = {
            "normal_jump": lambda V, c=dTn_dun: c * V,
            "viscous_normal_opening": lambda V, c=dTn_dunv: c * V,
        }
        shear_actions: dict[str, ChainRuleAction] = {
            "tangential_jump": lambda V, c=dTt_dut: c * V,
            "viscous_sliding": lambda V, c=dTt_dutv: c * V,
        }

        if "K_n" in self._promoted_params:
            dTn_dKn = where(separated, where(compress, zero, un - unv), zero)
            normal_actions[self._promoted_params["K_n"].input_name] = lambda V, c=dTn_dKn: c * V
        if "K_t" in self._promoted_params:
            dTt_dKt = where(separated, ut - utv, zero)
            shear_actions[self._promoted_params["K_t"].input_name] = lambda V, c=dTt_dKt: c * V
        if "K_p" in self._promoted_params:
            dTn_dKp = where(separated, where(compress, un_neg, zero), un_neg)
            normal_actions[self._promoted_params["K_p"].input_name] = lambda V, c=dTn_dKp: c * V

        v_n = self.apply_chain_rule(v, "normal_traction", normal_actions, output=Tn)
        v_t = self.apply_chain_rule(v, "shear_traction", shear_actions, output=Tt)
        return Tn, Tt, {**v_n, **v_t}


__all__ = ["ShamNeedlemanTraction"]
