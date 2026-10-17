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

"""Shared base for irreversible viscous damage updates."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import Scalar, gt, where
from ...chain_rule import ChainRuleDict
from ...model import Model


class ViscousDamageUpdate(Model, ABC):
    r"""Update a trial damage using $d=d_n+\alpha(d^*-d_n)$.

    Concrete subclasses define the time-integration coefficient ``alpha`` and
    its derivative with respect to the current time.
    """

    hit = HitSchema(
        input("trial_damage", Scalar, "Rate-independent trial damage"),
        output("damage", Scalar, "Updated irreversible damage"),
        derived_input("damage", Scalar, attr="_d_old_name", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t_name"),
        derived_input("time", Scalar, attr="_t_old_name", suffix="~1"),
        parameter(
            "viscosity",
            Scalar,
            "Damage viscosity, which must be non-negative. Zero recovers the "
            "rate-independent update exactly.",
            attr="eta",
            default="0",
        ),
    )

    eta: Scalar
    _d_old_name: str
    _t_name: str
    _t_old_name: str

    def __post_init__(self) -> None:
        if self.eta.ndim == 0 and self.eta.item() < 0.0:
            raise ValueError(f"{type(self).__name__}: viscosity must be non-negative")

    @abstractmethod
    def _coefficient(
        self, dt: Scalar, eta: Scalar, positive: Scalar, one: Scalar, zero: Scalar
    ) -> tuple[Scalar, Scalar]:
        """Return ``alpha`` and ``d(alpha)/d(dt)``."""

    def forward(  # type: ignore[override]
        self,
        d_trial: Scalar,
        d_old: Scalar,
        t: Scalar,
        t_old: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        eta = self._get_param("eta", promoted_params, Scalar)
        one = Scalar.from_value(1.0, like=d_trial)
        zero = Scalar.from_value(0.0, like=d_trial)
        advance = gt(d_trial, d_old)
        d_target = where(advance, d_trial, d_old)
        alpha, dalpha_dt = self._coefficient(t - t_old, eta, gt(eta, 0.0), one, zero)
        d = d_old + alpha * (d_target - d_old)
        if v is None:
            return d

        gap = d_target - d_old

        def trial_action(V: Scalar) -> Scalar:
            return alpha * where(advance, V, Scalar.zeros_like(V))

        def old_action(V: Scalar) -> Scalar:
            return (one - alpha) * V + alpha * where(advance, Scalar.zeros_like(V), V)

        def time_action(V: Scalar) -> Scalar:
            return dalpha_dt * gap * V

        return d, self.apply_chain_rule(
            v,
            "damage",
            {
                "trial_damage": trial_action,
                self._d_old_name: old_action,
                self._t_name: time_action,
                self._t_old_name: lambda V: -time_action(V),
            },
            output=d,
        )


__all__ = ["ViscousDamageUpdate"]
