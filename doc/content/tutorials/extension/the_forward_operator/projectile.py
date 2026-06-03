"""Custom ``ProjectileAcceleration`` model — the running example for the
``extension`` tutorial chain. Mirrors the C++ tutorial's projectile in
the Python-native model surface.

The equation is

    a = g - mu * v

where ``v`` is the projectile velocity (input), ``a`` is the
acceleration (output), ``g`` is the gravitational acceleration vector
(parameter), and ``mu`` is the scalar dynamic viscosity (parameter).
"""

from __future__ import annotations

from neml2.chain_rule import ChainRuleDict
from neml2.factory import register_native
from neml2.model import Model
from neml2.schema import HitSchema, input, output, parameter
from neml2.types import Scalar, Vec


@register_native("ProjectileAcceleration")
class ProjectileAcceleration(Model):
    """Newton's second law for a projectile in a viscous medium:
    ``a = g - mu * v``.
    """

    hit = HitSchema(
        input("velocity", Vec, "Velocity of the projectile", attr="_v_name"),
        output("acceleration", Vec, "Acceleration of the projectile"),
        parameter("gravitational_acceleration", Vec, "Gravity vector", attr="g"),
        parameter("dynamic_viscosity", Scalar, "Dynamic viscosity", attr="mu"),
    )

    _v_name: str
    g: Vec
    mu: Scalar

    def forward(  # type: ignore[override]
        self,
        v_in: Vec,
        *,
        v: ChainRuleDict | None = None,
        v2=None,
        vh=None,
    ):
        # Compute the value: a = g - mu * v.
        a = self.g - self.mu * v_in

        # Pure forward: return the typed output and stop.
        if v is None:
            return a

        # First-order chain rule: ∂a / ∂v_in = -mu * I. The closure
        # captures ``self.mu`` and receives an incoming tangent V
        # (a ``Vec`` shaped like the input), returns the contribution
        # to ∂(acceleration)/∂(seed-leaf).
        actions_1 = {self._v_name: lambda V: -self.mu * V}

        # ``propagate_tangents`` dispatches v / v2 / vh through
        # ``actions_1`` and returns the right-length tuple. This leaf
        # is linear in ``v_in`` so the Hessian vanishes — no
        # ``actions_2`` to pass.
        return a, *self.propagate_tangents(v, "acceleration", actions_1, output=a, v2=v2, vh=vh)
