"""Custom ``ProjectileAcceleration`` model — the running example for the
``extension`` tutorial chain. Mirrors the C++ tutorial's projectile in
the Python-native model surface.

The equation is

    a = g - mu * v

where ``v`` is the projectile velocity (input), ``a`` is the
acceleration (output), ``g`` is the gravitational acceleration vector
(buffer; constant), and ``mu`` is the scalar dynamic viscosity
(parameter; calibratable).
"""

from __future__ import annotations

from neml2.chain_rule import ChainRuleDict
from neml2.factory import register_neml2_object
from neml2.model import Model
from neml2.schema import HitSchema, buffer, input, output, parameter
from neml2.types import Scalar, Vec


@register_neml2_object("ProjectileAcceleration")
class ProjectileAcceleration(Model):
    """Newton's second law for a projectile in a viscous medium:
    ``a = g - mu * v``.
    """

    hit = HitSchema(
        input("velocity", Vec, "Velocity of the projectile", attr="_v_name"),
        output("acceleration", Vec, "Acceleration of the projectile"),
        buffer(
            "gravitational_acceleration",
            Vec,
            "Gravity vector",
            attr="g",
            default=Vec.fill(0.0, -9.81, 0.0),
        ),
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
        actions = {self._v_name: lambda V: -self.mu * V}

        # ``apply_chain_rule`` returns the v_out dict; pair it with the
        # value so the caller can unpack ``(a, v_out)``.
        return a, self.apply_chain_rule(v, "acceleration", actions, output=a)
