# A minimal NEML2 Model subclass illustrating the input-file wiring.
#
# Forward operator: gravity-plus-linear-drag projectile acceleration,
#     a = g - mu * v
# with `g` a Vec buffer (constant gravity) and `mu` a single Scalar
# parameter. The point of this file is the *connection to HIT* — the
# schema declaration and the `@register_neml2_object` decorator — not the
# physics.

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.model import Model
from neml2.schema import HitSchema, buffer, input, output, parameter
from neml2.types import Scalar, Vec


@register_neml2_object("ProjectileAcceleration")
class ProjectileAcceleration(Model):
    """Projectile acceleration under gravity and linear drag: a = g - mu * v."""

    hit = HitSchema(
        input("velocity", Vec, "Projectile velocity"),
        output("acceleration", Vec, "Projectile acceleration"),
        parameter("dynamic_viscosity", Scalar, "Drag coefficient mu", attr="mu"),
        buffer(
            "gravity",
            Vec,
            "Gravitational acceleration vector",
            attr="g",
            default=Vec.fill(0.0, -9.81, 0.0),
        ),
    )

    def forward(self, velocity, *nl_params, v=None):
        raise NotImplementedError
