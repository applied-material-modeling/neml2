# A minimal NEML2 Model subclass illustrating the input-file wiring.
#
# Forward operator: gravity-plus-linear-drag projectile acceleration,
#     a = g - mu * v
# with `g` a Vec buffer (constant gravity) and `mu` a single Scalar
# parameter. The point of this file is the *connection to HIT* — the
# schema declaration and the `@register_native` decorator — not the
# physics.

from __future__ import annotations

from neml2.chain_rule import ChainRuleDict
from neml2.factory import register_native
from neml2.model import Model
from neml2.schema import HitSchema, buffer, input, output, parameter
from neml2.types import Scalar, Vec


@register_native("ProjectileAcceleration")
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

    # Annotate the typed attributes so static checkers see the wrappers
    # that Model.__getattr__ returns.
    mu: Scalar
    g: Vec

    def forward(  # type: ignore[override]
        self,
        v: Vec,
        *nl_params: Scalar,
        v_jvp: ChainRuleDict | None = None,
    ):
        mu = self._get_param("mu", nl_params, Scalar)
        a = self.g - mu * v

        if v_jvp is None:
            return a

        in_name = next(iter(self.input_spec))
        out_name = next(iter(self.output_spec))
        actions = {in_name: lambda V, c=-mu: c * V}
        return a, self.apply_chain_rule(v_jvp, out_name, actions, output=a)
