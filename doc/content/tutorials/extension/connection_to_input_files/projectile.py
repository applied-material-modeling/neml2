# A minimal NEML2 Model subclass illustrating the input-file wiring.
#
# Forward operator: linear-drag deceleration of a projectile,
#     a = -mu * v
# with `mu` a single Scalar parameter. The point of this file is the
# *connection to HIT* — the schema declaration and the
# `@register_native` decorator — not the physics.

from __future__ import annotations

from neml2.chain_rule import ChainRuleDict
from neml2.factory import register_native
from neml2.model import Model
from neml2.schema import HitSchema, input, output, parameter
from neml2.types import Scalar


@register_native("ProjectileAcceleration")
class ProjectileAcceleration(Model):
    """Linear-drag projectile acceleration: a = -mu * v."""

    hit = HitSchema(
        input("velocity", Scalar, "Projectile velocity"),
        output("acceleration", Scalar, "Projectile acceleration"),
        parameter("dynamic_viscosity", Scalar, "Drag coefficient mu", attr="mu"),
    )

    # Annotate the parameter attribute so static checkers see the
    # typed wrapper that Model.__getattr__ returns.
    mu: Scalar

    def forward(  # type: ignore[override]
        self,
        v: Scalar,
        *nl_params: Scalar,
        v_jvp: ChainRuleDict | None = None,
    ):
        mu = self._get_param("mu", nl_params, Scalar)
        a = -mu * v

        if v_jvp is None:
            return a

        in_name = next(iter(self.input_spec))
        out_name = next(iter(self.output_spec))
        actions = {in_name: lambda V, c=-mu: c * V}
        return a, self.apply_chain_rule(v_jvp, out_name, actions, output=a)
