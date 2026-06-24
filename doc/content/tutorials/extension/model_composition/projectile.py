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

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
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
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Read the drag coefficient through ``_get_param`` rather than
        # ``self.mu``. ``_get_param`` resolves a static slot from ``self`` or
        # a promoted runtime input from ``*nl_params``, so the same forward
        # keeps working after ``mu`` is promoted (neml2-compile -p); a bare
        # ``self.mu`` would be rejected by the parameter-attribute guard.
        mu = self._get_param("mu", nl_params, Scalar)

        # Compute the value: a = g - mu * v. ``self.g`` is a buffer (not a
        # parameter), so reading it directly is fine.
        a = self.g - mu * v_in

        # Pure forward: return the typed output and stop.
        if v is None:
            return a

        # First-order chain rule: ∂a / ∂v_in = -mu * I. The closure
        # captures the local ``mu`` and receives an incoming tangent V
        # (a ``Vec`` shaped like the input), returns the contribution
        # to ∂(acceleration)/∂(seed-leaf).
        actions = {self._v_name: lambda V: -mu * V}

        # ``apply_chain_rule`` returns the v_out dict; pair it with the
        # value so the caller can unpack ``(a, v_out)``.
        return a, self.apply_chain_rule(v, "acceleration", actions, output=a)
